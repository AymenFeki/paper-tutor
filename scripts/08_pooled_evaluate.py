"""Recompute retrieval metrics with pooled relevance judgments.

Reads every result file saved by 07_evaluate.py in eval/results/ and the judgments in
eval/judgments.yaml (top-5 of bge-small and qwen3-0.6b plus the original paper, judged by an
LLM, not fully blind). Prints hit@5 and MRR with the original relevant paper(s) ("single") and
with all papers judged relevant ("pooled").

Papers that were never judged count as not relevant, so pooled numbers are a lower bound.
This matters for new retrieval variants: they can find papers that nobody judged, so the
"unjudged" column shows how many top-5 papers are missing a judgment. Questions with no
judgments at all (added after the pool was judged) fall back to their own relevant list.
"""

import json
from collections import defaultdict
from pathlib import Path

import yaml

from paper_tutor.evaluation import first_relevant_rank, metrics, wilson_interval

POOL_DEPTH = 5

result_files = sorted(Path("eval/results").glob("*.json"))
results = {path.stem: json.loads(path.read_text()) for path in result_files}
judgments = yaml.safe_load(Path("eval/judgments.yaml").read_text())

judged = defaultdict(dict)  # question -> paper_id -> relevant
for j in judgments:
    judged[j["question"]][j["paper_id"]] = j["relevant"]


def evaluate(questions, relevant_for):
    """Ranks, metrics and hit@5 Wilson interval for one result file under one relevance definition."""
    ranks = [first_relevant_rank([r["id"] for r in q["retrieved"]], relevant_for(q))
             for q in questions]
    m = metrics(ranks)
    hits5 = sum(r is not None and r <= 5 for r in ranks)
    m["ci"] = wilson_interval(hits5, len(ranks))
    m["hits5"] = hits5
    return m


def single(q):
    return set(q["relevant"])


def pooled(q):
    if q["question"] not in judged:
        return set(q["relevant"])
    return {pid for pid, rel in judged[q["question"]].items() if rel}


def count_unjudged(questions):
    """Number of top-5 papers without a judgment, over the questions that were judged."""
    return sum(
        r["id"] not in judged[q["question"]]
        for q in questions if q["question"] in judged
        for r in q["retrieved"][:POOL_DEPTH]
    )


n_questions = len(judged)
n_relevant = [sum(papers.values()) for papers in judged.values()]
print(f"\n{n_questions} judged questions, {len(judgments)} judged papers, "
      f"{sum(n_relevant)} relevant ({sum(n_relevant) / n_questions:.1f} per question, "
      f"min {min(n_relevant)}, max {max(n_relevant)})")

header = (f"{'result':<32}{'relevance':<10}{'hit@5':>7}  {'95% Wilson CI':<15}{'MRR':>7}"
          f"{'unjudged top-5':>16}")
print(f"\n{header}\n{'-' * len(header)}")
for key, res in results.items():
    unjudged = count_unjudged(res["questions"])
    for label, relevant_for in (("single", single), ("pooled", pooled)):
        m = evaluate(res["questions"], relevant_for)
        low, high = m["ci"]
        extra = f"{unjudged:>16}" if label == "pooled" else ""
        print(f"{key:<32}{label:<10}{m['hits5']:>3}/{m['n']:<3}  [{low:.2f}, {high:.2f}]   {m['mrr']:>7.3f}{extra}")

print("\nPooled numbers are a lower bound: unjudged papers count as not relevant.")
print("A fair comparison of new variants needs new judgments for their unjudged top-5 papers.")

# Original papers that the judgment did not mark as relevant
for q in yaml.safe_load(Path("eval/questions.yaml").read_text()):
    for pid in q["relevant"]:
        if q["question"] in judged and not judged[q["question"]].get(pid, False):
            print(f"\nNote: original paper {pid} judged not relevant for: {q['question']}")
