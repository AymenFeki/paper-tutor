"""Recompute retrieval metrics with pooled relevance judgments.

Reads the per-question results saved by 07_evaluate.py for each model and the judgments in
eval/judgments.yaml (top-5 of all models plus the original paper, judged blind). Prints hit@5
and MRR with the original single relevant paper and with all papers judged relevant.

Papers ranked 6-10 that were not in the pool are unjudged and count as not relevant,
so pooled MRR is a lower bound; hit@5 is fully judged.
"""

import json
from collections import defaultdict
from pathlib import Path

import yaml

from paper_tutor.corpus import load_config
from paper_tutor.evaluation import first_relevant_rank, metrics, wilson_interval

POOL_DEPTH = 5

config = load_config()
model_keys = [key for key in config["embeddings"]["models"]
              if Path(f"eval/results/{key}.json").exists()]
results = {key: json.load(open(f"eval/results/{key}.json")) for key in model_keys}
judgments = yaml.safe_load(open("eval/judgments.yaml"))

judged = defaultdict(dict)  # question -> paper_id -> relevant
for j in judgments:
    judged[j["question"]][j["paper_id"]] = j["relevant"]


def evaluate(questions, relevant_for):
    """Ranks, metrics and hit@5 Wilson interval for one model under one relevance definition."""
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
    return {pid for pid, rel in judged[q["question"]].items() if rel}


# Check that every pooled paper was judged
for key, res in results.items():
    for q in res["questions"]:
        missing = [r["id"] for r in q["retrieved"][:POOL_DEPTH] if r["id"] not in judged[q["question"]]]
        if missing:
            raise SystemExit(f"{key}: unjudged top-{POOL_DEPTH} papers for: {q['question']} {missing}")

n_questions = len(judged)
n_relevant = [sum(papers.values()) for papers in judged.values()]
print(f"\n{n_questions} questions, {len(judgments)} judged papers, "
      f"{sum(n_relevant)} relevant ({sum(n_relevant) / n_questions:.1f} per question, "
      f"min {min(n_relevant)}, max {max(n_relevant)})")

header = f"{'model':<14}{'relevance':<10}{'hit@5':>7}  {'95% Wilson CI':<15}{'MRR':>7}"
print(f"\n{header}\n{'-' * len(header)}")
for key, res in results.items():
    for label, relevant_for in (("single", single), ("pooled", pooled)):
        m = evaluate(res["questions"], relevant_for)
        low, high = m["ci"]
        print(f"{key:<14}{label:<10}{m['hits5']:>3}/{m['n']:<3}  [{low:.2f}, {high:.2f}]   {m['mrr']:>7.3f}")

# Original papers that the blind judgment did not mark as relevant
for q in next(iter(results.values()))["questions"]:
    for pid in q["relevant"]:
        if not judged[q["question"]].get(pid, False):
            print(f"\nNote: original paper {pid} judged not relevant for: {q['question']}")
