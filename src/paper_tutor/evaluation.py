"""Retrieval and answer metrics shared by the evaluation scripts."""

import math
import re

import yaml

HITS_AT = (1, 5, 10)


def first_relevant_rank(retrieved_ids, relevant):
    """1-based rank of the first relevant paper in the result list, or None."""
    for rank, paper_id in enumerate(retrieved_ids, start=1):
        if paper_id in relevant:
            return rank
    return None


def metrics(ranks):
    """hit@k and MRR for a list of ranks (None = no relevant paper in the result list)."""
    n = len(ranks)
    result = {f"hit@{k}": sum(r is not None and r <= k for r in ranks) / n for k in HITS_AT}
    result["mrr"] = sum(1 / r for r in ranks if r is not None) / n
    result["n"] = n
    return result


def wilson_interval(successes, n, z=1.96):
    """95% Wilson score interval for a binomial proportion."""
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def threshold_effect(questions, out_of_scope, min_score):
    """What a minimum score would do, computed from results saved with the threshold off.

    questions: in-scope results from 07_evaluate.py (with "rank" and scored "retrieved" papers).
    out_of_scope: out-of-scope results (with "top_score").
    Returns how many in-scope questions would get no papers, how many top-5 hits would survive,
    and how many out-of-scope questions would get no papers (the goal).
    """
    in_scope_refused = sum(q["retrieved"][0]["score"] < min_score for q in questions)
    hits5_kept = sum(
        q["rank"] is not None and q["rank"] <= 5 and q["retrieved"][q["rank"] - 1]["score"] >= min_score
        for q in questions
    )
    out_of_scope_refused = sum(q["top_score"] < min_score for q in out_of_scope)
    return {
        "in_scope_refused": in_scope_refused,
        "hits5_kept": hits5_kept,
        "out_of_scope_refused": out_of_scope_refused,
    }


def split_sentences(text):
    """Split an answer into sentences, keeping a trailing citation like "... model. [2]" with its sentence.

    Rough on purpose: a new sentence starts after ".", "!", "?" or "]" followed by a space and a
    capital letter, and every line is split separately. Bullet markers are removed.
    """
    sentences = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        sentences.extend(s for s in re.split(r"(?<=[.!?\]])\s+(?=[A-Z])", line) if s)
    return sentences


def citations(sentence):
    """Source numbers cited in a sentence: "[1]", "[2][4]" and "[1, 3]" all count. In order, no repeats."""
    numbers = []
    for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", sentence):
        for number in group.split(","):
            if int(number) not in numbers:
                numbers.append(int(number))
    return numbers


def strip_citations(sentence):
    """The sentence without its citation brackets, for the judge."""
    return re.sub(r"\s*\[\d+(?:\s*,\s*\d+)*\]", "", sentence).strip()


def unjudged_pairs(results, judgments, depth=5):
    """(question, paper) pairs from the top `depth` of the results that nobody has judged yet.

    results: result dictionaries saved by 07_evaluate.py; judgments: the list in eval/judgments.yaml.
    The paper a question was written for is skipped (it is relevant by construction). Each pair
    appears once; pairs are grouped by question and sorted by paper id inside a question, so
    their order does not reveal which retrieval variant found them or at which rank.
    """
    judged = {(j["question"], j["paper_id"]) for j in judgments}
    pairs = {}
    for result in results:
        for q in result["questions"]:
            for paper in q["retrieved"][:depth]:
                key = (q["question"], paper["id"])
                if key not in judged and paper["id"] not in q["relevant"]:
                    pairs[key] = {"question": q["question"], "paper_id": paper["id"], "title": paper["title"]}
    question_order = {}
    for question, _ in pairs:
        question_order.setdefault(question, len(question_order))
    return sorted(pairs.values(), key=lambda p: (question_order[p["question"]], p["paper_id"]))


def judgment_yaml(question, paper_id, relevant):
    """One human judgment as YAML text, ready to append to eval/judgments.yaml."""
    entry = {"question": question, "paper_id": paper_id, "relevant": relevant, "judge": "human"}
    return yaml.safe_dump([entry], sort_keys=False, allow_unicode=True, width=1000)
