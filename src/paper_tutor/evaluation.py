"""Retrieval metrics shared by the evaluation scripts."""

import math

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
