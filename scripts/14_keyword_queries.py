"""Compare relevance-threshold rules on long questions and short keyword queries (#22).

Four groups of queries: long in-scope questions (eval/questions.yaml), long out-of-scope
questions (eval/out_of_scope.yaml), and short in-scope / off-topic keyword queries
(eval/keyword_queries.yaml). For each query the 30 vector candidates get both scores
(embedding similarity and reranker score). Then every rule is checked: a query counts as
answered if at least one candidate passes the threshold (rag.passes_threshold, the same
function the retriever uses). Saves the scores to eval/keyword_query_scores.json.
"""

import json
from pathlib import Path

import yaml

from paper_tutor.corpus import load_config
from paper_tutor.rag import build_retriever, is_short_query, passes_threshold

config = load_config()
settings = config["retrieval"]
config["retrieval"] = {**settings, "rerank": True, "threshold": False, "hybrid": False}  # both scores, no cut-off
retrieve = build_retriever(config)

keyword_queries = yaml.safe_load(Path("eval/keyword_queries.yaml").read_text())
groups = {
    "long, in scope": [q["question"] for q in yaml.safe_load(Path("eval/questions.yaml").read_text())],
    "long, off topic": yaml.safe_load(Path("eval/out_of_scope.yaml").read_text()),
    "short, in scope": keyword_queries["in_scope"],
    "short, off topic": keyword_queries["off_topic"],
}
candidates = {group: {q: retrieve(q, k=settings["candidates"]) for q in queries} for group, queries in groups.items()}

print(f"\n{'group':<18}{'n':>4}{'words':>8}{'best similarity':>22}{'best reranker score':>24}")
scores = {}
for group, results in candidates.items():
    rows = [{"query": q, "words": len(q.split()),
             "similarity": max(p["similarity"] for p in papers),
             "rerank_score": max(p["rerank_score"] for p in papers)} for q, papers in results.items()]
    scores[group] = rows
    words = [r["words"] for r in rows]
    sims = [r["similarity"] for r in rows]
    rerank = [r["rerank_score"] for r in rows]
    print(f"{group:<18}{len(rows):>4}{min(words):>4}-{max(words):<3}"
          f"{min(sims):>12.3f} to {max(sims):.3f}{min(rerank):>14.4f} to {max(rerank):.4f}")

rules = {
    f"similarity >= {settings['min_similarity']} for all queries (before)": {"short_query_words": 0},
    f"reranker >= {settings['min_rerank_score_short']} for all queries": {"short_query_words": 1000},
    f"<= {settings['short_query_words']} words: reranker, longer: similarity (now)": {},
}
print(f"\n{'rule':<56}" + "".join(f"{group:>18}" for group in groups))
for name, change in rules.items():
    rule = {**settings, "rerank": False, **change}
    cells = []
    for group, results in candidates.items():
        answered = sum(
            any(passes_threshold(p, is_short_query(q, rule["short_query_words"]), rule) for p in papers)
            for q, papers in results.items()
        )
        good = answered if "in scope" in group else len(results) - answered
        cells.append(f"{good}/{len(results)} {'answered' if 'in scope' in group else 'refused'}")
    print(f"{name:<56}" + "".join(f"{cell:>18}" for cell in cells))

Path("eval/keyword_query_scores.json").write_text(json.dumps(scores, indent=2, ensure_ascii=False))
print("\nSaved eval/keyword_query_scores.json")
