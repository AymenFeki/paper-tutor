"""Measure retrieval quality on eval/questions.yaml with the retrieval settings in the config.

For each question: retrieve the top 10 papers with the same retriever the API uses and record
the rank of the relevant paper. Reports hit@1/5/10 and MRR overall and per area, a Wilson
interval for hit@5, and the misses. Also runs eval/out_of_scope.yaml and the short queries in
eval/keyword_queries.yaml and counts which get no papers (only possible with the threshold on).

Retrieval settings can be overridden for one run, so variants can be compared without
editing the config, e.g.:
    uv run python scripts/07_evaluate.py hybrid=true rerank=true
The result is saved as eval/results/<model>[+hybrid][+rerank][+threshold].json.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from paper_tutor.corpus import load_config
from paper_tutor.embed import active_model
from paper_tutor.evaluation import HITS_AT, first_relevant_rank, metrics, wilson_interval
from paper_tutor.rag import build_retriever

TOP_K = 10

config = load_config()
for override in sys.argv[1:]:
    key, value = override.split("=")
    config["retrieval"][key] = yaml.safe_load(value)  # "true" -> True, "0.5" -> 0.5

settings = config["retrieval"]
model_key, model_cfg = active_model(config)
name = model_key + "".join(f"+{step}" for step in ("hybrid", "rerank", "threshold") if settings[step])
questions = yaml.safe_load(Path("eval/questions.yaml").read_text())
out_of_scope = yaml.safe_load(Path("eval/out_of_scope.yaml").read_text())
keyword_queries = yaml.safe_load(Path("eval/keyword_queries.yaml").read_text())

retrieve = build_retriever(config)
print(f"\n{name}: {model_cfg['name']}, retrieval settings {settings}")

results = []
for item in questions:
    retrieved = [
        {"id": p["id"], "title": p["title"], "area": p["area"],
         "similarity": round(p["similarity"], 4), "score": round(p["score"], 4)}
        for p in retrieve(item["question"], k=TOP_K)
    ]
    results.append({
        "question": item["question"],
        "area": item["area"],
        "relevant": item["relevant"],
        "paper_title": item["paper_title"],
        "rank": first_relevant_rank([r["id"] for r in retrieved], set(item["relevant"])),
        "retrieved": retrieved,
    })

oos_results = []
for question in out_of_scope:
    papers = retrieve(question, k=TOP_K)
    oos_results.append({
        "question": question,
        "n_papers": len(papers),
        "top_score": round(papers[0]["score"], 4) if papers else None,
        "top_title": papers[0]["title"] if papers else None,
    })

keywords_answered = sum(bool(retrieve(q, k=TOP_K)) for q in keyword_queries["in_scope"])
keywords_refused = sum(not retrieve(q, k=TOP_K) for q in keyword_queries["off_topic"])

# Overall and per-area metrics
by_area = defaultdict(list)
for r in results:
    by_area[r["area"]].append(r["rank"])
overall = metrics([r["rank"] for r in results])
per_area = {area: metrics(ranks) for area, ranks in by_area.items()}

hits5 = sum(r["rank"] is not None and r["rank"] <= 5 for r in results)
low, high = wilson_interval(hits5, len(results))

header = f"{'area':<22}{'n':>4}" + "".join(f"{'hit@' + str(k):>8}" for k in HITS_AT) + f"{'MRR':>8}"
print(f"\n{header}\n{'-' * len(header)}")
for area, m in list(per_area.items()) + [("overall", overall)]:
    print(f"{area:<22}{m['n']:>4}" + "".join(f"{m[f'hit@{k}']:>8.2f}" for k in HITS_AT) + f"{m['mrr']:>8.3f}")

print(f"\nhit@5 = {hits5}/{len(results)} = {hits5 / len(results):.2f}, "
      f"95% Wilson CI [{low:.2f}, {high:.2f}]")

no_papers = sum(r["n_papers"] == 0 for r in oos_results)
in_scope_empty = sum(not r["retrieved"] for r in results)
print(f"out of scope: {no_papers}/{len(oos_results)} questions got no papers; "
      f"in scope: {in_scope_empty}/{len(results)} questions got no papers")
print(f"short queries: {keywords_answered}/{len(keyword_queries['in_scope'])} in scope answered, "
      f"{keywords_refused}/{len(keyword_queries['off_topic'])} off topic refused")

# Questions where the relevant paper is not in the top 5
misses = [r for r in results if r["rank"] is None or r["rank"] > 5]
print(f"\n{len(misses)} questions with the relevant paper not in the top 5:")
for r in misses:
    rank = r["rank"] if r["rank"] is not None else f">{TOP_K}"
    print(f"\n[{r['area']}] {r['question']}")
    print(f"  expected (rank {rank}): {r['paper_title']}")
    for i, hit in enumerate(r["retrieved"][:5], start=1):
        print(f"  {i}. {hit['title']} [{hit['area']}] ({hit['score']:.3f})")

out_path = Path("eval/results") / f"{name}.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump({
        "model": name,
        "model_name": model_cfg["name"],
        "retrieval": settings,
        "top_k": TOP_K,
        "overall": overall,
        "hit@5_wilson_95": [round(low, 4), round(high, 4)],
        "per_area": per_area,
        "out_of_scope_no_papers": no_papers,
        "keyword_queries": {"in_scope_answered": keywords_answered, "in_scope": len(keyword_queries["in_scope"]),
                            "off_topic_refused": keywords_refused, "off_topic": len(keyword_queries["off_topic"])},
        "questions": results,
        "out_of_scope": oos_results,
    }, f, indent=2, ensure_ascii=False)
print(f"\nSaved {out_path}")
