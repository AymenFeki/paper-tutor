"""Measure retrieval quality of the active embedding model on eval/questions.yaml.

For each question: encode it, take the top 10 papers by cosine distance (same query as
06_search.py, plus the paper id) and record the rank of the relevant paper.
Reports hit@1/5/10 and MRR overall and per area, a Wilson interval for hit@5, and the misses.
"""

import json
from collections import defaultdict
from pathlib import Path

import yaml
from pgvector.psycopg import register_vector

from paper_tutor.corpus import load_config
from paper_tutor.db import connect
from paper_tutor.embed import active_model, encode_query, load_model
from paper_tutor.evaluation import HITS_AT, first_relevant_rank, metrics, wilson_interval

TOP_K = 10

SEARCH = """
    SELECT p.id, p.title, t.area,
           1 - (e.embedding <=> %(q)s) AS similarity
    FROM embeddings e
    JOIN papers p ON p.id = e.paper_id
    JOIN topics t ON t.id = p.topic_id
    WHERE e.model = %(model)s
    ORDER BY e.embedding <=> %(q)s
    LIMIT %(k)s
"""

config = load_config()
model_key, model_cfg = active_model(config)
questions = yaml.safe_load(Path("eval/questions.yaml").read_text())

model = load_model(model_cfg)

results = []
with connect() as conn:
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM embeddings WHERE model = %s", (model_key,))
        print(f"\nModel: {model_key} ({model_cfg['name']}), {cur.fetchone()[0]} papers embedded")

        for item in questions:
            query_vector = encode_query(model, model_cfg, item["question"])
            cur.execute(SEARCH, {"q": query_vector, "model": model_key, "k": TOP_K})
            retrieved = [
                {"id": pid, "title": title, "area": area, "similarity": round(float(sim), 4)}
                for pid, title, area, sim in cur.fetchall()
            ]
            results.append({
                "question": item["question"],
                "area": item["area"],
                "relevant": item["relevant"],
                "paper_title": item["paper_title"],
                "rank": first_relevant_rank([r["id"] for r in retrieved], set(item["relevant"])),
                "retrieved": retrieved,
            })

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

# Questions where the relevant paper is not in the top 5
misses = [r for r in results if r["rank"] is None or r["rank"] > 5]
print(f"\n{len(misses)} questions with the relevant paper not in the top 5:")
for r in misses:
    rank = r["rank"] if r["rank"] is not None else f">{TOP_K}"
    print(f"\n[{r['area']}] {r['question']}")
    print(f"  expected (rank {rank}): {r['paper_title']}")
    for i, hit in enumerate(r["retrieved"][:5], start=1):
        print(f"  {i}. {hit['title']} [{hit['area']}] ({hit['similarity']:.3f})")

out_path = Path("eval/results") / f"{model_key}.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump({
        "model": model_key,
        "model_name": model_cfg["name"],
        "top_k": TOP_K,
        "overall": overall,
        "hit@5_wilson_95": [round(low, 4), round(high, 4)],
        "per_area": per_area,
        "questions": results,
    }, f, indent=2, ensure_ascii=False)
print(f"\nSaved {out_path}")
