"""Data-quality checks on the papers in the database. Report only: nothing is changed or deleted.

(a) #19 reviews, surveys, tutorials and lecture notes, found by title and abstract patterns
(b) #20 title-abstract mismatch: similarity between the title embedding and the abstract
    embedding; the lowest 1% are flagged
(c) #21 papers whose embedding is closer to another area's centroid than to their own area's
(d) #20 arXiv papers whose OpenAlex title differs from arXiv's own title (corrupted metadata;
    uses data/raw/arxiv_titles.json from 02_fetch.py)

Prints counts and 10 examples per check and writes all flagged papers to eval/data_checks/*.csv.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pgvector.psycopg import register_vector

from paper_tutor.checks import area_centroids, cosine_rows, lowest_share, nearest_area, review_signals
from paper_tutor.corpus import has_wrong_arxiv_title, load_config
from paper_tutor.db import connect
from paper_tutor.embed import active_model, load_model

LOWEST_SHARE = 0.01

PAPERS = """
    SELECT p.id, p.doi, p.title, p.abstract, p.type, t.area, e.embedding
    FROM papers p
    JOIN topics t ON t.id = p.topic_id
    JOIN embeddings e ON e.paper_id = p.id AND e.model = %s
    ORDER BY p.id
"""

config = load_config()
model_key, model_cfg = active_model(config)
with connect() as conn:
    register_vector(conn)
    rows = conn.execute(PAPERS, (model_key,)).fetchall()
papers = pd.DataFrame(rows, columns=["id", "doi", "title", "abstract", "type", "area", "embedding"])
vectors = np.array([v.to_numpy() for v in papers.pop("embedding")])

questions = yaml.safe_load(Path("eval/questions.yaml").read_text())
eval_ids = {pid for q in questions for pid in q["relevant"]}
out_dir = Path("eval/data_checks")
out_dir.mkdir(parents=True, exist_ok=True)
print(f"{len(papers)} papers, embeddings from {model_key}")


def report(name, flagged, columns, sort_by=None, ascending=True):
    """Print the count and 10 examples, and write all flagged papers to a CSV file."""
    if sort_by:
        flagged = flagged.sort_values(sort_by, ascending=ascending)
    n_eval = flagged["id"].isin(eval_ids).sum()
    print(f"\n{name}: {len(flagged)} papers ({len(flagged) / len(papers):.1%}), "
          f"{n_eval} of them are relevant papers in eval/questions.yaml")
    print(flagged["area"].value_counts().to_string())
    examples = flagged.head(10) if sort_by else flagged.sample(min(10, len(flagged)), random_state=0)
    for _, row in examples.iterrows():
        print("  " + " | ".join(str(row[c])[:70] for c in columns))
    flagged.drop(columns=["abstract"], errors="ignore").to_csv(out_dir / f"{name}.csv", index=False)


# (a) reviews and tutorials
papers["signals"] = [", ".join(review_signals(t, a)) for t, a in zip(papers["title"], papers["abstract"])]
reviews = papers[papers["signals"] != ""]
print(f"\nOpenAlex type 'review': {(papers['type'] == 'review').sum()} papers")
report("reviews_and_tutorials", reviews, ["id", "area", "title", "signals"])

# (b) title-abstract mismatch
model = load_model(model_cfg)
titles = model.encode(papers["title"].fillna("").tolist(), normalize_embeddings=True, batch_size=64)
abstracts = model.encode(papers["abstract"].fillna("").tolist(), normalize_embeddings=True, batch_size=64)
papers["title_abstract_similarity"] = cosine_rows(titles, abstracts).astype(float).round(3)
lowest = papers.iloc[lowest_share(papers["title_abstract_similarity"].to_numpy(), LOWEST_SHARE)]
print(f"\nlowest 1% title-abstract similarity: below {lowest['title_abstract_similarity'].max():.3f} "
      f"(median of all papers: {papers['title_abstract_similarity'].median():.3f})")
known = papers[papers["id"] == "W2040503026"]  # the example in issue #20
if len(known):
    similarity = known["title_abstract_similarity"].iloc[0]
    rank = (papers["title_abstract_similarity"] < similarity).sum() + 1
    print(f"W2040503026 (issue #20): similarity {similarity:.3f}, rank {rank} from the lowest")
lowest = lowest.assign(abstract_start=lowest["abstract"].str[:120].str.rstrip())
report("title_abstract_mismatch", lowest, ["id", "title_abstract_similarity", "title", "abstract_start"],
       sort_by="title_abstract_similarity")

# (c) closer to another area's centroid
centroids = area_centroids(vectors, papers["area"].tolist())
nearest, margins = [], []
for vector, area in zip(vectors, papers["area"]):
    best, similarities = nearest_area(vector, centroids)
    nearest.append(best)
    margins.append(round(similarities[best] - similarities[area], 3))
papers["nearest_area"], papers["margin"] = nearest, margins
wrong_area = papers[papers["nearest_area"] != papers["area"]]
print("\nown area -> nearest area (most common):")
print(wrong_area.groupby(["area", "nearest_area"]).size().sort_values(ascending=False).head(8).to_string())
report("closer_to_other_area", wrong_area, ["id", "area", "nearest_area", "margin", "title"],
       sort_by="margin", ascending=False)

# (d) wrong arXiv title
arxiv_titles = json.loads(Path("data/raw/arxiv_titles.json").read_text())
wrong = [has_wrong_arxiv_title({"doi": doi, "title": title}, arxiv_titles)
         for doi, title in zip(papers["doi"].fillna(""), papers["title"])]
report("wrong_arxiv_title", papers[wrong], ["id", "area", "doi", "title"])

print(f"\nWrote CSV files to {out_dir}/")
