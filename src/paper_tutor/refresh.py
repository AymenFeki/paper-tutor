"""Weekly refresh: fetch recently published papers from OpenAlex and add the ones that pass the rules."""

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pyalex
from dotenv import load_dotenv
from pyalex import Works

from paper_tutor.corpus import author_names, deduplicate, is_known, passes_rules, rebuild_abstract, select_top
from paper_tutor.db import connect, known_papers, save_papers
from paper_tutor.embed import embed_missing, embedding_text, encode_query

load_dotenv()
pyalex.config.api_key = os.getenv("OPENALEX_API_KEY")

FIELDS = [
    "id", "doi", "title", "publication_year", "type", "language",
    "is_retracted", "cited_by_count", "fwci", "citation_normalized_percentile",
    "abstract_inverted_index", "primary_location", "best_oa_location", "primary_topic",
    "authorships",
]

RAW_DIR = Path("data/raw")


def fetch_recent(config, days=14):
    """Papers published in the last `days` days for every active topic area, as (paper, topic_id) pairs.

    OpenAlex already filters out what the credibility rules would drop anyway (no abstract, not
    English, wrong type, retracted), and every page is fetched, so no topic is cut off.
    """
    settings = config["settings"]
    since = (datetime.now(UTC).date() - timedelta(days=days)).isoformat()
    records = []
    for area in config["areas"].values():
        if not area["active"]:
            continue
        for topic_id in area["topics"]:
            query = (
                Works()
                .filter(
                    primary_topic={"id": topic_id},
                    from_publication_date=since,
                    has_abstract=True,
                    language=settings["language"],
                    type="|".join(settings["keep_types"]),
                    is_retracted=False,
                )
                .select(FIELDS)
            )
            for page in query.paginate(per_page=200, n_max=None):
                records += [(paper, topic_id) for paper in page]
    return records


def rank_by_syllabus(records, config, model, model_cfg):
    """Score each (paper, topic_id) by its highest cosine similarity to any learn query, best first."""
    queries = [q for area in config["areas"].values() if area["active"]
               for item in area["learn"] for q in item["queries"]]
    query_vectors = np.array([encode_query(model, model_cfg, q) for q in queries])

    texts = [embedding_text(p["title"], rebuild_abstract(p["abstract_inverted_index"])) for p, _ in records]
    paper_vectors = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True)

    scores = (paper_vectors @ query_vectors.T).max(axis=1)
    order = np.argsort(-scores)
    return [(records[i], float(scores[i])) for i in order]


def save_raw(records, raw_dir=RAW_DIR):
    """Add the papers to raw_dir/recent/<today>.json and their authors to raw_dir/authors.json.

    The raw files are the source of truth: 04_load.py deletes every paper that is not in them.
    A second refresh on the same day adds to that day's file. Returns the updated authors dict.
    """
    recent_file = raw_dir / "recent" / f"{datetime.now(UTC).date().isoformat()}.json"
    recent_file.parent.mkdir(parents=True, exist_ok=True)
    entries = json.loads(recent_file.read_text()) if recent_file.exists() else []
    entries += [{"topic_id": topic_id, "paper": paper} for paper, topic_id in records]
    recent_file.write_text(json.dumps(entries, indent=2))

    authors_file = raw_dir / "authors.json"
    authors = json.loads(authors_file.read_text()) if authors_file.exists() else {}
    authors.update({paper["id"]: author_names(paper) for paper, _ in records})
    authors_file.write_text(json.dumps(authors))
    return authors


def refresh(config, model, model_cfg):
    """Add the recently published papers that pass the rules and match the syllabus best.

    Fetch the last `days` days, apply the credibility rules, drop duplicates within the batch, take
    the papers scoring at least min_similarity (at most max_papers) and add those that are not in
    the database yet: write them to the raw files, save them to Postgres and embed them.
    The top papers are chosen before known papers are skipped, so running it again adds nothing
    and the next refresh only adds what has entered the top of the window.
    """
    settings, refresh_cfg = config["settings"], config["refresh"]
    records = [record for record in fetch_recent(config, refresh_cfg["days"]) if passes_rules(record[0], settings)]
    records, _ = deduplicate(records, settings["dedup_year_window"])

    ranked = rank_by_syllabus(records, config, model, model_cfg) if records else []
    top = select_top(ranked, refresh_cfg["min_similarity"], refresh_cfg["max_papers"])
    with connect() as conn, conn.cursor() as cur:
        known_ids, known_titles = known_papers(cur)
    selected = [(record, score) for record, score in top if not is_known(record[0], known_ids, known_titles)]

    kept = [record for record, _ in selected]
    if kept:
        authors = save_raw(kept)
        with connect() as conn, conn.cursor() as cur:
            save_papers(cur, {}, kept, authors, settings)
        embed_missing(config)

    return {
        "added": len(kept),
        "papers": [
            {"id": paper["id"].split("/")[-1], "title": paper["title"], "year": paper["publication_year"],
             "score": round(score, 3), "topic_id": topic_id}
            for (paper, topic_id), score in selected
        ],
        "candidates": len(records),
        "already_known": len(top) - len(kept),
    }
