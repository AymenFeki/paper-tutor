"""Weekly refresh: fetch recently published papers from OpenAlex and add the ones that pass the rules."""

import json
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pyalex
from dotenv import load_dotenv
from pyalex import Works

from paper_tutor.corpus import (
    author_names,
    deduplicate,
    is_known,
    matches_area_keywords,
    passes_rules,
    rebuild_abstract,
    refresh_topic_areas,
    refresh_topic_rows,
    select_top,
)
from paper_tutor.db import connect, known_papers, save_papers, save_refreshed_raw
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
    """Papers published in the last `days` days for the topics and refresh_topics of every active area,
    as (paper, topic_id) pairs.

    OpenAlex already filters out what the credibility rules would drop anyway (no abstract, not
    English, wrong type, retracted), and every page is fetched, so no topic is cut off.
    """
    settings = config["settings"]
    since = (datetime.now(UTC).date() - timedelta(days=days)).isoformat()
    records = []
    for area in config["areas"].values():
        if not area["active"]:
            continue
        for topic_id in area["topics"] + area.get("refresh_topics", []):
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


def add_recent(entries, day, raw_dir=RAW_DIR):
    """Add {"topic_id", "paper"} entries to raw_dir/recent/<day>.json and their authors to raw_dir/authors.json.

    Papers that are already in that day's file are skipped, so adding the same entries twice changes
    nothing. Returns the number of entries added and the updated authors dict.
    """
    recent_file = raw_dir / "recent" / f"{day}.json"
    recent_file.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(recent_file.read_text()) if recent_file.exists() else []
    known = {entry["paper"]["id"] for entry in existing}
    new = [entry for entry in entries if entry["paper"]["id"] not in known]
    if new:
        recent_file.write_text(json.dumps(existing + new, indent=2))

    authors_file = raw_dir / "authors.json"
    authors = json.loads(authors_file.read_text()) if authors_file.exists() else {}
    authors.update({entry["paper"]["id"]: author_names(entry["paper"]) for entry in entries})
    authors_file.write_text(json.dumps(authors))
    return len(new), authors


def save_raw(records, raw_dir=RAW_DIR):
    """Add the papers to raw_dir/recent/<today>.json and their authors to raw_dir/authors.json.

    The raw files are the source of truth: 04_load.py deletes every paper that is not in them.
    A second refresh on the same day adds to that day's file. Returns the updated authors dict.
    In the cloud container these files are lost on restart, so refresh() also keeps every raw
    record in the refreshed_raw table; scripts/16_pull_refreshed.py copies them back to data/raw/.
    """
    entries = [{"topic_id": topic_id, "paper": paper} for paper, topic_id in records]
    _, authors = add_recent(entries, datetime.now(UTC).date().isoformat(), raw_dir)
    return authors


def refresh(config, model, model_cfg, dry_run=False):
    """Add the recently published papers that pass the rules and match the syllabus best.

    Fetch the last `days` days, apply the credibility rules (and the area keywords for papers from
    refresh topics), drop duplicates within the batch, take
    the papers scoring at least min_similarity (at most max_papers) and add those that are not in
    the database yet: write them to the raw files, save them to Postgres and embed them.
    The top papers are chosen before known papers are skipped, so running it again adds nothing
    and the next refresh only adds what has entered the top of the window.
    With dry_run, nothing is written: the result shows what would be added. timings_s has the seconds
    per phase (fetch from OpenAlex; rules, deduplication and ranking; database check and saving).
    """
    settings, refresh_cfg = config["settings"], config["refresh"]
    refresh_areas = refresh_topic_areas(config)
    timings, start = {}, time.perf_counter()

    def phase_done(name):
        """Store the seconds since the last phase under name, and log them (the refresh runs for minutes)."""
        nonlocal start
        timings[name] = round(time.perf_counter() - start, 1)
        start = time.perf_counter()
        print(f"refresh: {name} took {timings[name]} s", flush=True)

    fetched = fetch_recent(config, refresh_cfg["days"])
    phase_done("fetch")
    records = [record for record in fetched
               if passes_rules(record[0], settings) and matches_area_keywords(record, config, refresh_areas)]
    records, _ = deduplicate(records, settings["dedup_year_window"])

    ranked = rank_by_syllabus(records, config, model, model_cfg) if records else []
    top = select_top(ranked, refresh_cfg["min_similarity"], refresh_cfg["max_papers"])
    phase_done("rank")
    with connect() as conn, conn.cursor() as cur:
        known_ids, known_titles = known_papers(cur)
    selected = [(record, score) for record, score in top if not is_known(record[0], known_ids, known_titles)]

    kept = [record for record, _ in selected]
    if kept and not dry_run:
        authors = save_raw(kept)
        with connect() as conn, conn.cursor() as cur:
            save_refreshed_raw(cur, kept)
            save_papers(cur, refresh_topic_rows(kept, refresh_areas), kept, authors, settings)
        embed_missing(config)
    phase_done("save")

    return {
        "added": 0 if dry_run else len(kept),
        "dry_run": dry_run,
        "papers": [
            {"id": paper["id"].split("/")[-1], "title": paper["title"], "year": paper["publication_year"],
             "score": round(score, 3), "topic_id": topic_id}
            for (paper, topic_id), score in selected
        ],
        "fetched": len(fetched),
        "candidates": len(records),
        "already_known": len(top) - len(kept),
        "timings_s": timings,
    }
