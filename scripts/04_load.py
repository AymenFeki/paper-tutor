import json
from pathlib import Path

from paper_tutor.corpus import (
    load_config, get_source, venue_lists, rebuild_abstract, passes_rules,
    normalize_title, record_quality,
)
from paper_tutor.db import connect

UPSERT_TOPIC = """
    INSERT INTO topics (id, name, area)
    VALUES (%s, %s, %s)
    ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name, area = EXCLUDED.area
"""

UPSERT_PAPER = """
    INSERT INTO papers (id, doi, title, abstract, year, type, language, venue,
                        venue_lists, is_retracted, fwci, oa_url, topic_id)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (id) DO UPDATE SET
        title = EXCLUDED.title, abstract = EXCLUDED.abstract, fwci = EXCLUDED.fwci,
        venue_lists = EXCLUDED.venue_lists, oa_url = EXCLUDED.oa_url, loaded_at = now()
"""

config = load_config()
settings = config["settings"]
raw_dir = Path("data/raw")

topics = {}  # topic_id -> (name, area)
best = {}    # (normalised title, year) -> (paper, topic_id)
seen = 0

# Keep the best record per (title, year)
for selection in settings["selections"]:
    for area_name, area in config["areas"].items():
        if not area["active"]:
            continue
        for topic_id in area["topics"]:
            with open(raw_dir / selection / f"{topic_id}.json") as f:
                papers = json.load(f)
            if papers:
                topics[topic_id] = (papers[0]["primary_topic"]["display_name"], area_name)

            for paper in papers:
                if not passes_rules(paper, settings):
                    continue
                seen += 1
                key = (normalize_title(paper["title"]), paper["publication_year"])
                if key not in best or record_quality(paper) > record_quality(best[key][0]):
                    best[key] = (paper, topic_id)

print(f"{seen} kept records, {len(best)} unique papers, {seen - len(best)} duplicates removed")

# 2. Write: topics first, then the unique papers
with connect() as conn, conn.cursor() as cur:
    for topic_id, (name, area) in topics.items():
        cur.execute(UPSERT_TOPIC, (topic_id, name, area))

    for paper, topic_id in best.values():
        source = get_source(paper)
        oa = paper.get("best_oa_location") or {}
        cur.execute(UPSERT_PAPER, (
            paper["id"].split("/")[-1],
            paper["doi"],
            paper["title"],
            rebuild_abstract(paper["abstract_inverted_index"]),
            paper["publication_year"],
            paper["type"],
            paper["language"],
            source.get("display_name"),
            venue_lists(paper),
            paper["is_retracted"],
            paper["fwci"],
            oa.get("pdf_url") or oa.get("landing_page_url"),
            topic_id,
        ))

print(len(best), "papers loaded")