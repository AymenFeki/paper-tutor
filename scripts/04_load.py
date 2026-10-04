import json
from pathlib import Path

from paper_tutor.corpus import load_config, get_source, venue_lists, rebuild_abstract, passes_rules
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
loaded = 0

with connect() as conn, conn.cursor() as cur:
    for area_name, area in config["areas"].items():
        if not area["active"]:
            continue
        for topic_id in area["topics"]:
            with open(raw_dir / f"{topic_id}.json") as f:
                papers = json.load(f)

            topic_name = papers[0]["primary_topic"]["display_name"] if papers else None
            cur.execute(UPSERT_TOPIC, (topic_id, topic_name, area_name))

            for paper in papers:
                if not passes_rules(paper, settings):
                    continue
                best = paper.get("best_oa_location") or {}
                cur.execute(UPSERT_PAPER, (
                    paper["id"].split("/")[-1],
                    paper["doi"],
                    paper["title"],
                    rebuild_abstract(paper["abstract_inverted_index"]),
                    paper["publication_year"],
                    paper["type"],
                    paper["language"],
                    get_source(paper).get("display_name"),
                    venue_lists(paper),
                    paper["is_retracted"],
                    paper["fwci"],
                    best.get("pdf_url") or best.get("landing_page_url"),
                    topic_id,
                ))
                loaded += 1

print(loaded, "papers loaded")