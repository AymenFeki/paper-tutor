import json
from pathlib import Path

from paper_tutor.corpus import (
    deduplicate,
    get_source,
    load_config,
    passes_rules,
    rebuild_abstract,
    venue_lists,
)
from paper_tutor.db import connect

UPSERT_TOPIC = """
    INSERT INTO topics (id, name, area)
    VALUES (%s, %s, %s)
    ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name, area = EXCLUDED.area
"""

UPSERT_PAPER = """
    INSERT INTO papers (id, doi, title, abstract, year, type, language, venue,
                        venue_lists, is_retracted, fwci, oa_url, topic_id, authors)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (id) DO UPDATE SET
        title = EXCLUDED.title, abstract = EXCLUDED.abstract, fwci = EXCLUDED.fwci,
        venue_lists = EXCLUDED.venue_lists, oa_url = EXCLUDED.oa_url,
        authors = EXCLUDED.authors, loaded_at = now()
"""

# Papers that no longer pass the rules or were merged as duplicates (embeddings first: they point to papers)
DELETE_EMBEDDINGS = "DELETE FROM embeddings WHERE NOT (paper_id = ANY(%s))"
DELETE_PAPERS = "DELETE FROM papers WHERE NOT (id = ANY(%s))"

config = load_config()
settings = config["settings"]
raw_dir = Path("data/raw")
authors_file = raw_dir / "authors.json"
authors = json.loads(authors_file.read_text()) if authors_file.exists() else {}

topics = {}   # topic_id -> (name, area)
records = []  # (paper, topic_id) for every record that passes the rules

# 1. Read: apply the credibility rules
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
                if passes_rules(paper, settings):
                    records.append((paper, topic_id))

# 2. Keep the best record per paper (same title, within the year window)
best, merged = deduplicate(records, settings["dedup_year_window"])
cross_year = [(kept, dropped) for kept, dropped in merged
              if kept["publication_year"] != dropped["publication_year"]]
print(f"{len(records)} kept records, {len(best)} unique papers, {len(merged)} duplicate papers merged "
      f"({len(cross_year)} across years)")
for kept, dropped in cross_year[:10]:
    print(f"  {kept['title'][:70]!r}: kept {kept['publication_year']} {kept['type']}, "
          f"dropped {dropped['publication_year']} {dropped['type']}")

# 3. Write: topics first, then the unique papers, then remove papers that are no longer kept
with connect() as conn, conn.cursor() as cur:
    for topic_id, (name, area) in topics.items():
        cur.execute(UPSERT_TOPIC, (topic_id, name, area))

    for paper, topic_id in best:
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
            authors.get(paper["id"], [])[:settings["authors_per_paper"]],
        ))

    kept_ids = [paper["id"].split("/")[-1] for paper, _ in best]
    cur.execute(DELETE_EMBEDDINGS, (kept_ids,))
    print(cur.rowcount, "embeddings of removed papers deleted")
    cur.execute(DELETE_PAPERS, (kept_ids,))
    print(cur.rowcount, "papers removed")

print(len(best), "papers loaded")