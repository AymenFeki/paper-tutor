import json
from pathlib import Path

from paper_tutor.corpus import deduplicate, load_config, passes_rules, select_seed_papers
from paper_tutor.db import connect, save_papers

# Papers that no longer pass the rules or were merged as duplicates (embeddings first: they point to papers)
DELETE_EMBEDDINGS = "DELETE FROM embeddings WHERE NOT (paper_id = ANY(%s))"
DELETE_PAPERS = "DELETE FROM papers WHERE NOT (id = ANY(%s))"

config = load_config()
settings = config["settings"]
raw_dir = Path("data/raw")
authors_file = raw_dir / "authors.json"
authors = json.loads(authors_file.read_text()) if authors_file.exists() else {}
titles_file = raw_dir / "arxiv_titles.json"
use_arxiv_titles = settings["check_arxiv_titles"] and titles_file.exists()
arxiv_titles = json.loads(titles_file.read_text()) if use_arxiv_titles else {}

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

# Areas built from seed papers: rules, on-topic keywords, correct arXiv metadata, most cited.
# They come after the topic areas, so a paper that is already in a topic area keeps that area.
for area_name, area in config["areas"].items():
    if not area["active"] or not area.get("seeds"):
        continue
    papers = json.loads((raw_dir / "seeds" / f"{area_name}.json").read_text())
    selected = select_seed_papers(papers, settings, area["keywords"], arxiv_titles, area["max_papers"])
    topic_id = f"seeds-{area_name}"
    topics[topic_id] = ("Papers citing or cited by the seed papers", area_name)
    records += [(paper, topic_id) for paper in selected]
    print(f"{area_name}: {len({p['id'] for p in papers})} candidates from seed papers, {len(selected)} selected")

# Papers added by the weekly refresh (paper_tutor.refresh), saved as {"topic_id", "paper"} entries.
# They come after the topic and seed areas, so a paper that is already there keeps its area.
recent = []
for recent_file in sorted((raw_dir / "recent").glob("*.json")):
    recent += [(entry["paper"], entry["topic_id"]) for entry in json.loads(recent_file.read_text())
               if entry["topic_id"] in topics and passes_rules(entry["paper"], settings)]
records += recent
print(f"recent: {len(recent)} papers from weekly refreshes")

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
    save_papers(cur, topics, best, authors, settings)

    kept_ids = [paper["id"].split("/")[-1] for paper, _ in best]
    cur.execute(DELETE_EMBEDDINGS, (kept_ids,))
    print(cur.rowcount, "embeddings of removed papers deleted")
    cur.execute(DELETE_PAPERS, (kept_ids,))
    print(cur.rowcount, "papers removed")

print(len(best), "papers loaded")