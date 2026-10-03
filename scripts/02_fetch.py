import json
import os
from pathlib import Path

import pyalex
import yaml
from dotenv import load_dotenv
from pyalex import Works

load_dotenv()
pyalex.config.api_key = os.getenv("OPENALEX_API_KEY")

with open("config/syllabus.yaml") as f:
    config = yaml.safe_load(f)

raw_dir = Path("data/raw")
raw_dir.mkdir(parents=True, exist_ok=True)

n_per_topic = config["settings"]["papers_per_topic"]
fields = [
    "id", "doi", "title", "publication_year", "type", "language",
    "is_retracted", "cited_by_count", "fwci", "citation_normalized_percentile",
    "abstract_inverted_index", "primary_location", "best_oa_location", "primary_topic",
]

for area_name, area in config["areas"].items():
    if not area["active"]:
        continue
    for topic_id in area["topics"]:
        out_file = raw_dir / f"{topic_id}.json"
        if out_file.exists():
            print(f"skip {topic_id} (already downloaded)")
            continue
        papers = (
            Works()
            .filter(primary_topic={"id": topic_id})
            .sample(n_per_topic, seed=42)
            .select(fields)
            .get(per_page=200)
        )
        with open(out_file, "w") as f:
            json.dump(papers, f, indent=2)
        print(f"{area_name} | {topic_id} | {len(papers)} papers")