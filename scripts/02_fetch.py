import json
import os
from pathlib import Path

import pyalex
from dotenv import load_dotenv
from pyalex import Works

from paper_tutor.corpus import load_config

load_dotenv()
pyalex.config.api_key = os.getenv("OPENALEX_API_KEY")

config = load_config()
settings = config["settings"]
n_per_topic = settings["papers_per_topic"]
fields = [
    "id", "doi", "title", "publication_year", "type", "language",
    "is_retracted", "cited_by_count", "fwci", "citation_normalized_percentile",
    "abstract_inverted_index", "primary_location", "best_oa_location", "primary_topic",
]


def fetch(topic_id, selection):
    query = Works().filter(primary_topic={"id": topic_id}).select(fields)
    if selection == "random":
        query = query.sample(n_per_topic, seed=42)
    elif selection == "top_cited":
        query = query.sort(cited_by_count="desc")
    else:
        raise ValueError(f"unknown selection: {selection}")
    return query.get(per_page=n_per_topic)


for selection in settings["selections"]:
    raw_dir = Path("data/raw") / selection
    raw_dir.mkdir(parents=True, exist_ok=True)
    for area_name, area in config["areas"].items():
        if not area["active"]:
            continue
        for topic_id in area["topics"]:
            out_file = raw_dir / f"{topic_id}.json"
            if out_file.exists():
                print(f"skip {selection}/{topic_id} (already downloaded)")
                continue
            papers = fetch(topic_id, selection)
            with open(out_file, "w") as f:
                json.dump(papers, f, indent=2)
            print(f"{selection} | {area_name} | {topic_id} | {len(papers)} papers")