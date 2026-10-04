import json
from pathlib import Path

import pandas as pd

from paper_tutor.corpus import load_config, venue_lists, check_rules

config = load_config()
settings = config["settings"]
raw_dir = Path("data/raw")

# Load all raw papers into one table, with one True/False column per rule
rows = []
for area_name, area in config["areas"].items():
    if not area["active"]:
        continue
    for topic_id in area["topics"]:
        with open(raw_dir / f"{topic_id}.json") as f:
            papers = json.load(f)
        for paper in papers:
            row = {
                "area": area_name,
                "topic": topic_id,
                "id": paper["id"],
                "year": paper["publication_year"],
                "type": paper["type"],
                "language": paper["language"],
                "venue_lists": venue_lists(paper),
                "fwci": paper["fwci"],
            }
            row.update(check_rules(paper, settings))
            rows.append(row)

df = pd.DataFrame(rows)
print("papers, columns:", df.shape)
print("duplicate ids:", df["id"].duplicated().sum())

rule_cols = ["bad_type", "no_abstract", "not_english", "retracted", "not_listed"]
df["kept"] = ~df[rule_cols].any(axis=1)

print("\nShare of papers breaking each rule, per area:")
print(df.groupby("area")[rule_cols + ["kept"]].mean().round(2))
print(df["kept"].sum(), "of", len(df), "papers kept")

dropped = df[df["bad_type"]]
print("\nDropped types per area:")
print(pd.crosstab(dropped["type"], dropped["area"]))

other_rules = ["bad_type", "no_abstract", "not_english", "retracted"]
df["only_venue"] = df["not_listed"] & ~df[other_rules].any(axis=1)
print("\nShare of papers failing only the venue rule, per area:")
print(df.groupby("area")["only_venue"].mean().round(2))