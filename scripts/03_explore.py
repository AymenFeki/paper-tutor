import json
from pathlib import Path

import pandas as pd
import yaml

with open("config/syllabus.yaml") as f:
    config = yaml.safe_load(f)

settings = config["settings"]
raw_dir = Path("data/raw")


def venue_lists(paper):
    location = paper.get("primary_location") or {}
    source = location.get("source") or {}
    return source.get("listed_in") or []


# Load all raw papers into one table
rows = []
for area_name, area in config["areas"].items():
    if not area["active"]:
        continue
    for topic_id in area["topics"]:
        with open(raw_dir / f"{topic_id}.json") as f:
            papers = json.load(f)
        for paper in papers:
            rows.append({
                "area": area_name,
                "topic": topic_id,
                "id": paper["id"],
                "year": paper["publication_year"],
                "type": paper["type"],
                "language": paper["language"],
                "has_abstract": paper["abstract_inverted_index"] is not None,
                "is_retracted": paper["is_retracted"],
                "venue_lists": venue_lists(paper),
                "fwci": paper["fwci"],
            })

df = pd.DataFrame(rows)
print("papers, columns:", df.shape)
print("duplicate ids:", df["id"].duplicated().sum())

# Credibility rules: True means the paper breaks the rule
allowed_lists = settings["venue_lists"]
exempt = settings["venue_exempt_types"]

df["bad_type"] = ~df["type"].isin(settings["keep_types"])
df["no_abstract"] = ~df["has_abstract"]
df["not_english"] = ~df["language"].eq(settings["language"])
df["retracted"] = df["is_retracted"]

# True if the journal is on at least one of the allowed lists
on_list = df["venue_lists"].apply(lambda lists: any(name in lists for name in allowed_lists))
df["not_listed"] = ~df["type"].isin(exempt) & ~on_list

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