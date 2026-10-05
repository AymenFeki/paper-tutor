import os

import pyalex
import yaml
from dotenv import load_dotenv
from pyalex import Works

load_dotenv()
key = os.getenv("OPENALEX_API_KEY")
print(key is not None)
pyalex.config.api_key = key

with open("config/syllabus.yaml") as f:
    data = yaml.safe_load(f)

for area_name, area in data["areas"].items():
    print(f"\n=== {area_name} ===")
    for item in area["learn"]:
        print(f"Searching for topics related to: {item['name']}")
        for query in item["queries"]:
            print(f"Query: {query}")
            results = Works().search_filter(title_and_abstract=f'"{query}"').group_by("primary_topic.id").get()
            for group in results[:5]:
                short_id = group["key"].split("/")[-1]
                print(f"  {short_id} | {group['key_display_name']} | {group['count']} papers")