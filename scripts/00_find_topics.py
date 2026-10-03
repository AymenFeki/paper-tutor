import os
from dotenv import load_dotenv
import pyalex
import yaml
from pyalex import Topics

load_dotenv()
key = os.getenv("OPENALEX_API_KEY")
print(key is not None)
pyalex.config.api_key = key

with open("config/syllabus.yaml") as f:
  data  = yaml.safe_load(f)

for area_name, area in data["areas"].items():
    if area["topics"]:
        continue
    print(f"\n=== {area_name} ===")
    for item in area["learn"]:
        print(f"Searching for topics related to: {item['name']}")
        for query in item["queries"]:
            print(f"Query: {query}")
            results = Topics().search(query).select(["id", "display_name", "subfield", "works_count"]).get()
            for topic in results[:5]:
                short_id = topic["id"].split("/")[-1]
                print(f"  {short_id} | {topic['display_name']} | {topic['subfield']['display_name']} | {topic['works_count']}")