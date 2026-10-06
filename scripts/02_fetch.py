import json
import os
import re
import time
from pathlib import Path
from xml.etree import ElementTree

import httpx
import pyalex
from dotenv import load_dotenv
from pyalex import Works

from paper_tutor.corpus import arxiv_id, load_config, normalize_title

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


def fetch_by_ids(paper_ids):
    """Full records for a list of OpenAlex ids, 100 per request."""
    papers = []
    for start in range(0, len(paper_ids), 100):
        batch = paper_ids[start:start + 100]
        papers += Works().filter_or(openalex_id=batch).select(fields).get(per_page=len(batch))
    return papers


def fetch_seed_area(area):
    """Papers that cite the area's seed papers, papers cited by them, and the seeds themselves.

    A seed can have several OpenAlex records (arXiv preprint, conference version): the one found
    by its DOI and the ones with exactly its title. Citations to all of them are collected.
    References are only taken from records that really have the seed's title, because some
    records under a famous arXiv DOI show an unrelated paper's title, abstract and references.
    """
    papers = []
    for seed in area["seeds"]:
        title = normalize_title(seed["title"])
        doi_record = Works()[f"https://doi.org/{seed['doi']}"]
        same_title = [r for r in Works().search_filter(title=title).get(per_page=10)
                      if normalize_title(r["title"]) == title]
        records = {r["id"]: r for r in [doi_record] + same_title}
        good = [r for r in records.values() if normalize_title(r["title"]) == title]

        citing = Works().filter(cites="|".join(records.keys())).sort(cited_by_count="desc").select(fields)
        citing = citing.get(per_page=area["citing_per_seed"])
        references = sorted({ref for r in good for ref in r["referenced_works"]})
        papers += citing + fetch_by_ids(references) + fetch_by_ids([r["id"] for r in good])
        print(f"  {seed['title'][:50]}: {len(records)} records ({len(good)} with the right title), "
              f"{len(citing)} citing, {len(references)} references")
    return papers


def fetch_arxiv_titles(ids):
    """Titles from the arXiv API for up to 100 arXiv ids (arXiv asks for one request every 3 seconds)."""
    response = httpx.get("https://export.arxiv.org/api/query",
                         params={"id_list": ",".join(ids), "max_results": len(ids)}, timeout=60)
    response.raise_for_status()
    atom = "{http://www.w3.org/2005/Atom}"
    titles = {}
    for entry in ElementTree.fromstring(response.text).findall(f"{atom}entry"):
        url = entry.find(f"{atom}id").text  # e.g. http://arxiv.org/abs/2005.11401v4
        titles[re.sub(r"v\d+$", "", url.rsplit("/abs/", 1)[-1])] = " ".join(entry.find(f"{atom}title").text.split())
    time.sleep(3)
    return titles


def fetch_authors(paper_ids):
    """Author names in author order, for up to 100 papers in one request."""
    papers = Works().filter_or(openalex_id=paper_ids).select(["id", "authorships"]).get(per_page=len(paper_ids))
    return {
        paper["id"]: [authorship["author"]["display_name"] for authorship in paper["authorships"]]
        for paper in papers
    }


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

# Areas built from seed papers instead of topics
seed_dir = Path("data/raw/seeds")
seed_dir.mkdir(parents=True, exist_ok=True)
for area_name, area in config["areas"].items():
    if not area["active"] or not area.get("seeds"):
        continue
    out_file = seed_dir / f"{area_name}.json"
    if out_file.exists():
        print(f"skip seeds/{area_name} (already downloaded)")
        continue
    print(f"seeds | {area_name}")
    out_file.write_text(json.dumps(fetch_seed_area(area), indent=2))

# Authors were not in the original field list, so they are cached separately for every downloaded paper
authors_file = Path("data/raw/authors.json")
authors = json.loads(authors_file.read_text()) if authors_file.exists() else {}
paper_ids = set()
for raw_file in Path("data/raw").glob("*/*.json"):
    paper_ids.update(paper["id"] for paper in json.loads(raw_file.read_text()))
missing = sorted(paper_ids - authors.keys())
print(f"authors: {len(paper_ids)} papers, {len(missing)} without cached authors")
for start in range(0, len(missing), 100):
    batch = missing[start:start + 100]
    found = fetch_authors(batch)
    for paper_id in batch:
        authors[paper_id] = found.get(paper_id, [])
    authors_file.write_text(json.dumps(authors))
    print(f"  authors fetched for {start + len(batch)} of {len(missing)} papers")

# arXiv's own titles for papers with an arXiv DOI, to catch OpenAlex records with wrong metadata
titles_file = Path("data/raw/arxiv_titles.json")
arxiv_titles = json.loads(titles_file.read_text()) if titles_file.exists() else {}
ids = set()
for raw_file in Path("data/raw").glob("*/*.json"):
    ids.update(arxiv_id(paper["doi"]) for paper in json.loads(raw_file.read_text()))
missing = sorted(ids - arxiv_titles.keys() - {None})
print(f"arXiv titles: {len(ids - {None})} papers with an arXiv DOI, {len(missing)} not cached")
for start in range(0, len(missing), 100):
    batch = missing[start:start + 100]
    found = fetch_arxiv_titles(batch)
    for paper_arxiv_id in batch:
        arxiv_titles[paper_arxiv_id] = found.get(paper_arxiv_id)
    titles_file.write_text(json.dumps(arxiv_titles))
    print(f"  arXiv titles fetched for {start + len(batch)} of {len(missing)} papers")
