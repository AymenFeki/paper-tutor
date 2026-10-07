"""Config loading and credibility rules shared by all scripts."""

import re
from collections import defaultdict

import yaml


def load_config(path="config/syllabus.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def get_source(paper):
    location = paper.get("primary_location") or {}
    return location.get("source") or {}


def venue_lists(paper):
    return get_source(paper).get("listed_in") or []


def rebuild_abstract(inverted_index):
    """Turn OpenAlex's inverted index back into plain text."""
    if inverted_index is None:
        return None
    pairs = []
    for word, positions in inverted_index.items():
        for pos in positions:
            pairs.append((pos, word))
    return " ".join(word for _, word in sorted(pairs))


def abstract_word_count(inverted_index):
    if not inverted_index:
        return 0
    return sum(len(positions) for positions in inverted_index.values())


def check_rules(paper, settings):
    """Return a dict: rule name -> True if the paper breaks that rule."""
    on_list = any(name in venue_lists(paper) for name in settings["venue_lists"])
    return {
        "bad_type": paper["type"] not in settings["keep_types"],
        "no_abstract": paper["abstract_inverted_index"] is None,
        "short_abstract": abstract_word_count(paper["abstract_inverted_index"]) < settings["min_abstract_words"],
        "not_english": paper["language"] != settings["language"],
        "retracted": bool(paper["is_retracted"]),
        "not_listed": paper["type"] not in settings["venue_exempt_types"] and not on_list,
    }


def passes_rules(paper, settings):
    return not any(check_rules(paper, settings).values())


def normalize_title(title):
    """Lowercase, and keep only letters and digits, so small differences don't matter."""
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def record_quality(paper):
    """Higher is better: prefer the published version (not a preprint), then a DOI, then a venue,
    then a longer abstract."""
    is_published = paper["type"] != "preprint"
    has_doi = paper["doi"] is not None
    has_venue = get_source(paper).get("display_name") is not None
    return (is_published, has_doi, has_venue, abstract_word_count(paper["abstract_inverted_index"]))


def split_by_year_window(records, year_window):
    """Split (paper, topic_id) records into groups whose years are at most year_window apart.

    A group starts at its earliest year and takes every record up to year_window years later.
    """
    groups = []
    start_year = None
    for record in sorted(records, key=lambda record: record[0]["publication_year"]):
        year = record[0]["publication_year"]
        if groups and year - start_year <= year_window:
            groups[-1].append(record)
        else:
            groups.append([record])
            start_year = year
    return groups


def deduplicate(records, year_window):
    """Keep one record per paper: same normalised title, published within year_window years.

    records is a list of (paper, topic_id). Returns the kept records and a list of
    (kept_paper, dropped_paper) pairs with different OpenAlex ids, so merges can be checked.
    Papers without a usable title are grouped by their own id, so they are never merged.
    """
    by_title = defaultdict(list)
    for record in records:
        paper = record[0]
        by_title[normalize_title(paper["title"]) or paper["id"]].append(record)

    kept, merged = [], []
    for title_records in by_title.values():
        for group in split_by_year_window(title_records, year_window):
            best = max(group, key=lambda record: record_quality(record[0]))
            kept.append(best)
            seen_ids = {best[0]["id"]}
            for paper, _ in group:
                if paper["id"] not in seen_ids:
                    merged.append((best[0], paper))
                    seen_ids.add(paper["id"])
    return kept, merged


def arxiv_id(doi):
    """The arXiv id in an arXiv DOI ("10.48550/arxiv.2005.11401" -> "2005.11401"), or None."""
    match = re.search(r"arxiv\.(\d{4}\.\d{4,5})", (doi or "").lower())
    return match.group(1) if match else None


def mentions_keywords(paper, keywords):
    """True if the title or abstract contains one of the keywords (case-insensitive, at a word start)."""
    text = f"{paper['title'] or ''} {rebuild_abstract(paper['abstract_inverted_index']) or ''}".lower()
    return any(re.search(r"\b" + re.escape(keyword.lower()), text) for keyword in keywords)


def has_wrong_arxiv_title(paper, arxiv_titles):
    """True if OpenAlex gives this arXiv paper a different title than arXiv itself.

    Some OpenAlex records keep the arXiv DOI and citation count of a famous paper but show an
    unrelated title and abstract. arxiv_titles maps arXiv id -> title, fetched by 02_fetch.py.
    """
    real_title = arxiv_titles.get(arxiv_id(paper["doi"]))
    return real_title is not None and normalize_title(real_title) != normalize_title(paper["title"])


def select_seed_papers(papers, settings, keywords, arxiv_titles, max_papers):
    """Pick the papers of a seed area: credibility rules, on topic, correct metadata, most cited first."""
    unique = {paper["id"]: paper for paper in papers}.values()
    selected = [
        paper for paper in unique
        if passes_rules(paper, settings)
        and mentions_keywords(paper, keywords)
        and not has_wrong_arxiv_title(paper, arxiv_titles)
    ]
    selected.sort(key=lambda paper: paper["cited_by_count"], reverse=True)
    return selected[:max_papers]


def paper_row(paper, topic_id, authors, settings):
    """One raw OpenAlex record as the values for UPSERT_PAPER, in column order."""
    source = get_source(paper)
    oa = paper.get("best_oa_location") or {}
    return (
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
        authors[:settings["authors_per_paper"]],
    )