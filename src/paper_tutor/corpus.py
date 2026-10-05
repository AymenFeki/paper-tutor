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
