"""Config loading and credibility rules shared by all scripts."""

import yaml

import re

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

def check_rules(paper, settings):
    """Return a dict: rule name -> True if the paper breaks that rule."""
    on_list = any(name in venue_lists(paper) for name in settings["venue_lists"])
    return {
        "bad_type": paper["type"] not in settings["keep_types"],
        "no_abstract": paper["abstract_inverted_index"] is None,
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
    """Higher is better: prefer records with a DOI, then with a longer abstract."""
    has_doi = paper["doi"] is not None
    abstract_length = len(paper["abstract_inverted_index"] or {})
    return (has_doi, abstract_length)