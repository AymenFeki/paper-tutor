"""Tests for the weekly refresh: raw files (temporary directory) and the selection (no network or database)."""

import json
from unittest.mock import MagicMock

from paper_tutor import refresh as refresh_module
from paper_tutor.refresh import add_recent, refresh, save_raw


def paper(paper_id, *names):
    return {"id": f"https://openalex.org/{paper_id}", "title": paper_id,
            "authorships": [{"author": {"display_name": name}} for name in names]}


def test_save_raw_writes_recent_file_and_merges_authors(tmp_path):
    (tmp_path / "authors.json").write_text(json.dumps({"https://openalex.org/W0": ["Old Author"]}))

    authors = save_raw([(paper("W1", "Ada", "Bob"), "T1")], raw_dir=tmp_path)

    [recent_file] = (tmp_path / "recent").glob("*.json")
    assert json.loads(recent_file.read_text()) == [{"topic_id": "T1", "paper": paper("W1", "Ada", "Bob")}]
    expected = {"https://openalex.org/W0": ["Old Author"], "https://openalex.org/W1": ["Ada", "Bob"]}
    assert authors == expected
    assert json.loads((tmp_path / "authors.json").read_text()) == expected


def test_save_raw_twice_on_one_day_keeps_both(tmp_path):
    save_raw([(paper("W1"), "T1")], raw_dir=tmp_path)
    save_raw([(paper("W2"), "T2")], raw_dir=tmp_path)

    [recent_file] = (tmp_path / "recent").glob("*.json")
    assert [entry["paper"]["id"] for entry in json.loads(recent_file.read_text())] == [
        "https://openalex.org/W1", "https://openalex.org/W2"]


def test_add_recent_skips_papers_already_in_the_file(tmp_path):
    entries = [{"topic_id": "T1", "paper": paper("W1", "Ada")}]
    assert add_recent(entries, "2026-10-04", raw_dir=tmp_path)[0] == 1
    assert add_recent(entries, "2026-10-04", raw_dir=tmp_path)[0] == 0

    saved = json.loads((tmp_path / "recent" / "2026-10-04.json").read_text())
    assert [entry["paper"]["id"] for entry in saved] == ["https://openalex.org/W1"]


def candidate(paper_id, title, topic_id):
    """A recent record that passes the credibility rules."""
    record = {
        "id": f"https://openalex.org/{paper_id}", "title": title, "publication_year": 2026, "type": "article",
        "doi": None, "abstract_inverted_index": {f"word{i}": [i] for i in range(40)}, "language": "en",
        "is_retracted": False, "primary_topic": {"display_name": f"Topic {topic_id}"},
        "primary_location": {"source": {"display_name": "Journal X", "listed_in": ["cwts-core"]}},
    }
    return record, topic_id


REFRESH_CONFIG = {
    "settings": {"language": "en", "keep_types": ["article"], "venue_lists": ["cwts-core"],
                 "venue_exempt_types": [], "min_abstract_words": 30, "preprint_servers": [], "untrusted_repositories": [],
                 "dedup_year_window": 3, "authors_per_paper": 5},
    "refresh": {"days": 14, "max_papers": 20, "min_similarity": 0.75},
    "areas": {
        "statistics": {"active": True, "topics": ["T1"]},
        "agents": {"active": True, "topics": [], "refresh_topics": ["T9"], "keywords": ["language model"]},
    },
}


def fake_refresh(monkeypatch, records):
    """Patch the network, model and database parts of refresh(); returns the save_papers mock."""
    monkeypatch.setattr(refresh_module, "fetch_recent", lambda config, days: records)
    monkeypatch.setattr(refresh_module, "rank_by_syllabus", lambda recs, *args: [(r, 0.8) for r in recs])
    monkeypatch.setattr(refresh_module, "connect", MagicMock())
    monkeypatch.setattr(refresh_module, "known_papers", lambda cur: (set(), set()))
    monkeypatch.setattr(refresh_module, "save_raw", MagicMock(return_value={}))
    monkeypatch.setattr(refresh_module, "save_refreshed_raw", MagicMock())
    monkeypatch.setattr(refresh_module, "embed_missing", MagicMock())
    save_papers = MagicMock()
    monkeypatch.setattr(refresh_module, "save_papers", save_papers)
    return save_papers


def test_refresh_topic_papers_need_a_keyword_and_get_a_topic_row(monkeypatch):
    records = [candidate("W1", "Lasso paths", "T1"), candidate("W2", "A large language model", "T9"),
               candidate("W3", "Latent Dirichlet allocation", "T9")]
    save_papers = fake_refresh(monkeypatch, records)

    result = refresh(REFRESH_CONFIG, None, None)

    assert [p["id"] for p in result["papers"]] == ["W1", "W2"]
    topics, kept = save_papers.call_args.args[1:3]
    assert topics == {"T9": ("Topic T9", "agents")}
    assert [paper["title"] for paper, _ in kept] == ["Lasso paths", "A large language model"]


def test_refresh_dry_run_writes_nothing(monkeypatch):
    save_papers = fake_refresh(monkeypatch, [candidate("W1", "Lasso paths", "T1")])

    result = refresh(REFRESH_CONFIG, None, None, dry_run=True)

    assert result["dry_run"] and result["added"] == 0
    assert [p["id"] for p in result["papers"]] == ["W1"]
    save_papers.assert_not_called()
    refresh_module.save_raw.assert_not_called()
    refresh_module.embed_missing.assert_not_called()
