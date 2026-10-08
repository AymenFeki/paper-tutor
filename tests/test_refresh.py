"""Tests for the weekly refresh's raw files (temporary directory, no network or database)."""

import json

from paper_tutor.refresh import add_recent, save_raw


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
