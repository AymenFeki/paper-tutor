"""Tests for the RAG helpers."""

from paper_tutor.rag import format_context


def test_format_context_numbers_papers():
    papers = [
        {"title": "Paper A", "year": 2020, "venue": "Journal X", "abstract": "Abstract A."},
        {"title": "Paper B", "year": 2021, "venue": "Journal Y", "abstract": "Abstract B."},
    ]
    result = format_context(papers)
    assert "[1] Paper A (2020, Journal X)" in result
    assert "[2] Paper B (2021, Journal Y)" in result


def test_format_context_missing_venue():
    papers = [{"title": "Paper A", "year": 2020, "venue": None, "abstract": "Abstract A."}]
    result = format_context(papers)
    assert "unknown venue" in result

def test_format_context_includes_abstract_after_title():
    papers = [{"title": "Paper A", "year": 2020, "venue": "Journal X", "abstract": "Abstract A."}]
    assert format_context(papers) == "[1] Paper A (2020, Journal X)\nAbstract A."


def test_format_context_separates_papers_with_blank_line():
    papers = [
        {"title": "Paper A", "year": 2020, "venue": "Journal X", "abstract": "Abstract A."},
        {"title": "Paper B", "year": 2021, "venue": "Journal Y", "abstract": "Abstract B."},
    ]
    assert "Abstract A.\n\n[2] Paper B" in format_context(papers)


def test_format_context_no_papers():
    assert format_context([]) == ""
