"""Tests for the RAG helpers."""

from paper_tutor.rag import apply_threshold, document_text, format_context, reciprocal_rank_fusion


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


def test_format_context_shows_authors():
    papers = [{"title": "Paper A", "year": 2020, "venue": "Journal X", "authors": ["Ann Lee", "Bo Chen"],
               "abstract": "Abstract A."}]
    assert format_context(papers) == "[1] Paper A (2020, Journal X)\nAuthors: Ann Lee, Bo Chen\nAbstract A."


def test_format_context_no_authors_line_when_unknown():
    papers = [{"title": "Paper A", "year": 2020, "venue": "Journal X", "authors": [], "abstract": "Abstract A."}]
    assert "Authors" not in format_context(papers)


# reciprocal_rank_fusion


def test_rrf_paper_found_by_both_searches_wins():
    vector = ["a", "b", "c"]
    text = ["c", "d", "a"]
    # a: 1/61 + 1/63, c: 1/63 + 1/61 (tie, a was seen first), b: 1/62, d: 1/62
    assert reciprocal_rank_fusion([vector, text]) == ["a", "c", "b", "d"]


def test_rrf_single_list_keeps_order():
    assert reciprocal_rank_fusion([["x", "y", "z"]]) == ["x", "y", "z"]


def test_rrf_scores_by_rank_not_by_list():
    # "b" is 2nd and 2nd, "a" is 1st and missing: 2/(k+2) > 1/(k+1) for k=60
    assert reciprocal_rank_fusion([["a", "b"], ["c", "b"]])[0] == "b"


def test_rrf_empty():
    assert reciprocal_rank_fusion([[], []]) == []


# document_text and apply_threshold


def test_document_text_joins_title_and_abstract():
    assert document_text({"title": "Lasso", "abstract": "We propose..."}) == "Lasso. We propose..."


def test_document_text_missing_abstract():
    assert document_text({"title": "Lasso", "abstract": None}) == "Lasso"


def test_apply_threshold_keeps_scores_at_or_above_minimum():
    papers = [{"id": "a", "score": 0.8}, {"id": "b", "score": 0.62}, {"id": "c", "score": 0.5}]
    assert [p["id"] for p in apply_threshold(papers, 0.62)] == ["a", "b"]


def test_apply_threshold_can_return_nothing():
    assert apply_threshold([{"id": "a", "score": 0.3}], 0.62) == []
