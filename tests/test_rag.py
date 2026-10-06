"""Tests for the RAG helpers."""

from paper_tutor.rag import document_text, format_context, is_short_query, passes_threshold, reciprocal_rank_fusion


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


# document_text, is_short_query and passes_threshold


def test_document_text_joins_title_and_abstract():
    assert document_text({"title": "Lasso", "abstract": "We propose..."}) == "Lasso. We propose..."


def test_document_text_missing_abstract():
    assert document_text({"title": "Lasso", "abstract": None}) == "Lasso"


def test_is_short_query():
    assert is_short_query("Jeffreys prior", 5)
    assert is_short_query("retrieval augmented generation", 5)
    assert not is_short_query("How should I choose a prior distribution for my model?", 5)
    assert not is_short_query("lasso", 0)  # 0 switches the short-query rule off


SETTINGS = {"min_similarity": 0.62, "min_rerank_score_short": 0.5}


def test_passes_threshold_long_question_uses_similarity():
    assert passes_threshold({"similarity": 0.70, "rerank_score": 0.001}, False, SETTINGS)
    assert not passes_threshold({"similarity": 0.60, "rerank_score": 0.9}, False, SETTINGS)


def test_passes_threshold_short_query_uses_reranker():
    # "pasta recipe": similar enough by embedding, but the reranker says off-topic
    assert not passes_threshold({"similarity": 0.63, "rerank_score": 0.003}, True, SETTINGS)
    assert passes_threshold({"similarity": 0.63, "rerank_score": 0.99}, True, SETTINGS)
