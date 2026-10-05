"""Tests for the corpus helpers: abstracts, credibility rules, titles and record quality."""

import pytest

from paper_tutor.corpus import check_rules, normalize_title, passes_rules, rebuild_abstract, record_quality

# Same shape as the settings block in config/syllabus.yaml
SETTINGS = {
    "language": "en",
    "keep_types": ["article", "review", "preprint", "conference-paper"],
    "venue_lists": ["cwts-core"],
    "venue_exempt_types": ["preprint", "conference-paper"],
}


def make_paper(**changes):
    """A paper that passes every rule; pass keyword arguments to change single fields."""
    paper = {
        "type": "article",
        "doi": "https://doi.org/10.1000/example",
        "abstract_inverted_index": {"A": [0], "short": [1], "abstract": [2]},
        "language": "en",
        "is_retracted": False,
        "primary_location": {"source": {"listed_in": ["cwts-core", "doaj"]}},
    }
    paper.update(changes)
    return paper


# rebuild_abstract


def test_rebuild_abstract_puts_words_in_order():
    inverted_index = {"world": [1], "Hello": [0], "again": [2]}
    assert rebuild_abstract(inverted_index) == "Hello world again"


def test_rebuild_abstract_repeated_words():
    inverted_index = {"the": [0, 3], "cat": [1], "saw": [2], "dog": [4]}
    assert rebuild_abstract(inverted_index) == "the cat saw the dog"


def test_rebuild_abstract_none():
    assert rebuild_abstract(None) is None


def test_rebuild_abstract_empty():
    assert rebuild_abstract({}) == ""


# check_rules and passes_rules


def test_good_paper_passes_all_rules():
    paper = make_paper()
    assert not any(check_rules(paper, SETTINGS).values())
    assert passes_rules(paper, SETTINGS)


def test_bad_type():
    paper = make_paper(type="dataset")
    assert check_rules(paper, SETTINGS)["bad_type"]
    assert not passes_rules(paper, SETTINGS)


def test_no_abstract():
    paper = make_paper(abstract_inverted_index=None)
    assert check_rules(paper, SETTINGS)["no_abstract"]
    assert not passes_rules(paper, SETTINGS)


def test_not_english():
    paper = make_paper(language="de")
    assert check_rules(paper, SETTINGS)["not_english"]
    assert not passes_rules(paper, SETTINGS)


def test_retracted():
    paper = make_paper(is_retracted=True)
    assert check_rules(paper, SETTINGS)["retracted"]
    assert not passes_rules(paper, SETTINGS)


def test_article_not_on_venue_list():
    paper = make_paper(primary_location={"source": {"listed_in": ["doaj"]}})
    assert check_rules(paper, SETTINGS)["not_listed"]
    assert not passes_rules(paper, SETTINGS)


def test_article_without_source_is_not_listed():
    paper = make_paper(primary_location=None)
    assert check_rules(paper, SETTINGS)["not_listed"]


@pytest.mark.parametrize("paper_type", ["preprint", "conference-paper"])
def test_exempt_types_do_not_need_a_listed_venue(paper_type):
    paper = make_paper(type=paper_type, primary_location=None)
    assert not check_rules(paper, SETTINGS)["not_listed"]
    assert passes_rules(paper, SETTINGS)


# normalize_title


def test_normalize_title_ignores_case_and_punctuation():
    a = normalize_title("Deep Learning: A Review!")
    b = normalize_title("deep learning - a review")
    assert a == b == "deep learning a review"


def test_normalize_title_ignores_extra_spaces():
    assert normalize_title("  Random   Forests ") == "random forests"


def test_normalize_title_none():
    assert normalize_title(None) == ""


def test_normalize_title_keeps_different_titles_apart():
    assert normalize_title("Random Forests") != normalize_title("Random Fields")


@pytest.mark.xfail(reason="non-Latin letters are removed, so different titles become the same empty key")
def test_normalize_title_non_latin_titles_stay_different():
    assert normalize_title("ベイズ統計") != normalize_title("機械学習")


# record_quality


def test_record_quality_prefers_doi():
    with_doi = make_paper()
    without_doi = make_paper(doi=None)
    assert record_quality(with_doi) > record_quality(without_doi)


def test_record_quality_prefers_longer_abstract_when_doi_is_equal():
    short = make_paper(abstract_inverted_index={"one": [0]})
    longer = make_paper(abstract_inverted_index={"one": [0], "two": [1], "three": [2]})
    assert record_quality(longer) > record_quality(short)


def test_record_quality_doi_beats_abstract_length():
    doi_short = make_paper(abstract_inverted_index={"one": [0]})
    no_doi_long = make_paper(doi=None, abstract_inverted_index={"one": [0], "two": [1], "three": [2]})
    assert record_quality(doi_short) > record_quality(no_doi_long)


def test_record_quality_missing_abstract():
    assert record_quality(make_paper(abstract_inverted_index=None)) == (True, 0)
