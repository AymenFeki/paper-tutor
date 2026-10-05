"""Tests for the corpus helpers: abstracts, credibility rules, titles, record quality and dedup."""

import pytest

from paper_tutor.corpus import (
    abstract_word_count,
    check_rules,
    deduplicate,
    normalize_title,
    passes_rules,
    rebuild_abstract,
    record_quality,
    split_by_year_window,
)

# Same shape as the settings block in config/syllabus.yaml
SETTINGS = {
    "language": "en",
    "keep_types": ["article", "review", "preprint", "conference-paper"],
    "venue_lists": ["cwts-core"],
    "venue_exempt_types": ["preprint", "conference-paper"],
    "min_abstract_words": 30,
}

LONG_ABSTRACT = {f"word{i}": [i] for i in range(40)}  # 40 different words


def make_paper(**changes):
    """A paper that passes every rule; pass keyword arguments to change single fields."""
    paper = {
        "id": "https://openalex.org/W1",
        "title": "A Paper",
        "publication_year": 2020,
        "type": "article",
        "doi": "https://doi.org/10.1000/example",
        "abstract_inverted_index": LONG_ABSTRACT,
        "language": "en",
        "is_retracted": False,
        "primary_location": {"source": {"display_name": "Journal X", "listed_in": ["cwts-core", "doaj"]}},
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


# abstract_word_count


def test_abstract_word_count_counts_repeated_words():
    assert abstract_word_count({"the": [0, 3], "cat": [1], "saw": [2], "dog": [4]}) == 5


def test_abstract_word_count_missing_abstract():
    assert abstract_word_count(None) == 0
    assert abstract_word_count({}) == 0


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


def test_short_abstract():
    paper = make_paper(abstract_inverted_index={"Journal": [0], "of": [1], "Statistics,": [2], "vol.": [3], "3": [4]})
    assert check_rules(paper, SETTINGS)["short_abstract"]
    assert not passes_rules(paper, SETTINGS)


def test_short_abstract_rule_off_with_zero():
    paper = make_paper(abstract_inverted_index={"Short": [0]})
    assert not check_rules(paper, {**SETTINGS, "min_abstract_words": 0})["short_abstract"]


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
    no_doi_long = make_paper(doi=None)
    assert record_quality(doi_short) > record_quality(no_doi_long)


def test_record_quality_published_version_beats_preprint_with_doi():
    journal_no_doi = make_paper(doi=None)
    preprint_with_doi = make_paper(type="preprint")
    assert record_quality(journal_no_doi) > record_quality(preprint_with_doi)


def test_record_quality_prefers_venue():
    with_venue = make_paper()
    without_venue = make_paper(primary_location=None)
    assert record_quality(with_venue) > record_quality(without_venue)


def test_record_quality_missing_abstract():
    assert record_quality(make_paper(abstract_inverted_index=None)) == (True, True, True, 0)


# split_by_year_window and deduplicate


def record(paper_id, year, title="A Paper", **changes):
    """A (paper, topic_id) record as the loader builds it."""
    return (make_paper(id=paper_id, publication_year=year, title=title, **changes), "T1")


def years(groups):
    return [[paper["publication_year"] for paper, _ in group] for group in groups]


def test_split_by_year_window_groups_close_years():
    records = [record("W3", 2012), record("W1", 2006), record("W2", 2009)]
    assert years(split_by_year_window(records, 3)) == [[2006, 2009], [2012]]


def test_split_by_year_window_zero_means_same_year_only():
    records = [record("W1", 2006), record("W2", 2006), record("W3", 2007)]
    assert years(split_by_year_window(records, 0)) == [[2006, 2006], [2007]]


def test_deduplicate_merges_across_years_and_keeps_journal_version():
    # The known example from issue #4: conference version 2006, journal version 2009
    title = "A Large-Scale Study of Failures in High-Performance Computing Systems"
    conference = record("W1", 2006, title, type="conference-paper", primary_location=None)
    journal = record("W2", 2009, title.lower())
    kept, merged = deduplicate([conference, journal], year_window=3)
    assert [paper["id"] for paper, _ in kept] == ["W2"]
    assert [(a["id"], b["id"]) for a, b in merged] == [("W2", "W1")]


def test_deduplicate_keeps_papers_outside_the_window():
    kept, merged = deduplicate([record("W1", 2000), record("W2", 2010)], year_window=3)
    assert len(kept) == 2
    assert merged == []


def test_deduplicate_keeps_different_titles():
    kept, _ = deduplicate([record("W1", 2020, "Random Forests"), record("W2", 2020, "Random Fields")], 3)
    assert len(kept) == 2


def test_deduplicate_same_id_twice_is_not_reported_as_merge():
    # The same paper can be in both the random and the top-cited sample
    kept, merged = deduplicate([record("W1", 2020), record("W1", 2020)], year_window=3)
    assert len(kept) == 1
    assert merged == []


def test_deduplicate_never_merges_papers_without_a_title():
    kept, _ = deduplicate([record("W1", 2020, title=None), record("W2", 2020, title="!!!")], year_window=3)
    assert len(kept) == 2
