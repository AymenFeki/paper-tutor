"""Tests for the corpus helpers: abstracts, credibility rules, titles, record quality and dedup."""

import pytest

from paper_tutor.corpus import (
    abstract_word_count,
    arxiv_id,
    author_names,
    check_rules,
    deduplicate,
    has_wrong_arxiv_title,
    is_known,
    mentions_keywords,
    normalize_title,
    passes_rules,
    rebuild_abstract,
    record_quality,
    select_seed_papers,
    select_top,
    split_by_year_window,
)

# Same shape as the settings block in config/syllabus.yaml
SETTINGS = {
    "language": "en",
    "keep_types": ["article", "review", "preprint", "conference-paper"],
    "venue_lists": ["cwts-core"],
    "venue_exempt_types": ["preprint", "conference-paper"],
    "min_abstract_words": 30,
    "preprint_servers": ["arXiv (Cornell University)", "SSRN Electronic Journal"],
}

LONG_ABSTRACT = {f"word{i}": [i] for i in range(40)}  # 40 different words
ARXIV = {"source": {"display_name": "arXiv (Cornell University)"}}  # a trusted preprint server


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
    paper = make_paper(type=paper_type, primary_location=ARXIV)
    assert not check_rules(paper, SETTINGS)["not_listed"]
    assert passes_rules(paper, SETTINGS)


def test_preprint_from_trusted_server_passes():
    paper = make_paper(type="preprint", primary_location=ARXIV)
    assert not check_rules(paper, SETTINGS)["untrusted_preprint"]
    assert passes_rules(paper, SETTINGS)


def test_preprint_from_untrusted_server_fails():
    zenodo = {"source": {"display_name": "Zenodo (CERN European Organization for Nuclear Research)"}}
    paper = make_paper(type="preprint", primary_location=zenodo)
    assert check_rules(paper, SETTINGS)["untrusted_preprint"]
    assert not passes_rules(paper, SETTINGS)


def test_preprint_without_source_is_untrusted():
    assert check_rules(make_paper(type="preprint", primary_location=None), SETTINGS)["untrusted_preprint"]


def test_untrusted_preprint_rule_ignores_other_types():
    # Articles are checked against the venue list instead, whatever their source is called
    paper = make_paper(primary_location={"source": {"display_name": "Zenodo", "listed_in": ["cwts-core"]}})
    assert not check_rules(paper, SETTINGS)["untrusted_preprint"]


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


# seed areas: arxiv_id, mentions_keywords, has_wrong_arxiv_title and select_seed_papers


def test_arxiv_id():
    assert arxiv_id("https://doi.org/10.48550/arxiv.2005.11401") == "2005.11401"
    assert arxiv_id("https://doi.org/10.48550/arXiv.2201.11903") == "2201.11903"
    assert arxiv_id("https://doi.org/10.1000/example") is None
    assert arxiv_id(None) is None


def test_mentions_keywords_in_title_or_abstract():
    in_title = make_paper(title="Large Language Models as Agents")
    in_abstract = make_paper(abstract_inverted_index={"We": [0], "prompt": [1], "models": [2]})
    off_topic = make_paper(title="Bootstrap confidence intervals")
    keywords = ["language model", "prompt"]
    assert mentions_keywords(in_title, keywords)
    assert mentions_keywords(in_abstract, keywords)
    assert not mentions_keywords(off_topic, keywords)


def test_mentions_keywords_matches_at_word_start_only():
    # "llm" must not match inside another word, but "agent" matches "agents"
    assert not mentions_keywords(make_paper(title="A survey of Tallmadge county"), ["llm"])
    assert mentions_keywords(make_paper(title="Cooperative agents"), ["agent"])


ARXIV_TITLES = {"2201.11903": "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"}


def test_wrong_arxiv_title_is_detected():
    corrupted = make_paper(doi="https://doi.org/10.48550/arxiv.2201.11903", title="BNAI, NO-TOKEN, and MIND-UNITY")
    assert has_wrong_arxiv_title(corrupted, ARXIV_TITLES)


def test_right_arxiv_title_or_no_arxiv_doi_is_fine():
    correct = make_paper(doi="https://doi.org/10.48550/arxiv.2201.11903",
                         title="Chain-of-thought prompting elicits reasoning in large language models")
    assert not has_wrong_arxiv_title(correct, ARXIV_TITLES)
    assert not has_wrong_arxiv_title(make_paper(), ARXIV_TITLES)


def seed_candidate(paper_id, citations, title="A Language Model Paper", **changes):
    return make_paper(id=paper_id, title=title, cited_by_count=citations, type="preprint",
                      primary_location=ARXIV, **changes)


def test_select_seed_papers_filters_sorts_and_caps():
    papers = [
        seed_candidate("W1", 10),
        seed_candidate("W2", 500),
        seed_candidate("W3", 300, title="Vision Transformers for Images"),  # off topic
        seed_candidate("W4", 900, language="de"),                          # breaks a rule
        seed_candidate("W5", 800, doi="https://doi.org/10.48550/arxiv.2201.11903"),  # wrong arXiv title
        seed_candidate("W6", 200),
        seed_candidate("W2", 500),                                          # same paper twice
    ]
    selected = select_seed_papers(papers, SETTINGS, ["language model"], ARXIV_TITLES, max_papers=2)
    assert [paper["id"] for paper in selected] == ["W2", "W6"]


# weekly refresh: author_names, is_known and select_top


def test_author_names_in_author_order():
    paper = make_paper(authorships=[{"author": {"display_name": "Ada"}}, {"author": {"display_name": "Bob"}}])
    assert author_names(paper) == ["Ada", "Bob"]


def test_author_names_without_authorships():
    assert author_names(make_paper()) == []
    assert author_names(make_paper(authorships=None)) == []


KNOWN_IDS = {"W1"}
KNOWN_TITLES = {"random forests"}


def test_is_known_by_short_id():
    assert is_known(make_paper(id="https://openalex.org/W1", title="Something New"), KNOWN_IDS, KNOWN_TITLES)


def test_is_known_by_normalised_title():
    # A different OpenAlex record of a paper we already have, e.g. a new preprint version
    assert is_known(make_paper(id="https://openalex.org/W2", title="Random Forests!"), KNOWN_IDS, KNOWN_TITLES)


def test_new_paper_is_not_known():
    assert not is_known(make_paper(id="https://openalex.org/W2", title="Random Fields"), KNOWN_IDS, KNOWN_TITLES)


def test_empty_title_is_never_a_match():
    paper = make_paper(id="https://openalex.org/W2", title="!!!")
    assert not is_known(paper, KNOWN_IDS, KNOWN_TITLES | {""})


def test_select_top_applies_threshold_and_cap():
    ranked = [("a", 0.9), ("b", 0.8), ("c", 0.76), ("d", 0.7)]
    assert select_top(ranked, min_score=0.75, max_papers=2) == [("a", 0.9), ("b", 0.8)]
    assert select_top(ranked, min_score=0.75, max_papers=20) == [("a", 0.9), ("b", 0.8), ("c", 0.76)]


def test_select_top_threshold_is_inclusive():
    assert select_top([("a", 0.75)], min_score=0.75, max_papers=20) == [("a", 0.75)]


def test_select_top_nothing_good_enough():
    assert select_top([("a", 0.6)], min_score=0.75, max_papers=20) == []
    assert select_top([], min_score=0.75, max_papers=20) == []
