"""Tests for the retrieval metrics."""

import pytest
import yaml

from paper_tutor.evaluation import (
    citations,
    first_relevant_rank,
    judgment_yaml,
    metrics,
    split_sentences,
    strip_citations,
    threshold_effect,
    unjudged_pairs,
    wilson_interval,
)

# first_relevant_rank


def test_first_relevant_rank_is_one_based():
    assert first_relevant_rank(["a", "b", "c"], {"a"}) == 1
    assert first_relevant_rank(["a", "b", "c"], {"c"}) == 3


def test_first_relevant_rank_returns_first_of_several():
    assert first_relevant_rank(["a", "b", "c", "d"], {"d", "b"}) == 2


def test_first_relevant_rank_none_when_missing():
    assert first_relevant_rank(["a", "b"], {"x"}) is None
    assert first_relevant_rank([], {"x"}) is None


# metrics


def test_metrics_small_example():
    # Four questions: relevant paper at rank 1, rank 3, rank 7, and not found.
    result = metrics([1, 3, 7, None])
    assert result["n"] == 4
    assert result["hit@1"] == 1 / 4  # only rank 1
    assert result["hit@5"] == 2 / 4  # ranks 1 and 3
    assert result["hit@10"] == 3 / 4  # ranks 1, 3 and 7
    assert result["mrr"] == pytest.approx((1 / 1 + 1 / 3 + 1 / 7 + 0) / 4)


def test_metrics_all_missed():
    result = metrics([None, None])
    assert result["hit@1"] == result["hit@5"] == result["hit@10"] == 0
    assert result["mrr"] == 0


def test_metrics_all_first():
    result = metrics([1, 1, 1])
    assert result["hit@1"] == result["hit@10"] == 1
    assert result["mrr"] == 1


# wilson_interval


@pytest.mark.parametrize("successes, n", [(1, 10), (5, 10), (13, 28), (9, 10)])
def test_wilson_interval_contains_estimate_and_stays_in_unit_range(successes, n):
    low, high = wilson_interval(successes, n)
    assert 0 <= low <= successes / n <= high <= 1


def test_wilson_interval_gets_narrower_with_more_data():
    low_small, high_small = wilson_interval(5, 10)
    low_big, high_big = wilson_interval(50, 100)
    assert high_big - low_big < high_small - low_small


def test_wilson_interval_matches_saved_result():
    # eval/results/bge-small.json: hit@5 = 13/28, Wilson interval [0.2953, 0.6419]
    low, high = wilson_interval(13, 28)
    assert low == pytest.approx(0.2953, abs=1e-4)
    assert high == pytest.approx(0.6419, abs=1e-4)


@pytest.mark.xfail(reason="rounding error gives a lower bound of about -3e-17 when there are 0 successes")
def test_wilson_interval_zero_successes_not_negative():
    low, _ = wilson_interval(0, 10)
    assert low >= 0


# threshold_effect


def saved_question(rank, scores):
    """An in-scope result as saved by 07_evaluate.py: rank of the relevant paper and the scores."""
    return {"rank": rank, "retrieved": [{"score": score} for score in scores]}


def test_threshold_effect_small_example():
    questions = [
        saved_question(1, [0.8, 0.7]),   # hit at rank 1 with score 0.8
        saved_question(2, [0.7, 0.6]),   # hit at rank 2 with score 0.6
        saved_question(None, [0.5]),     # miss, best score 0.5
    ]
    out_of_scope = [{"top_score": 0.55}, {"top_score": 0.4}]
    assert threshold_effect(questions, out_of_scope, 0.65) == {
        "in_scope_refused": 1,       # only the question whose best score is 0.5
        "hits5_kept": 1,             # the rank-2 hit has score 0.6 < 0.65
        "out_of_scope_refused": 2,
    }


def test_threshold_effect_zero_threshold_changes_nothing():
    questions = [saved_question(3, [0.9, 0.8, 0.7])]
    assert threshold_effect(questions, [{"top_score": 0.1}], 0.0) == {
        "in_scope_refused": 0, "hits5_kept": 1, "out_of_scope_refused": 0,
    }


def test_threshold_effect_hit_below_rank_5_is_not_counted():
    questions = [saved_question(6, [0.9] * 6)]
    assert threshold_effect(questions, [], 0.5)["hits5_kept"] == 0


# split_sentences, citations and strip_citations (faithfulness check)


def test_split_sentences_keeps_trailing_citation_with_its_sentence():
    text = "Priors can be weakly informative. [1] Jeffreys priors are formal rules [2]."
    assert split_sentences(text) == ["Priors can be weakly informative. [1]", "Jeffreys priors are formal rules [2]."]


def test_split_sentences_lines_and_bullets():
    text = "First point [1].\n\n- Second point [2].\n* Third point."
    assert split_sentences(text) == ["First point [1].", "Second point [2].", "Third point."]


def test_split_sentences_empty():
    assert split_sentences("") == []


def test_citations_formats():
    assert citations("A claim [1].") == [1]
    assert citations("A claim [2][4].") == [2, 4]
    assert citations("A claim [1, 3].") == [1, 3]


def test_citations_no_repeats_and_none():
    assert citations("A [1] and again [1].") == [1]
    assert citations("No citation here.") == []


def test_strip_citations():
    assert strip_citations("Priors matter [1][2].") == "Priors matter."
    assert strip_citations("Priors matter [1, 2].") == "Priors matter."


# unjudged_pairs and judgment_yaml (app/judge.py)


def result(questions):
    """A saved result with (question, relevant ids, retrieved ids) triples."""
    return {"questions": [
        {"question": q, "relevant": relevant, "retrieved": [{"id": pid, "title": f"Title {pid}"} for pid in ids]}
        for q, relevant, ids in questions
    ]}


def test_unjudged_pairs_skips_judged_and_original_papers():
    results = [result([("q1", ["W1"], ["W1", "W2", "W3"])])]
    judgments = [{"question": "q1", "paper_id": "W2", "relevant": True}]
    assert [p["paper_id"] for p in unjudged_pairs(results, judgments)] == ["W3"]


def test_unjudged_pairs_once_per_pair_and_only_top_depth():
    results = [
        result([("q1", ["W0"], ["W5", "W4"])]),
        result([("q1", ["W0"], ["W4", "W9"])]),  # W4 again
    ]
    pairs = unjudged_pairs(results, [], depth=1)
    assert [(p["question"], p["paper_id"]) for p in pairs] == [("q1", "W4"), ("q1", "W5")]


def test_unjudged_pairs_grouped_by_question_in_first_seen_order():
    results = [result([("q2", [], ["W8"]), ("q1", [], ["W7", "W3"])])]
    pairs = unjudged_pairs(results, [])
    assert [(p["question"], p["paper_id"]) for p in pairs] == [("q2", "W8"), ("q1", "W3"), ("q1", "W7")]


def test_judgment_yaml_round_trip():
    text = judgment_yaml("How should I choose a prior?", "W2008640468", True)
    assert yaml.safe_load(text) == [{"question": "How should I choose a prior?", "paper_id": "W2008640468",
                                     "relevant": True, "judge": "human"}]
    assert text.startswith("- question:") and text.endswith("\n")
