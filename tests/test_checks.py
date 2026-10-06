"""Tests for the data-quality checks."""

import numpy as np
import pytest

from paper_tutor.checks import area_centroids, cosine_rows, lowest_share, nearest_area, review_signals


@pytest.mark.parametrize("title, abstract", [
    ("A Survey on Evaluation of Large Language Models", ""),
    ("Variable selection – A review and recommendations", ""),
    ("A Tutorial on Support Vector Machines", ""),
    ("An Introduction to Statistical Learning", ""),
    ("Some Title", "In this paper we review recent methods for causal inference."),
    ("Some Title", "This tutorial explains Gibbs sampling."),
    ("Stochastic Calculus", "These lecture notes cover Brownian motion."),
])
def test_review_signals_found(title, abstract):
    assert review_signals(title, abstract)


def test_review_signals_research_paper():
    assert review_signals("Regression Shrinkage and Selection via the Lasso",
                          "We propose a new method for estimation in linear models.") == []


def test_review_signals_known_false_positive():
    # "review" as the topic of a paper also matches: the check is rough, so it only reports
    assert review_signals("Peer review and citation counts", "")


def test_review_signals_missing_text():
    assert review_signals(None, None) == []


def test_cosine_rows():
    a = np.array([[1.0, 0.0], [1.0, 1.0]])
    b = np.array([[2.0, 0.0], [-1.0, -1.0]])
    assert cosine_rows(a, b) == pytest.approx([1.0, -1.0])


def test_lowest_share():
    scores = np.array([0.9, 0.1, 0.5, 0.3, 0.7, 0.8, 0.6, 0.4, 0.2, 0.95])
    assert lowest_share(scores, 0.2) == [1, 8]   # the two lowest, lowest first
    assert lowest_share(scores, 0.01) == [1]     # always at least one


def test_area_centroids_and_nearest_area():
    vectors = np.array([[1.0, 0.1], [1.0, -0.1], [0.1, 1.0], [-0.1, 1.0], [0.9, 0.2]])
    areas = ["stats", "stats", "finance", "finance", "finance"]  # the last one points at stats
    centroids = area_centroids(vectors, areas)
    assert set(centroids) == {"stats", "finance"}
    assert np.linalg.norm(centroids["stats"]) == pytest.approx(1.0)
    assert nearest_area(vectors[0], centroids)[0] == "stats"
    assert nearest_area(vectors[2], centroids)[0] == "finance"
    assert nearest_area(vectors[4], centroids)[0] == "stats"
