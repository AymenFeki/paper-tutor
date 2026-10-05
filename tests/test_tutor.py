"""Tests for the tutor's routing: when to refuse."""

from paper_tutor.tutor import route_papers


def test_route_papers():
    assert route_papers({"papers": [{"id": "W1"}]}) == "answer"
    assert route_papers({"papers": []}) == "no_papers"
