"""Tests for the tutor's routing: when to ask for clarification and when to refuse."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from paper_tutor.tutor import is_vague, route_papers, route_question


@pytest.mark.parametrize("question", [
    "When does it fail?",
    "How does it work?",
    "Can you explain this?",
    "Is that biased?",
    "Why do they differ?",
])
def test_vague_questions(question):
    assert is_vague(question)


@pytest.mark.parametrize("question", [
    "When does the Jeffreys prior fail?",
    "What is the lasso and when should I use it?",
    "What is bootstrap?",                  # short, but names its topic
    "how should I choose a prior distribution?",
    "Is this estimator unbiased?",         # names what "this" refers to
])
def test_clear_questions(question):
    assert not is_vague(question)


def test_route_vague_first_question_to_clarify():
    state = {"messages": [HumanMessage("When does it fail?")]}
    assert route_question(state) == "clarify"


def test_route_clear_first_question_to_search():
    state = {"messages": [HumanMessage("When does the Jeffreys prior fail?")]}
    assert route_question(state) == "rewrite"


def test_route_vague_follow_up_to_search():
    # With a conversation, "it" can be resolved by the rewrite step
    state = {"messages": [
        HumanMessage("What is the Jeffreys prior?"),
        AIMessage("The Jeffreys prior is ... [1]"),
        HumanMessage("When does it fail?"),
    ]}
    assert route_question(state) == "rewrite"


def test_route_papers():
    assert route_papers({"papers": [{"id": "W1"}]}) == "answer"
    assert route_papers({"papers": []}) == "no_papers"
