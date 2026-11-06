"""Tests for the quiz's answer-key check (fake generator and checker, no LLM)."""

import pytest
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import RunnableLambda
from langsmith import tracing_context

from paper_tutor import quiz as quiz_module
from paper_tutor.llm import build_llm
from paper_tutor.quiz import Answer, QuizError, QuizQuestion, first_confirmed, key_is_confirmed, make_quiz

GROQ_OTPM_LIMIT = 1000  # output tokens per minute of qwen3.8-27b on Groq's free tier


def question(correct_index=0):
    return QuizQuestion(question="Which penalty does the lasso use?", options=["L1", "L2", "L0", "None"],
                        correct_index=correct_index, explanation="The lasso penalises the sum of absolute values.")


class FakeChecker:
    """Answers every question with a fixed option, and remembers what it was shown."""

    def __init__(self, index):
        self.index = index

    def invoke(self, inputs):
        self.inputs = inputs
        return Answer(correct_index=self.index)


def test_key_is_confirmed_only_when_the_checker_picks_the_same_option():
    paper = {"abstract": "We propose the lasso."}
    assert key_is_confirmed(FakeChecker(0), question(0), paper)
    assert not key_is_confirmed(FakeChecker(1), question(0), paper)
    assert not key_is_confirmed(FakeChecker(-1), question(0), paper)


def test_checker_never_sees_the_answer_key():
    checker = FakeChecker(0)
    key_is_confirmed(checker, question(0), {"abstract": "We propose the lasso."})
    assert set(checker.inputs) == {"abstract", "question", "options"}
    assert "0: L1" in checker.inputs["options"]


def test_first_confirmed_skips_rejected_and_invalid_questions():
    outputs = iter([OutputParserException("not JSON"), question(1), question(0)])

    def generate():
        output = next(outputs)
        if isinstance(output, Exception):
            raise output
        return output

    quiz, attempts = first_confirmed(generate, lambda quiz: quiz.correct_index == 0, attempts=3)
    assert (quiz.correct_index, attempts) == (0, 3)


def test_first_confirmed_gives_up_after_all_attempts():
    with pytest.raises(QuizError):
        first_confirmed(lambda: question(1), lambda quiz: False, attempts=2)


def test_make_quiz_caps_both_llm_calls_below_groqs_output_limit(monkeypatch):
    """Without max_tokens Groq estimates the output, sometimes above its limit, and rejects the request (429)."""

    class FakeLLM:
        def __init__(self, output):
            self.output = output

        def with_structured_output(self, schema):
            return RunnableLambda(lambda _: self.output)

    outputs = iter([question(0), Answer(correct_index=0)])  # the writer is built first, then the checker
    caps = []

    def fake_build_llm(llm_cfg, temperature=None, max_tokens=None):
        caps.append(max_tokens)
        return FakeLLM(next(outputs))

    monkeypatch.setattr(quiz_module, "build_llm", fake_build_llm)
    with tracing_context(enabled=False):  # the fake chains must not reach LangSmith
        make_quiz({"id": "W1", "title": "Lasso", "year": 1996, "abstract": "We propose the lasso."}, {})

    assert len(caps) == 2
    assert all(cap is not None and cap < GROQ_OTPM_LIMIT for cap in caps)


def test_build_llm_passes_max_tokens_to_groq_and_leaves_it_open_by_default(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "not-a-real-key")
    llm_cfg = {"provider": "groq", "groq_model": "qwen/qwen3.8-27b", "temperature": 0, "reasoning": False}
    assert build_llm(llm_cfg, max_tokens=500).max_tokens == 500
    assert build_llm(llm_cfg).max_tokens is None  # answers in /ask and /chat are not capped
