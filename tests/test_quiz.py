"""Tests for the quiz's answer-key check (fake generator and checker, no LLM)."""

import pytest
from langchain_core.exceptions import OutputParserException

from paper_tutor.quiz import Answer, QuizError, QuizQuestion, first_confirmed, key_is_confirmed


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
