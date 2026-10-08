"""Daily quiz: a multiple-choice question about a paper for a random learn item from the syllabus.

The answer key is checked by a second LLM call that answers the question without seeing the key (#24).
"""

import random
from typing import Annotated

from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError, field_validator

from paper_tutor.llm import build_llm


class QuizQuestion(BaseModel):
    """One multiple-choice question; the limits are Telegram's limits for quiz polls."""

    question: str = Field(
        max_length=300,
        description="A question that tests understanding of the paper's main idea",
    )
    options: list[Annotated[str, Field(max_length=100)]] = Field(
        min_length=4,
        max_length=4,
        description="Four answer options, exactly one of them correct",
    )
    correct_index: int = Field(ge=0, le=3, description="Position of the correct option (0 to 3)")
    explanation: str = Field(
        max_length=200,
        description="Why the correct option is right, based only on the abstract",
    )

    @field_validator("question")
    @classmethod
    def no_reference_to_paper(cls, value):
        """Reject questions that can only be answered by someone who read this specific paper."""
        if any(word in value.lower() for word in ("paper", "study", "authors", "this article", "proposed", "this method", "this approach", "presented")):
            raise ValueError("the question must not refer to the paper")
        return value

    @field_validator("question", "explanation")
    @classmethod
    def english_only(cls, value):
        """Reject text with Chinese characters, which qwen3 sometimes drifts into."""
        if any("\u4e00" <= char <= "\u9fff" for char in value):
            raise ValueError("the text must be in English")
        return value


class Answer(BaseModel):
    """The checker's own answer to a quiz question, given without the answer key."""

    correct_index: int = Field(
        ge=-1,
        le=3,
        description="Position of the only correct option (0 to 3), or -1 if no option or more than one is correct",
    )


class QuizError(Exception):
    """No valid quiz question with a confirmed answer key after all attempts."""


def syllabus_paper(config, retrieve):
    """A paper for a random learn item from the syllabus, found by the normal retriever."""
    items = [item for area in config["areas"].values() if area["active"] for item in area["learn"]]
    item = random.choice(items)
    query = random.choice(item["queries"])
    papers = [p for p in retrieve(query, k=5) if len(p["abstract"] or "") > 800]
    if not papers:
        raise LookupError(f"no paper with a long abstract for '{query}'")
    return item["name"], random.choice(papers)


SYSTEM = (
    "You write one multiple-choice quiz question for a statistics and data science student. "
    "Use the paper's title and abstract below only as inspiration: the question must test a concept "
    "or method that the paper uses, and must be answerable by a student who knows that topic "
    "without having read this paper. Never refer to the paper, the study or the authors. "
    "The question must be self-contained: name the model or setting it is about.\n\n"
    "Bad: 'What is the main contribution of the paper?'\n"
    "Good: 'Why can copula models price CDO tranches without Monte Carlo simulation?'\n\n"
    "Give exactly four options. Exactly one may be correct; the others must be clearly wrong for "
    "a student who knows the topic, but still plausible. Keep the question under 200 characters, "
    "each option under 70 characters (one short phrase) and the explanation under 150 characters "
    "(one or two short sentences). Write in English only."
)

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM), ("human", "Title: {title}\n\nAbstract: {abstract}")]
)


CHECK_SYSTEM = (
    "You check a multiple-choice quiz question for a statistics and data science student. "
    "Answer the question yourself, using the abstract below and your own knowledge. "
    "Give the position (0 to 3) of the only correct option. If no option is correct, or if more "
    "than one option could be correct, answer -1."
)

CHECK_PROMPT = ChatPromptTemplate.from_messages(
    [("system", CHECK_SYSTEM), ("human", "Abstract: {abstract}\n\nQuestion: {question}\n\nOptions:\n{options}")]
)


def key_is_confirmed(checker, quiz, paper):
    """True if the checker, answering without the key, picks the same option as the quiz's answer key."""
    options = "\n".join(f"{i}: {option}" for i, option in enumerate(quiz.options))
    answer = checker.invoke({"abstract": paper["abstract"], "question": quiz.question, "options": options})
    return answer.correct_index == quiz.correct_index


def first_confirmed(generate, confirm, attempts):
    """Generate quiz questions until one is valid and confirmed; returns it and the number of attempts used.

    Invalid LLM output (too long, mentions the paper, not JSON) counts as a failed attempt, like a
    question whose answer key the checker does not confirm.
    """
    for attempt in range(1, attempts + 1):
        try:
            quiz = generate()
            if confirm(quiz):
                return quiz, attempt
        except (ValidationError, OutputParserException):
            continue
    raise QuizError(f"no quiz question with a confirmed answer key after {attempts} attempts")


def make_quiz(paper, llm_cfg, attempts=3):
    """Ask the LLM for a QuizQuestion about the paper, keep the first one whose answer key a second call confirms.

    The question is written at temperature 0.7 (variety); the check runs at the config temperature (0)
    and never sees the answer key. Then the options are shuffled.
    """
    writer = PROMPT | build_llm(llm_cfg, temperature=0.7).with_structured_output(QuizQuestion)
    checker = CHECK_PROMPT | build_llm(llm_cfg).with_structured_output(Answer)

    quiz, used = first_confirmed(
        lambda: writer.invoke({"title": paper["title"], "abstract": paper["abstract"]}),
        lambda quiz: key_is_confirmed(checker, quiz, paper),
        attempts,
    )

    correct = quiz.options[quiz.correct_index]
    options = list(quiz.options)
    random.shuffle(options)
    return {
        "question": quiz.question,
        "options": options,
        "correct_index": options.index(correct),
        "explanation": quiz.explanation,
        "paper": {"id": paper["id"], "title": paper["title"], "year": paper["year"]},
        "attempts": used,
    }