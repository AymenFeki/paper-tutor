"""Daily quiz: a multiple-choice question about a paper for a random learn item from the syllabus."""

import random
from typing import Annotated

from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, ValidationError, field_validator


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
        if any(word in value.lower() for word in ("paper", "study", "authors", "this article")):
            raise ValueError("the question must not refer to the paper")
        return value


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
    "without having read this paper. Never refer to the paper, the study or the authors.\n\n"
    "Bad: 'What is the main contribution of the paper?'\n"
    "Good: 'Why can copula models price CDO tranches without Monte Carlo simulation?'\n\n"
    "Give exactly four options, exactly one correct, with plausible wrong options. Keep the "
    "question under 300 characters, each option under 100 and the explanation under 200."
)

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM), ("human", "Title: {title}\n\nAbstract: {abstract}")]
)


def make_quiz(paper, llm_cfg, attempts=3):
    """Ask the LLM for a QuizQuestion about the paper, retrying on invalid output, then shuffle the options."""
    llm = ChatOllama(model=llm_cfg["model"], temperature=0.7, reasoning=llm_cfg["reasoning"])
    chain = PROMPT | llm.with_structured_output(QuizQuestion)

    for attempt in range(attempts):
        try:
            quiz = chain.invoke({"title": paper["title"], "abstract": paper["abstract"]})
            break
        except (ValidationError, OutputParserException):
            if attempt == attempts - 1:
                raise

    correct = quiz.options[quiz.correct_index]
    options = list(quiz.options)
    random.shuffle(options)
    return {
        "question": quiz.question,
        "options": options,
        "correct_index": options.index(correct),
        "explanation": quiz.explanation,
        "paper": {"id": paper["id"], "title": paper["title"], "year": paper["year"]},
    }