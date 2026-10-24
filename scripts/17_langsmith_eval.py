"""Retrieval and answer evaluation as LangSmith experiments, on the questions of eval/questions.yaml.

The dataset "paper-tutor-retrieval" holds one example per question: the question as input and
the relevant paper ids as reference output, with the pooled judgments in eval/judgments.yaml
(the same ground truth as scripts/08_pooled_evaluate.py). Syncing is idempotent: examples are
keyed by their question, so a re-run only adds, updates or removes what changed in the files.

Each experiment runs only the retriever (no LLM) with the retrieval settings in the config,
reranking switched on or off, and scores the top 5 papers: hit@5 and reciprocal rank per
question, hit rate and MRR over all questions. Unjudged papers count as not relevant, so
like the pooled numbers these are lower bounds.

The answer experiment runs the full tutor on 10 of the questions (the same as
scripts/11_faithfulness.py, dataset "paper-tutor-answers") and lets the Groq model judge
groundedness: is every claim in the answer supported by the cited papers? The judge is the
model that wrote the answers, so this is a rough signal; check its reasoning on a sample.

    uv run python scripts/17_langsmith_eval.py dataset          # create or update both datasets
    uv run python scripts/17_langsmith_eval.py retrieval        # experiments rerank-on and rerank-off
    uv run python scripts/17_langsmith_eval.py retrieval off    # only one of them
    uv run python scripts/17_langsmith_eval.py answers          # experiment tutor-groundedness
"""

import re
import sys
import uuid
from pathlib import Path

import yaml
from groq import RateLimitError
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langsmith import Client

from paper_tutor.corpus import load_config
from paper_tutor.embed import active_model, load_model
from paper_tutor.evaluation import first_relevant_rank, pooled_relevant
from paper_tutor.llm import build_llm
from paper_tutor.rag import NO_PAPERS_MESSAGE, build_retriever, format_context
from paper_tutor.tracing import trace_metadata
from paper_tutor.tutor import build_tutor

RETRIEVAL_DATASET = "paper-tutor-retrieval"
ANSWER_DATASET = "paper-tutor-answers"
K = 5

GROUNDEDNESS_SYSTEM = """You check whether a study tutor's answer is grounded in its sources.
You get the question, the numbered sources (title and abstract of research papers) and the answer,
which cites sources as [1], [2] and so on.

Is every claim in the answer supported by the sources it cites?
- A claim is supported only if a cited source's title or abstract states it or it follows directly from it.
- A claim that is true but comes from outside knowledge, not from the cited sources, is NOT supported.
- A claim without a citation is NOT supported, unless it only restates the question or says that
  the sources do not cover something.
- Citing a source number that does not exist is NOT supported.

Reply in exactly this format, nothing else:
reasoning: <one to three sentences; if something is unsupported, name the first unsupported claim>
is_grounded: <true only if every claim is supported, otherwise false>"""

GROUNDEDNESS_HUMAN = """Question: {question}

Sources:
{context}

Answer:
{answer}"""

# Groq's free tier allows only so many tokens per minute: wait and retry instead of failing the example
RATE_LIMIT_RETRY = {"retry_if_exception_type": (RateLimitError,), "stop_after_attempt": 6,
                    "exponential_jitter_params": {"initial": 10, "max": 60}}


def parse_verdict(text):
    """{"is_grounded": bool, "reasoning": str} from the judge's reply.

    Parsed from plain text: with structured output (tool calling or JSON schema) the Groq model
    often misnames or drops the boolean field. Raises ValueError if the reply has no verdict.
    """
    verdict = re.search(r"is_grounded\W*(true|false)", text, re.IGNORECASE)
    if not verdict:
        raise ValueError(f"no is_grounded verdict in the judge's reply: {text[:200]!r}")
    before = text[:verdict.start()]
    reasoning = re.search(r"reasoning\W*(.*)", before, re.IGNORECASE | re.DOTALL)
    reasoning = reasoning.group(1) if reasoning else before
    return {"is_grounded": verdict.group(1).lower() == "true", "reasoning": reasoning.strip().rstrip("*").strip()}


def retrieval_examples(dataset_id):
    """The 34 retrieval examples: question in, relevant paper ids (pooled judgments) as reference."""
    questions = yaml.safe_load(Path("eval/questions.yaml").read_text())
    judgments = yaml.safe_load(Path("eval/judgments.yaml").read_text())
    return {
        uuid.uuid5(dataset_id, q["question"]): {
            "inputs": {"question": q["question"]},
            "outputs": {"relevant": sorted(pooled_relevant(q, judgments))},
            "metadata": {"area": q["area"], "paper_title": q["paper_title"], "original_relevant": q["relevant"]},
        }
        for q in questions
    }


def answer_examples(dataset_id):
    """The 10 answer examples (every third question, as in 11_faithfulness.py); no reference answer needed."""
    questions = yaml.safe_load(Path("eval/questions.yaml").read_text())[::3][:10]
    return {
        uuid.uuid5(dataset_id, q["question"]): {
            "inputs": {"question": q["question"]},
            "outputs": {},
            "metadata": {"area": q["area"], "paper_title": q["paper_title"]},
        }
        for q in questions
    }


def sync_dataset(client, name, description, examples_for):
    """Create the dataset if needed and make its examples match examples_for(dataset_id).

    Example ids are derived from the dataset and the question, so re-running adds nothing twice.
    """
    if client.has_dataset(dataset_name=name):
        dataset = client.read_dataset(dataset_name=name)
    else:
        dataset = client.create_dataset(name, description=description)
    wanted = examples_for(dataset.id)
    existing = {example.id: example for example in client.list_examples(dataset_id=dataset.id)}

    def unchanged(example, want):
        metadata = {key: (example.metadata or {}).get(key) for key in want["metadata"]}
        return (example.inputs, example.outputs, metadata) == (want["inputs"], want["outputs"], want["metadata"])

    new = [{"id": i, **want} for i, want in wanted.items() if i not in existing]
    changed = [{"id": i, **want} for i, want in wanted.items() if i in existing and not unchanged(existing[i], want)]
    stale = [i for i in existing if i not in wanted]  # questions no longer in eval/questions.yaml
    if new:
        client.create_examples(dataset_id=dataset.id, examples=new)
    if changed:
        client.update_examples(dataset_id=dataset.id, updates=changed)
    if stale:
        client.delete_examples(stale)
    print(f"Dataset {name}: {len(wanted)} examples ({len(new)} added, {len(changed)} updated, "
          f"{len(stale)} removed)\n{dataset.url}")
    return dataset


def relevant_rank(outputs, reference_outputs):
    """Rank of the first relevant paper in the retrieved top K, or None."""
    return first_relevant_rank(outputs["paper_ids"][:K], set(reference_outputs["relevant"]))


def hit_at_k(outputs, reference_outputs):
    return {"key": f"hit@{K}", "score": int(relevant_rank(outputs, reference_outputs) is not None)}


def reciprocal_rank(outputs, reference_outputs):
    rank = relevant_rank(outputs, reference_outputs)
    return {"key": "reciprocal_rank", "score": 1 / rank if rank else 0.0}


def hit_rate(outputs, reference_outputs):
    """Share of questions with a relevant paper in the top K."""
    ranks = [relevant_rank(o, r) for o, r in zip(outputs, reference_outputs)]
    return {"key": f"hit_rate@{K}", "score": sum(rank is not None for rank in ranks) / len(ranks)}


def mean_reciprocal_rank(outputs, reference_outputs):
    """MRR over the top K (a question without a relevant paper in the top K counts 0)."""
    ranks = [relevant_rank(o, r) for o, r in zip(outputs, reference_outputs)]
    return {"key": f"mrr@{K}", "score": sum(1 / rank for rank in ranks if rank) / len(ranks)}


def run_retrieval_experiment(client, config, embed_model, rerank):
    """One experiment over the dataset; returns its name, hits and MRR."""
    config = {**config, "retrieval": {**config["retrieval"], "rerank": rerank}}
    retrieve = build_retriever(config, embed_model)

    def target(inputs):
        papers = retrieve(inputs["question"], k=K)
        return {"paper_ids": [p["id"] for p in papers], "titles": [p["title"] for p in papers]}

    label = "rerank-on" if rerank else "rerank-off"
    results = client.evaluate(
        target,
        data=RETRIEVAL_DATASET,
        evaluators=[hit_at_k, reciprocal_rank],
        summary_evaluators=[hit_rate, mean_reciprocal_rank],
        experiment_prefix=label,
        description=f"Retriever only, top {K}, reranking {'on' if rerank else 'off'}; pooled relevance judgments.",
        metadata={**trace_metadata(config), "k": K, "relevance": "pooled"},
        max_concurrency=0,  # one question at a time: the models run locally
    )
    scores = {}
    for row in results:
        for result in row["evaluation_results"]["results"]:
            scores.setdefault(result.key, []).append(result.score)
    n = len(scores[f"hit@{K}"])
    return {"label": label, "experiment": results.experiment_name, "n": n,
            "hits": sum(scores[f"hit@{K}"]), "mrr": sum(scores["reciprocal_rank"]) / n}


def run_answer_experiment(client, config):
    """The full tutor on the answer dataset, judged for groundedness; returns its name and grounded count."""
    retrieve = build_retriever(config)
    tutor = build_tutor(config, retrieve)
    prompt = ChatPromptTemplate.from_messages([("system", GROUNDEDNESS_SYSTEM), ("human", GROUNDEDNESS_HUMAN)])
    judge = (prompt | build_llm(config["llm"]) | StrOutputParser()).with_retry(**RATE_LIMIT_RETRY)
    # a new thread for every attempt, so a retried question doesn't see its first attempt
    ask = RunnableLambda(lambda question: tutor.invoke(
        {"messages": [("user", question)]}, {"configurable": {"thread_id": str(uuid.uuid4())}},
    )).with_retry(**RATE_LIMIT_RETRY)

    def target(inputs):
        result = ask.invoke(inputs["question"])
        return {"answer": result["messages"][-1].content, "context": format_context(result["papers"]),
                "paper_ids": [p["id"] for p in result["papers"]]}

    def groundedness(inputs, outputs):
        if not outputs["paper_ids"] and outputs["answer"] == NO_PAPERS_MESSAGE:
            return {"key": "is_grounded", "score": True, "comment": "No papers found; the tutor refused to answer."}
        verdict = parse_verdict(judge.invoke({"question": inputs["question"], "context": outputs["context"],
                                              "answer": outputs["answer"]}))
        return {"key": "is_grounded", "score": verdict["is_grounded"], "comment": verdict["reasoning"]}

    results = client.evaluate(
        target,
        data=ANSWER_DATASET,
        evaluators=[groundedness],
        experiment_prefix="tutor-groundedness",
        description="Full tutor; LLM judge (the Groq answer model): is every claim supported by the cited papers?",
        metadata={**trace_metadata(config), "judge_model": config["llm"]["groq_model"]},
        max_concurrency=1,  # one question at a time: Groq free-tier rate limits
    )
    verdicts = {}  # example id -> score (None if the judge failed)
    for row in results:
        scores = [result.score for result in row["evaluation_results"]["results"]]
        verdicts[row["example"].id] = next((score for score in scores if score is not None), None)
    judged = [score for score in verdicts.values() if score is not None]
    return {"experiment": results.experiment_name, "n": len(verdicts), "judged": len(judged),
            "grounded": sum(bool(score) for score in judged)}


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "retrieval"
    client = Client()
    dataset = sync_dataset(
        client, RETRIEVAL_DATASET,
        "paper-tutor retrieval test set: eval/questions.yaml, relevant papers from the pooled judgments in "
        "eval/judgments.yaml. Synced by scripts/17_langsmith_eval.py.",
        retrieval_examples,
    )
    sync_dataset(
        client, ANSWER_DATASET,
        "paper-tutor answer test set: 10 questions from eval/questions.yaml (as in scripts/11_faithfulness.py), "
        "judged for groundedness. Synced by scripts/17_langsmith_eval.py.",
        answer_examples,
    )

    if command == "answers":
        run = run_answer_experiment(client, load_config())
        print(f"\n{run['experiment']}: {run['grounded']}/{run['judged']} judged answers grounded "
              f"({run['n'] - run['judged']} of {run['n']} not judged: judge failed)")
    elif command == "retrieval":
        variants = {"on": [True], "off": [False]}.get(sys.argv[2] if len(sys.argv) > 2 else "", [True, False])
        config = load_config()
        embed_model = load_model(active_model(config)[1])  # shared by both retrievers
        runs = [run_retrieval_experiment(client, config, embed_model, rerank) for rerank in variants]

        print(f"\n{'experiment':<28}{'hit@' + str(K):>10}{'MRR@' + str(K):>8}")
        for run in runs:
            print(f"{run['experiment']:<28}{run['hits']:>4}/{run['n']:<5}{run['mrr']:>8.3f}")
        sessions = ",".join(str(client.read_project(project_name=run["experiment"]).id) for run in runs)
        print(f"\nCompare: {dataset.url}/compare?selectedSessions={sessions}")
    elif command != "dataset":
        raise SystemExit(f"Unknown command {command!r}: use dataset, retrieval or answers")
