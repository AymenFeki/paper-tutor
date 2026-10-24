"""Retrieval evaluation as LangSmith experiments, on the 34 questions of eval/questions.yaml.

The dataset "paper-tutor-retrieval" holds one example per question: the question as input and
the relevant paper ids as reference output, with the pooled judgments in eval/judgments.yaml
(the same ground truth as scripts/08_pooled_evaluate.py). Syncing is idempotent: examples are
keyed by their question, so a re-run only adds, updates or removes what changed in the files.

Each experiment runs only the retriever (no LLM) with the retrieval settings in the config,
reranking switched on or off, and scores the top 5 papers: hit@5 and reciprocal rank per
question, hit rate and MRR over all questions. Unjudged papers count as not relevant, so
like the pooled numbers these are lower bounds.

    uv run python scripts/17_langsmith_eval.py dataset          # create or update the dataset
    uv run python scripts/17_langsmith_eval.py retrieval        # experiments rerank-on and rerank-off
    uv run python scripts/17_langsmith_eval.py retrieval off    # only one of them
"""

import sys
import uuid
from pathlib import Path

import yaml
from langsmith import Client

from paper_tutor.corpus import load_config
from paper_tutor.embed import active_model, load_model
from paper_tutor.evaluation import first_relevant_rank, pooled_relevant
from paper_tutor.rag import build_retriever
from paper_tutor.tracing import trace_metadata

DATASET = "paper-tutor-retrieval"
K = 5


def dataset_examples(dataset_id):
    """The examples the dataset should hold, by example id (derived from the dataset and the question)."""
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


def sync_dataset(client):
    """Create the dataset if needed and make its examples match the question and judgment files."""
    if client.has_dataset(dataset_name=DATASET):
        dataset = client.read_dataset(dataset_name=DATASET)
    else:
        dataset = client.create_dataset(
            DATASET,
            description="paper-tutor retrieval test set: eval/questions.yaml, relevant papers from the pooled "
                        "judgments in eval/judgments.yaml. Synced by scripts/17_langsmith_eval.py.",
        )
    wanted = dataset_examples(dataset.id)
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
    print(f"Dataset {DATASET}: {len(wanted)} examples ({len(new)} added, {len(changed)} updated, "
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
        data=DATASET,
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


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "retrieval"
    client = Client()
    dataset = sync_dataset(client)

    if command == "retrieval":
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
        raise SystemExit(f"Unknown command {command!r}: use dataset or retrieval")
