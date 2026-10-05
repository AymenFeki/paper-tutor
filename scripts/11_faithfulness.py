"""Check whether the tutor's citations are supported by the papers they cite.

For 10 eval questions (every third one, so all areas are covered): ask the tutor, split the
answer into sentences, and for every cited source ask the LLM whether that source's title and
abstract support the sentence. Reports the share of supported citations and of sentences
that cite anything at all.

The answer prompt can be chosen for one run, so both prompts can be compared:
    uv run python scripts/11_faithfulness.py basic
    uv run python scripts/11_faithfulness.py strict
Results are saved to eval/faithfulness/<prompt>.json.

Caveat: the judge is the same 8B model that wrote the answers, so the numbers are a rough
signal, not ground truth. A sample should be checked by hand before relying on them.
"""

import json
import sys
from pathlib import Path

import yaml
from langchain_ollama import ChatOllama

from paper_tutor.corpus import load_config
from paper_tutor.evaluation import citations, split_sentences, strip_citations, wilson_interval
from paper_tutor.rag import build_retriever
from paper_tutor.tutor import build_tutor

JUDGE_PROMPT = """Source:
{title}
{abstract}

Claim: {claim}

Is the claim supported by the source above? Answer YES only if the source says it or it follows
directly from the source. Answer with one word: YES or NO."""

config = load_config()
if len(sys.argv) > 1:
    config["llm"]["prompt"] = sys.argv[1]
prompt_name = config["llm"]["prompt"]

questions = yaml.safe_load(Path("eval/questions.yaml").read_text())[::3][:10]
tutor = build_tutor(config, build_retriever(config))
judge = ChatOllama(model=config["llm"]["model"], temperature=0, reasoning=False)

results = []
for i, item in enumerate(questions):
    run = tutor.invoke({"messages": [("user", item["question"])]},
                       {"configurable": {"thread_id": f"faithfulness-{i}"}})
    answer, papers = run["messages"][-1].content, run["papers"]

    checks = []
    for sentence in split_sentences(answer):
        for number in citations(sentence):
            if not 1 <= number <= len(papers):
                checks.append({"sentence": sentence, "source": number, "supported": False, "reason": "no such source"})
                continue
            paper = papers[number - 1]
            verdict = judge.invoke(JUDGE_PROMPT.format(
                title=paper["title"], abstract=paper["abstract"], claim=strip_citations(sentence),
            )).content
            checks.append({"sentence": sentence, "source": number,
                           "supported": verdict.strip().upper().startswith("YES")})

    sentences = split_sentences(answer)
    results.append({
        "question": item["question"],
        "answer": answer,
        "n_sentences": len(sentences),
        "n_cited_sentences": sum(bool(citations(s)) for s in sentences),
        "checks": checks,
    })
    supported = sum(c["supported"] for c in checks)
    print(f"[{i + 1}/{len(questions)}] {supported}/{len(checks)} citations supported: {item['question'][:70]}")

n_checks = sum(len(r["checks"]) for r in results)
n_supported = sum(c["supported"] for r in results for c in r["checks"])
n_sentences = sum(r["n_sentences"] for r in results)
n_cited = sum(r["n_cited_sentences"] for r in results)
low, high = wilson_interval(n_supported, n_checks)

print(f"\nprompt: {prompt_name}, {len(results)} questions")
print(f"supported citations: {n_supported}/{n_checks} = {n_supported / n_checks:.2f} "
      f"(95% Wilson CI [{low:.2f}, {high:.2f}])")
print(f"sentences with a citation: {n_cited}/{n_sentences} = {n_cited / n_sentences:.2f}")
print("Note: the judge is the same 8B model as the tutor and is itself unreliable; "
      "treat this as a rough signal and check a sample by hand.")

out_path = Path("eval/faithfulness") / f"{prompt_name}.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps({
    "prompt": prompt_name,
    "judge": config["llm"]["model"],
    "supported_citations": n_supported,
    "citations": n_checks,
    "supported_wilson_95": [round(low, 4), round(high, 4)],
    "cited_sentences": n_cited,
    "sentences": n_sentences,
    "results": results,
}, indent=2, ensure_ascii=False))
print(f"Saved {out_path}")
