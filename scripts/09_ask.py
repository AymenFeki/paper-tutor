import sys

from paper_tutor.corpus import load_config
from paper_tutor.rag import NO_PAPERS_MESSAGE, build_chain, build_retriever, format_context

question = " ".join(sys.argv[1:])
if not question:
    raise SystemExit('Usage: uv run python scripts/09_ask.py "your question"')

config = load_config()
llm_cfg = config["llm"]

retrieve = build_retriever(config)
papers = retrieve(question, k=llm_cfg["top_k"])

chain = build_chain(llm_cfg)
if papers:
    answer = chain.invoke({"context": format_context(papers), "question": question})
else:
    answer = NO_PAPERS_MESSAGE

print(f"\nQuestion: {question}\n")
print(answer)
print("\nSources:")
for i, paper in enumerate(papers, start=1):
    print(f"[{i}] {paper['title']} ({paper['year']})  https://openalex.org/{paper['id']}")