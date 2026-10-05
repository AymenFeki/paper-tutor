import sys

from paper_tutor.corpus import load_config
from paper_tutor.embed import active_model, load_model
from paper_tutor.rag import build_chain, format_context, retrieve

question = " ".join(sys.argv[1:])
if not question:
    raise SystemExit('Usage: uv run python scripts/09_ask.py "your question"')

config = load_config()
model_key, model_cfg = active_model(config)
llm_cfg = config["llm"]

embed_model = load_model(model_cfg)
papers = retrieve(question, embed_model, model_key, model_cfg, k=llm_cfg["top_k"])

chain = build_chain(llm_cfg)
answer = chain.invoke({"context": format_context(papers), "question": question})

print(f"\nQuestion: {question}\n")
print(answer)
print("\nSources:")
for i, paper in enumerate(papers, start=1):
    print(f"[{i}] {paper['title']} ({paper['year']})  https://openalex.org/{paper['id']}")