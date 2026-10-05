"""Chat with the tutor in the terminal."""

from paper_tutor.corpus import load_config
from paper_tutor.rag import build_retriever
from paper_tutor.tutor import build_tutor

config = load_config()
tutor = build_tutor(config, build_retriever(config))

run_config = {"configurable": {"thread_id": "terminal"}}

while True:
    question = input("\nYou: ").strip()
    if question in {"", "quit", "exit"}:
        break
    result = tutor.invoke({"messages": [("user", question)]}, run_config)
    print(f"\n(search query: {result['query']})")
    for i, paper in enumerate(result["papers"], start=1):
        print(f"  [{i}] {paper['title']} ({paper['year']})")
    print(f"\nTutor: {result['messages'][-1].content}")