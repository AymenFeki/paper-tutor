"""Chat with the tutor in the terminal."""

from paper_tutor.corpus import load_config
from paper_tutor.embed import active_model, load_model
from paper_tutor.tutor import build_tutor

config = load_config()
model_key, model_cfg = active_model(config)
embed_model = load_model(model_cfg)
tutor = build_tutor(config, embed_model, model_key, model_cfg)

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