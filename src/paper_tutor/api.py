"""HTTP API for the paper tutor: semantic search and cited answers."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from paper_tutor.corpus import load_config
from paper_tutor.embed import active_model, load_model
from paper_tutor.rag import retrieve, format_context, build_chain

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the config, embedding model and LLM chain once, when the server starts."""
    config = load_config()
    model_key, model_cfg = active_model(config)
    state["config"] = config
    state["model_key"] = model_key
    state["model_cfg"] = model_cfg
    state["embed_model"] = load_model(model_cfg)
    state["chain"] = build_chain(config["llm"])
    yield
    state.clear()


app = FastAPI(title="paper-tutor", lifespan=lifespan)


class SearchRequest(BaseModel):
    question: str
    k: int = 5


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "embedding_model": state["model_key"]}


@app.post("/search")
def search(req: SearchRequest):
    papers = retrieve(req.question, state["embed_model"], state["model_key"], state["model_cfg"], k=req.k)
    return {"question": req.question, "papers": papers}


@app.post("/ask")
def ask(req: AskRequest):
    papers = retrieve(req.question, state["embed_model"], state["model_key"], state["model_cfg"], k=state["config"]["llm"]["top_k"])
    answer = state["chain"].invoke({"context": format_context(papers), "question": req.question})
    return {"question": req.question, "answer": answer, "papers": papers}