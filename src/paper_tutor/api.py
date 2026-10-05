"""HTTP API for the paper tutor: semantic search, cited answers and a tutor with memory."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from paper_tutor.corpus import load_config
from paper_tutor.embed import active_model, load_model
from paper_tutor.rag import build_chain, format_context, retrieve
from paper_tutor.tutor import build_tutor

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the config, embedding model, LLM chain and tutor once, when the server starts."""
    config = load_config()
    model_key, model_cfg = active_model(config)
    state["config"] = config
    state["model_key"] = model_key
    state["model_cfg"] = model_cfg
    state["embed_model"] = load_model(model_cfg)
    state["chain"] = build_chain(config["llm"])
    state["tutor"] = build_tutor(config, state["embed_model"], model_key, model_cfg)
    yield
    state.clear()


app = FastAPI(title="paper-tutor", lifespan=lifespan)


class SearchRequest(BaseModel):
    question: str
    k: int = 5


class AskRequest(BaseModel):
    question: str


class ChatRequest(BaseModel):
    question: str
    thread_id: str


@app.get("/health")
def health():
    """Check that the server is up and show which embedding model is active."""
    return {"status": "ok", "embedding_model": state["model_key"]}


@app.post("/search")
def search(req: SearchRequest):
    """Return the k papers closest to the question, without generating an answer."""
    papers = retrieve(req.question, state["embed_model"], state["model_key"], state["model_cfg"], k=req.k)
    return {"question": req.question, "papers": papers}


@app.post("/ask")
def ask(req: AskRequest):
    """Answer a single question from retrieved papers, with no conversation memory."""
    papers = retrieve(
        req.question,
        state["embed_model"],
        state["model_key"],
        state["model_cfg"],
        k=state["config"]["llm"]["top_k"],
    )
    answer = state["chain"].invoke({"context": format_context(papers), "question": req.question})
    return {"question": req.question, "answer": answer, "papers": papers}


@app.post("/chat")
def chat(req: ChatRequest):
    """Answer a question in the context of a conversation thread, with memory per thread_id."""
    result = state["tutor"].invoke(
        {"messages": [("user", req.question)]},
        {"configurable": {"thread_id": req.thread_id}},
    )
    return {
        "question": req.question,
        "query": result["query"],
        "papers": result["papers"],
        "answer": result["messages"][-1].content,
    }