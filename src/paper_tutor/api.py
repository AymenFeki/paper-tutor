"""HTTP API for the paper tutor: semantic search, cited answers and a tutor with memory."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from paper_tutor.corpus import load_config
from paper_tutor.embed import active_model
from paper_tutor.quiz import make_quiz, syllabus_paper
from paper_tutor.rag import NO_PAPERS_MESSAGE, build_chain, build_retriever, format_context
from paper_tutor.tutor import build_tutor

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the config, the models (embedding, reranker if on), LLM chain and tutor once, when the server starts."""
    config = load_config()
    state["config"] = config
    state["model_key"], _ = active_model(config)
    state["retrieve"] = build_retriever(config)
    state["chain"] = build_chain(config["llm"])
    state["tutor"] = build_tutor(config, state["retrieve"])
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
    """Check that the server is up and show which embedding model and retrieval settings are active."""
    return {"status": "ok", "embedding_model": state["model_key"], "retrieval": state["config"]["retrieval"]}


@app.post("/search")
def search(req: SearchRequest):
    """Return up to k relevant papers, without generating an answer (none if nothing is relevant enough)."""
    papers = state["retrieve"](req.question, k=req.k)
    return {"question": req.question, "papers": papers}


@app.post("/ask")
def ask(req: AskRequest):
    """Answer a single question from retrieved papers, with no conversation memory."""
    papers = state["retrieve"](req.question, k=state["config"]["llm"]["top_k"])
    if not papers:
        return {"question": req.question, "answer": NO_PAPERS_MESSAGE, "papers": []}
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


@app.post("/quiz")
def quiz():
    """A multiple-choice question about a paper for a random learn item from the syllabus."""
    topic, paper = syllabus_paper(state["config"], state["retrieve"])
    return {"topic": topic, **make_quiz(paper, state["config"]["llm"])}