"""HTTP API for the paper tutor: semantic search, cited answers, a tutor with memory and the weekly refresh."""

import os
import secrets
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from langsmith import configure, traceable
from pydantic import BaseModel

from paper_tutor.corpus import load_config
from paper_tutor.embed import active_model, load_model
from paper_tutor.quiz import make_quiz, syllabus_paper
from paper_tutor.rag import NO_PAPERS_MESSAGE, build_chain, build_retriever, format_context
from paper_tutor.refresh import refresh as refresh_papers
from paper_tutor.refresh_report import describe, failure_message, message, redact
from paper_tutor.tracing import trace_metadata
from paper_tutor.tutor import build_tutor

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the config, the models (embedding, reranker if on), LLM chain and tutor once, when the server starts."""
    config = load_config()
    state["config"] = config
    # metadata for the traces of /ask and /quiz; the retriever, chain and tutor attach it themselves
    configure(metadata=trace_metadata(config))
    state["model_key"], state["model_cfg"] = active_model(config)
    state["embed_model"] = load_model(state["model_cfg"])  # shared by the retriever and /refresh
    state["retrieve"] = build_retriever(config, state["embed_model"])
    state["chain"] = build_chain(config["llm"], trace_metadata(config))
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


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Security(api_key_header)):
    """Reject requests without the right key; if API_KEY is not set (local use), everything is allowed."""
    expected = os.getenv("API_KEY")
    if expected and not secrets.compare_digest(key or "", expected):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

@app.get("/health")
def health():
    """Check that the server is up and show which embedding model and retrieval settings are active."""
    return {"status": "ok", "embedding_model": state["model_key"], "retrieval": state["config"]["retrieval"]}


@app.post("/search", dependencies=[Depends(require_api_key)])
def search(req: SearchRequest):
    """Return up to k relevant papers, without generating an answer (none if nothing is relevant enough)."""
    papers = state["retrieve"](req.question, k=req.k)
    return {"question": req.question, "papers": papers}


@app.post("/ask", dependencies=[Depends(require_api_key)])
@traceable(run_type="chain", name="ask")  # one trace for the retrieval and the answer
def ask(req: AskRequest):
    """Answer a single question from retrieved papers, with no conversation memory."""
    papers = state["retrieve"](req.question, k=state["config"]["llm"]["top_k"])
    if not papers:
        return {"question": req.question, "answer": NO_PAPERS_MESSAGE, "papers": []}
    answer = state["chain"].invoke({"context": format_context(papers), "question": req.question})
    return {"question": req.question, "answer": answer, "papers": papers}


@app.post("/chat", dependencies=[Depends(require_api_key)])
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


@app.post("/quiz", dependencies=[Depends(require_api_key)])
@traceable(run_type="chain", name="quiz")  # one trace for the retrieval and all LLM attempts
def quiz():
    """A multiple-choice question about a paper for a random learn item from the syllabus."""
    topic, paper = syllabus_paper(state["config"], state["retrieve"])
    return {"topic": topic, **make_quiz(paper, state["config"]["llm"])}


# The refresh takes longer than Container Apps lets an HTTP request run (about 240 s, #37), so POST /refresh
# starts it in a background thread and GET /refresh/status reports it. The status lives in memory: this needs
# a single replica (maxReplicas 1), and a restart loses it (the workflow then sees 404 or another run_id).
refresh_lock = threading.Lock()  # one refresh at a time


def run_refresh(run):
    """Run the refresh for run, store its result or error in state["refresh_run"], then free the lock."""
    start = time.perf_counter()
    try:
        try:
            result = refresh_papers(state["config"], state["embed_model"], state["model_cfg"],
                                    dry_run=run["dry_run"])
            update = {"state": "done", "result": result, "message": message(result)}
        except Exception as error:  # noqa: BLE001 - every failure must reach the status
            print(redact(traceback.format_exc()), flush=True)
            update = {"state": "failed", "error": describe(error), "message": failure_message(error)}
        # a new dict, so a status request never sees a half-updated run
        state["refresh_run"] = {**run, **update, "finished_at": datetime.now(UTC).isoformat(),
                                "duration_s": round(time.perf_counter() - start)}
        print(f"refresh {run['run_id']}: {update['state']} in {state['refresh_run']['duration_s']} s", flush=True)
    finally:
        refresh_lock.release()  # only after the status is stored, so a new run can't be overwritten by this one


@app.post("/refresh", status_code=202, dependencies=[Depends(require_api_key)])
def refresh(dry_run: bool = False):
    """Start the weekly refresh in the background: add the best recently published papers that pass the rules
    and match the syllabus. With ?dry_run=true nothing is written. Answers 202 with the run (poll
    GET /refresh/status), or 409 with the current run if a refresh is already running."""
    if not refresh_lock.acquire(blocking=False):  # check and take in one step, so two requests can't both start
        return JSONResponse(status_code=409, content=state.get("refresh_run"))
    run = {"run_id": uuid.uuid4().hex, "state": "running", "dry_run": dry_run,
           "started_at": datetime.now(UTC).isoformat()}
    state["refresh_run"] = run
    threading.Thread(target=run_refresh, args=(run,), daemon=True).start()
    return run


@app.get("/refresh/status", dependencies=[Depends(require_api_key)])
def refresh_status():
    """The last refresh since the server started: state running, done (with result) or failed (with error), and
    the Telegram text in message. 404 if there was none, for example after a restart."""
    if "refresh_run" not in state:
        raise HTTPException(status_code=404, detail="No refresh since the server started")
    return state["refresh_run"]
