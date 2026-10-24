"""LangSmith tracing: load its settings from .env and describe the running app in every trace.

Tracing is switched on by environment variables (LANGSMITH_TRACING, LANGSMITH_API_KEY,
LANGSMITH_PROJECT); without them nothing is sent. LangChain and LangGraph calls are traced
automatically, the retrieval steps with @traceable in rag.py.
"""

import os
import subprocess
from functools import cache
from importlib.metadata import version
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # before the first traced call: langsmith caches the settings it reads


@cache
def git_sha():
    """Short commit hash: GIT_SHA (set when the Docker image is built), else from git, else "unknown"."""
    if os.getenv("GIT_SHA"):
        return os.getenv("GIT_SHA")[:7]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent, capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def trace_metadata(config):
    """What a trace was produced with, so traces can be filtered and compared in LangSmith."""
    llm_cfg = config["llm"]
    return {
        "app_version": version("paper-tutor"),
        "git_sha": git_sha(),
        "llm_provider": llm_cfg["provider"],
        "llm_model": llm_cfg["groq_model"] if llm_cfg["provider"] == "groq" else llm_cfg["model"],
        "embedding_model": config["embeddings"]["active"],
        "rerank": config["retrieval"]["rerank"],
        "hybrid": config["retrieval"]["hybrid"],
        "threshold": config["retrieval"]["threshold"],
    }
