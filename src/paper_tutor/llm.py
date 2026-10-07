"""The chat model used everywhere, built from the config: local Ollama or the Groq API."""

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

load_dotenv()


def build_llm(llm_cfg, temperature=None):
    """Chat model for llm_cfg["provider"]; temperature overrides the config value (e.g. 0.7 for the quiz)."""
    temperature = llm_cfg["temperature"] if temperature is None else temperature
    if llm_cfg["provider"] == "groq":
        return ChatGroq(
            model=llm_cfg["groq_model"],
            temperature=temperature,
            reasoning_effort="default" if llm_cfg["reasoning"] else "none",
        )
    return ChatOllama(model=llm_cfg["model"], temperature=temperature, reasoning=llm_cfg["reasoning"])