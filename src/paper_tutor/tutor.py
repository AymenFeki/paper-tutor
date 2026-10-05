"""LangGraph tutor: rewrite follow-ups, retrieve papers, answer with memory or say no paper is relevant."""

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from paper_tutor.rag import NO_PAPERS_MESSAGE, format_context


class TutorState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    query: str
    papers: list[dict]


REWRITE = ChatPromptTemplate.from_messages([
    ("system",
     ("Rewrite the user's last message as a short standalone search query for a "
      "database of research papers. Use the conversation to resolve words like "
      "'it' or 'that method'. Return only the query, nothing else.")),
    MessagesPlaceholder("messages"),
])

ANSWER_PROMPTS = {
    "basic": (
        "You are a tutor for statistics and machine learning. Answer the user's last "
        "message using ONLY the numbered sources below and the conversation so far. "
        "Cite sources as [n]. If the sources do not answer it, say so. Do not add "
        "facts that are not in the sources. Name authors only if they are listed in "
        "the sources. Keep it to at most two paragraphs.\n\nSources:\n{context}"
    ),
    "strict": (
        "You are a tutor for statistics and machine learning. Answer the user's last "
        "message using ONLY the numbered sources below; use the conversation only to "
        "understand the question. Every sentence must be supported by a source and end "
        "with its citation, like [1] or [2][3]. Cite a source only for what its text "
        "actually says. Do not use outside knowledge: no facts, examples, numbers or "
        "names that are not in the sources. Name authors only if they are listed in the "
        "sources. If the sources do not answer the question, say so instead of guessing. "
        "Keep it to at most two paragraphs.\n\nSources:\n{context}"
    ),
}


def route_papers(state):
    """No paper passed the relevance threshold -> say so instead of answering."""
    return "answer" if state["papers"] else "no_papers"


def build_tutor(config, retrieve):
    """retrieve is the function returned by rag.build_retriever."""
    llm_cfg = config["llm"]
    llm = ChatOllama(
        model=llm_cfg["model"],
        temperature=llm_cfg["temperature"],
        reasoning=llm_cfg["reasoning"],
    )
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", ANSWER_PROMPTS[llm_cfg["prompt"]]),
        MessagesPlaceholder("messages"),
    ])

    def rewrite(state: TutorState):
        if len(state["messages"]) == 1:
            return {"query": state["messages"][-1].content}
        query = (REWRITE | llm | StrOutputParser()).invoke({"messages": state["messages"]})
        return {"query": query.strip()}

    def retrieve_papers(state: TutorState):
        return {"papers": retrieve(state["query"], k=llm_cfg["top_k"])}

    def no_papers(state: TutorState):
        return {"messages": [AIMessage(NO_PAPERS_MESSAGE)]}

    def answer(state: TutorState):
        context = format_context(state["papers"])
        response = (answer_prompt | llm).invoke({
            "context": context,
            "messages": state["messages"],
        })
        return {"messages": [response]}

    graph = StateGraph(TutorState)
    graph.add_node("rewrite", rewrite)
    graph.add_node("retrieve", retrieve_papers)
    graph.add_node("no_papers", no_papers)
    graph.add_node("answer", answer)
    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_conditional_edges("retrieve", route_papers)
    graph.add_edge("no_papers", END)
    graph.add_edge("answer", END)

    return graph.compile(checkpointer=InMemorySaver())
