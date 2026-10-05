"""LangGraph tutor: rewrite follow-ups, retrieve papers, answer with memory."""

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from paper_tutor.rag import format_context, retrieve


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

ANSWER = ChatPromptTemplate.from_messages([
    ("system",
     ("You are a tutor for statistics and machine learning. Answer the user's last "
      "message using ONLY the numbered sources below and the conversation so far. "
      "Cite sources as [n]. If the sources do not answer it, say so. Do not add "
      "facts that are not in the sources, such as author names. Keep it to at most "
      "two paragraphs.\n\nSources:\n{context}")),
    MessagesPlaceholder("messages"),
])


def build_tutor(config, embed_model, model_key, model_cfg):
    llm_cfg = config["llm"]
    llm = ChatOllama(
        model=llm_cfg["model"],
        temperature=llm_cfg["temperature"],
        reasoning=llm_cfg["reasoning"],
    )

    def rewrite(state: TutorState):
        if len(state["messages"]) == 1:
            return {"query": state["messages"][-1].content}
        query = (REWRITE | llm | StrOutputParser()).invoke({"messages": state["messages"]})
        return {"query": query.strip()}

    def retrieve_papers(state: TutorState):
        papers = retrieve(state["query"], embed_model, model_key, model_cfg, k=llm_cfg["top_k"])
        return {"papers": papers}

    def answer(state: TutorState):
        context = format_context(state["papers"])
        response = (ANSWER | llm).invoke({
            "context": context,
            "messages": state["messages"],
        })
        return {"messages": [response]}

    graph = StateGraph(TutorState)
    graph.add_node("rewrite", rewrite)
    graph.add_node("retrieve", retrieve_papers)
    graph.add_node("answer", answer)
    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)

    return graph.compile(checkpointer=InMemorySaver())