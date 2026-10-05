"""Chat UI for the paper tutor. Talks to the FastAPI service over HTTP."""

import os
import uuid

import httpx
import streamlit as st

API_URL = os.getenv("PAPER_TUTOR_API", "http://127.0.0.1:8000")

st.set_page_config(page_title="paper-tutor", page_icon="📚")
st.title("paper-tutor")
st.caption("Answers from a curated database of research papers, with sources.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


def show_sources(papers):
    """Show the papers an answer was based on, inside a collapsible box."""
    with st.expander(f"Sources ({len(papers)})"):
        for i, paper in enumerate(papers, start=1):
            venue = paper.get("venue") or "unknown venue"
            url = f"https://openalex.org/{paper['id']}"
            st.markdown(f"**[{i}]** [{paper['title']}]({url}) ({paper['year']}, {venue})")
            if paper.get("authors"):
                st.caption(", ".join(paper["authors"]))


# Replay the conversation so far
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("query"):
            st.caption(f"Searched for: {msg['query']}")
        if msg.get("papers"):
            show_sources(msg["papers"])

question = st.chat_input("Ask about a topic you're studying")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching papers and writing an answer..."):
                response = httpx.post(
                    f"{API_URL}/chat",
                    json={"question": question, "thread_id": st.session_state.thread_id},
                    timeout=120,
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPError as error:
            st.error(f"The API could not be reached or returned an error: {error}")
            st.stop()

        st.markdown(data["answer"])
        st.caption(f"Searched for: {data['query']}")
        show_sources(data["papers"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": data["answer"],
            "query": data["query"],
            "papers": data["papers"],
        }
    )