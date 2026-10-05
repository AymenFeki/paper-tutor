"""Retrieval-augmented answering: find papers, then let a local LLM answer from them."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pgvector.psycopg import register_vector

from paper_tutor.db import connect
from paper_tutor.embed import encode_query

RETRIEVE = """
    SELECT p.id, p.title, p.year, p.venue, p.authors, p.abstract
    FROM embeddings e
    JOIN papers p ON p.id = e.paper_id
    WHERE e.model = %(model)s
    ORDER BY e.embedding <=> %(q)s
    LIMIT %(k)s
"""

SYSTEM = """You are a study tutor for a statistics and data science student.
Answer the question using ONLY the numbered sources below.
- Cite the sources you use with their numbers, like [1] or [2][4].
- If the sources do not answer the question, say so plainly instead of guessing.
- Do not add facts that are not in the sources, such as dates or results.
- Name authors only if they are listed in the sources.
- Explain clearly for a bachelor student, in at most two short paragraphs."""

HUMAN = """Sources:
{context}

Question: {question}"""


def retrieve(question, embed_model, model_key, model_cfg, k=5):
    """Return the k papers closest to the question, as dictionaries."""
    query_vector = encode_query(embed_model, model_cfg, question)
    with connect() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(RETRIEVE, {"q": query_vector, "model": model_key, "k": k})
            rows = cur.fetchall()
    return [
        {"id": pid, "title": title, "year": year, "venue": venue, "authors": authors or [], "abstract": abstract}
        for pid, title, year, venue, authors, abstract in rows
    ]


def format_context(papers):
    """Number the papers as sources: [1] Title (year, venue), the authors, then the abstract."""
    blocks = []
    for i, p in enumerate(papers, start=1):
        venue = p["venue"] or "unknown venue"
        header = f"[{i}] {p['title']} ({p['year']}, {venue})"
        if p.get("authors"):
            header += f"\nAuthors: {', '.join(p['authors'])}"
        blocks.append(f"{header}\n{p['abstract']}")
    return "\n\n".join(blocks)


def build_chain(llm_cfg):
    """Prompt -> local LLM -> plain text."""
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", HUMAN)])
    llm = ChatOllama(
        model=llm_cfg["model"],
        temperature=llm_cfg["temperature"],
        reasoning=llm_cfg["reasoning"],
    )
    return prompt | llm | StrOutputParser()