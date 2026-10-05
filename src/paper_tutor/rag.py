"""Retrieval-augmented answering: find papers, then let a local LLM answer from them."""

from collections import defaultdict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pgvector.psycopg import register_vector

from paper_tutor.db import connect
from paper_tutor.embed import active_model, encode_query, load_model, load_reranker

VECTOR_SEARCH = """
    SELECT paper_id
    FROM embeddings
    WHERE model = %(model)s
    ORDER BY embedding <=> %(q)s
    LIMIT %(n)s
"""

# plainto_tsquery joins the question's words with AND ('choos' & 'prior' & 'distribut'),
# which almost never matches a long question. Replacing & with | makes it an OR query,
# and ts_rank then puts the papers that match the most (and the rarest) words first.
TEXT_SEARCH = """
    WITH q AS (
        SELECT replace(plainto_tsquery('english', %(question)s)::text, ' & ', ' | ')::tsquery AS query
    )
    SELECT id
    FROM papers, q
    WHERE search_text @@ q.query
    ORDER BY ts_rank(search_text, q.query) DESC
    LIMIT %(n)s
"""

FETCH_PAPERS = """
    SELECT p.id, p.title, p.year, p.venue, p.authors, p.abstract, t.area,
           1 - (e.embedding <=> %(q)s) AS similarity
    FROM papers p
    JOIN embeddings e ON e.paper_id = p.id AND e.model = %(model)s
    LEFT JOIN topics t ON t.id = p.topic_id
    WHERE p.id = ANY(%(ids)s)
"""

SYSTEM = """You are a study tutor for a statistics and data science student.
Answer the question using ONLY the numbered sources below.
- Cite the sources you use with their numbers, like [1] or [2][4].
- If the sources do not answer the question, say so plainly instead of guessing.
- Do not add facts that are not in the sources, such as dates or results.
- Name authors only if they are listed in the sources.
- Explain clearly for a bachelor student, in at most two short paragraphs."""

NO_PAPERS_MESSAGE = (
    "I found no papers in the database that are relevant enough to answer this, so I won't guess. "
    "The database covers statistics, econometrics, machine learning, deep learning, economics, "
    "finance and cloud computing; try rephrasing with the name of a method or topic."
)

HUMAN = """Sources:
{context}

Question: {question}"""


def reciprocal_rank_fusion(rankings, k=60):
    """Combine ranked lists of ids: each list adds 1 / (k + rank) to an id's score. Best first."""
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, paper_id in enumerate(ranking, start=1):
            scores[paper_id] += 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


def document_text(paper):
    """The text a paper is compared on: title and abstract (the same text that was embedded)."""
    return ". ".join(part for part in (paper["title"], paper["abstract"]) if part)


def apply_threshold(papers, min_score):
    """Keep only the papers whose score is at least min_score."""
    return [p for p in papers if p["score"] >= min_score]


def build_retriever(config):
    """Load the models once and return a function question, k -> list of paper dictionaries.

    The steps are switched on and off in the `retrieval` section of the config:
    vector search, optionally fused with full-text search, optionally reranked by a
    cross-encoder, optionally cut off by a minimum score.
    """
    settings = config["retrieval"]
    model_key, model_cfg = active_model(config)
    embed_model = load_model(model_cfg)
    reranker = load_reranker(settings["reranker_model"]) if settings["rerank"] else None

    def retrieve(question, k=5):
        query_vector = encode_query(embed_model, model_cfg, question)
        n = max(k, settings["candidates"]) if settings["hybrid"] or reranker else k

        with connect() as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(VECTOR_SEARCH, {"q": query_vector, "model": model_key, "n": n})
                ids = [row[0] for row in cur.fetchall()]
                if settings["hybrid"]:
                    cur.execute(TEXT_SEARCH, {"question": question, "n": n})
                    text_ids = [row[0] for row in cur.fetchall()]
                    ids = reciprocal_rank_fusion([ids, text_ids], k=settings["rrf_k"])[:n]
                cur.execute(FETCH_PAPERS, {"q": query_vector, "model": model_key, "ids": ids})
                rows = cur.fetchall()

        found = {
            pid: {"id": pid, "title": title, "year": year, "venue": venue, "authors": authors or [],
                  "abstract": abstract, "area": area, "similarity": float(similarity)}
            for pid, title, year, venue, authors, abstract, area, similarity in rows
        }
        papers = [found[pid] for pid in ids if pid in found]  # keep the search order

        if reranker:
            scores = reranker.predict([(question, document_text(p)) for p in papers])
            for paper, score in zip(papers, scores):
                paper["score"] = float(score)
            papers.sort(key=lambda p: p["score"], reverse=True)
        else:
            for paper in papers:
                paper["score"] = paper["similarity"]

        if settings["threshold"]:
            min_score = settings["min_rerank_score"] if reranker else settings["min_similarity"]
            papers = apply_threshold(papers, min_score)
        return papers[:k]

    return retrieve


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
