"""Retrieval-augmented answering: find papers, then let a local LLM answer from them."""

from collections import defaultdict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable
from pgvector.psycopg import register_vector

from paper_tutor.db import connect
from paper_tutor.embed import active_model, encode_query, load_model, load_reranker
from paper_tutor.llm import build_llm
from paper_tutor.tracing import trace_metadata

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

SYSTEM_PROMPTS = {
    "basic": """You are a study tutor for a statistics and data science student.
Answer the question using ONLY the numbered sources below.
- Cite the sources you use with their numbers, like [1] or [2][4].
- If the sources do not answer the question, say so plainly instead of guessing.
- Do not add facts that are not in the sources, such as dates or results.
- Name authors only if they are listed in the sources.
- Explain clearly for a bachelor student, in at most two short paragraphs.""",
    "strict": """You are a study tutor for a statistics and data science student.
Answer the question using ONLY the numbered sources below.
- Every sentence must be supported by a source and end with its citation, like [1] or [2][4].
- Cite a source only for what its text actually says.
- Do not use outside knowledge: no facts, examples, numbers or dates that are not in the sources.
- Name authors only if they are listed in the sources.
- If the sources do not answer the question, say so plainly instead of guessing.
- Explain clearly for a bachelor student, in at most two short paragraphs.""",
}

NO_PAPERS_MESSAGE = (
    "I found no papers in the database that are relevant enough to answer this, so I won't guess. "
    "The database covers statistics, econometrics, machine learning, deep learning, LLMs and agents, "
    "economics, finance and cloud computing; try rephrasing with the name of a method or topic."
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


def as_documents(papers):
    """Papers as LangSmith shows retrieved documents: the text that was ranked, the rest as metadata.

    Copies the values, so later steps (reranking) don't change what an earlier run recorded.
    """
    return {"documents": [
        {"type": "Document", "page_content": document_text(p),
         "metadata": {key: value for key, value in p.items() if key != "abstract"}}
        for p in papers
    ]}


def is_short_query(question, max_words):
    """True for keyword-style queries such as "Jeffreys prior": at most max_words words."""
    return len(question.split()) <= max_words


def passes_threshold(paper, short_query, settings):
    """Is this paper relevant enough to keep?

    Short keyword queries are judged by the reranker score: their embedding similarity is about
    the same for on- and off-topic queries. Longer questions are judged by the similarity, which
    separates them better than the reranker score does (see README, Evaluation).
    """
    if short_query:
        return paper["rerank_score"] >= settings["min_rerank_score_short"]
    return paper["similarity"] >= settings["min_similarity"]


def build_retriever(config, embed_model=None):
    """Load the models once and return a function question, k -> list of paper dictionaries.

    The steps are switched on and off in the `retrieval` section of the config:
    vector search, optionally fused with full-text search, optionally reranked by a
    cross-encoder, optionally cut off by a minimum score. Pass embed_model to share an
    already loaded embedding model (it must be the active one) instead of loading it again.
    """
    settings = config["retrieval"]
    model_key, model_cfg = active_model(config)
    if embed_model is None:
        embed_model = load_model(model_cfg)
    short_query_threshold = settings["threshold"] and settings["short_query_words"] > 0
    reranker = load_reranker(settings["reranker_model"]) if settings["rerank"] or short_query_threshold else None

    @traceable(run_type="retriever", name="vector_search", process_outputs=as_documents)
    def search(question, n):
        """The n best papers by vector search (fused with full-text search if hybrid is on), best first."""
        query_vector = encode_query(embed_model, model_cfg, question)
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
        return [found[pid] for pid in ids if pid in found]  # keep the search order

    @traceable(run_type="retriever", name="rerank", process_outputs=as_documents)
    def add_rerank_scores(question, papers):
        """Score every paper against the question with the cross-encoder (adds "rerank_score")."""
        scores = reranker.predict([(question, document_text(p)) for p in papers])
        for paper, score in zip(papers, scores):
            paper["rerank_score"] = float(score)
        return papers

    @traceable(run_type="chain", name="paper_retriever", metadata=trace_metadata(config),
               process_outputs=as_documents)
    def retrieve(question, k=5):
        short_query = short_query_threshold and is_short_query(question, settings["short_query_words"])
        use_reranker = settings["rerank"] or short_query
        n = max(k, settings["candidates"]) if settings["hybrid"] or use_reranker else k

        papers = search(question, n)
        if use_reranker:
            papers = add_rerank_scores(question, papers)

        # score = what the papers are ranked by; without the reranker the search order stays
        for paper in papers:
            paper["score"] = paper["rerank_score"] if settings["rerank"] else paper["similarity"]
        if settings["rerank"]:
            papers.sort(key=lambda p: p["score"], reverse=True)

        if settings["threshold"]:
            papers = [p for p in papers if passes_threshold(p, short_query, settings)]
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


def build_chain(llm_cfg, metadata=None):
    """Prompt -> local LLM -> plain text. metadata (tracing.trace_metadata) is attached to its traces."""
    system = SYSTEM_PROMPTS[llm_cfg["prompt"]]
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", HUMAN)])
    llm = build_llm(llm_cfg)
    return (prompt | llm | StrOutputParser()).with_config(run_name="answer", metadata=metadata or {})
