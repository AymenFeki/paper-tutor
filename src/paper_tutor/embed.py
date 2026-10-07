"""Embedding and reranking model helpers shared by all scripts."""


from pgvector.psycopg import register_vector
from sentence_transformers import CrossEncoder, SentenceTransformer

from paper_tutor.db import connect

FIND_MISSING = """
    SELECT p.id, p.title, p.abstract
    FROM papers p
    WHERE NOT EXISTS (
        SELECT 1 FROM embeddings e
        WHERE e.paper_id = p.id AND e.model = %s
    )
"""

INSERT = "INSERT INTO embeddings (paper_id, model, embedding) VALUES (%s, %s, %s)"


def active_model(config):
    """Return the short name and the settings of the model chosen in the config."""
    key = config["embeddings"]["active"]
    return key, config["embeddings"]["models"][key]


def load_model(model_cfg):
    model = SentenceTransformer(model_cfg["name"])
    if "max_seq_length" in model_cfg:
        model.max_seq_length = model_cfg["max_seq_length"]
    return model


def encode_query(model, model_cfg, text):
    """Encode a search question, adding the model's query prompt if it has one."""
    if "query_prompt" in model_cfg:
        return model.encode(text, prompt=model_cfg["query_prompt"], normalize_embeddings=True)
    if "query_prompt_name" in model_cfg:
        return model.encode(text, prompt_name=model_cfg["query_prompt_name"], normalize_embeddings=True)
    return model.encode(text, normalize_embeddings=True)


def load_reranker(name):
    """Cross-encoder that scores a (question, paper text) pair directly; slower but more precise."""
    return CrossEncoder(name)


def embedding_text(title, abstract):
    """The text that gets embedded for one paper: title and abstract."""
    parts = [part for part in (title, abstract) if part]
    return ". ".join(parts)


def embed_missing(config, model_key=None):
    """Embed every paper that has no embedding yet for the given model; returns how many were embedded."""
    model_key = model_key or config["embeddings"]["active"]
    model_cfg = config["embeddings"]["models"][model_key]
    with connect() as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(FIND_MISSING, (model_key,))
            rows = cur.fetchall()
            if not rows:
                return 0

            texts = [embedding_text(title, abstract) for _, title, abstract in rows]

            model = load_model(model_cfg)
            vectors = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True)

            params = [
                (paper_id, model_key, vector)
                for (paper_id, _, _), vector in zip(rows, vectors)
            ]
            cur.executemany(INSERT, params)
    return len(params)