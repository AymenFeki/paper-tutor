from pgvector.psycopg import register_vector

from paper_tutor.corpus import load_config
from paper_tutor.db import connect
from paper_tutor.embed import load_model

config = load_config()
model_key = config["embeddings"]["active"]
model_cfg = config["embeddings"]["models"][model_key]

FIND_MISSING = """
    SELECT p.id, p.title, p.abstract
    FROM papers p
    WHERE NOT EXISTS (
        SELECT 1 FROM embeddings e
        WHERE e.paper_id = p.id AND e.model = %s
    )
"""

INSERT = "INSERT INTO embeddings (paper_id, model, embedding) VALUES (%s, %s, %s)"


def document_text(title, abstract):
    """The text that gets embedded for one paper: title and abstract."""
    parts = [part for part in (title, abstract) if part]
    return ". ".join(parts)


with connect() as conn:
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute(FIND_MISSING, (model_key,))
        rows = cur.fetchall()
        print(f"{len(rows)} papers to embed with {model_key}")
        if not rows:
            raise SystemExit

        texts = [document_text(title, abstract) for _, title, abstract in rows]

        model = load_model(model_cfg)
        vectors = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=True)

        params = [
            (paper_id, model_key, vector)
            for (paper_id, _, _), vector in zip(rows, vectors)
        ]
        cur.executemany(INSERT, params)
        print(f"{len(params)} embeddings saved")

print("done")