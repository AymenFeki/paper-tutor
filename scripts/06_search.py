import sys

from pgvector.psycopg import register_vector

from paper_tutor.corpus import load_config
from paper_tutor.db import connect
from paper_tutor.embed import active_model, load_model, encode_query

SEARCH = """
    SELECT p.title, p.year, p.venue, t.area,
           1 - (e.embedding <=> %(q)s) AS similarity
    FROM embeddings e
    JOIN papers p ON p.id = e.paper_id
    JOIN topics t ON t.id = p.topic_id
    WHERE e.model = %(model)s
    ORDER BY e.embedding <=> %(q)s
    LIMIT 5
"""

question = " ".join(sys.argv[1:])
if not question:
    raise SystemExit('Usage: uv run python scripts/06_search.py "your question"')

config = load_config()
model_key, model_cfg = active_model(config)
model = load_model(model_cfg)
query_vector = encode_query(model, model_cfg, question)

with connect() as conn:
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute(SEARCH, {"q": query_vector, "model": model_key})
        results = cur.fetchall()

print(f"\nQuestion: {question}\n")
for title, year, venue, area, similarity in results:
    print(f"{title} ({year}) - {venue} - {area} (similarity: {similarity:.3f})")