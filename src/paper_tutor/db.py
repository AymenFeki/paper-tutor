"""Database connection shared by all scripts."""

import os

import psycopg
from dotenv import load_dotenv

from paper_tutor.corpus import paper_row

UPSERT_TOPIC = """
    INSERT INTO topics (id, name, area)
    VALUES (%s, %s, %s)
    ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name, area = EXCLUDED.area
"""

UPSERT_PAPER = """
    INSERT INTO papers (id, doi, title, abstract, year, type, language, venue,
                        venue_lists, is_retracted, fwci, oa_url, topic_id, authors)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (id) DO UPDATE SET
        title = EXCLUDED.title, abstract = EXCLUDED.abstract, fwci = EXCLUDED.fwci,
        venue_lists = EXCLUDED.venue_lists, oa_url = EXCLUDED.oa_url,
        authors = EXCLUDED.authors, loaded_at = now()
"""

def connect():
    load_dotenv()
    return psycopg.connect(
        host="localhost",
        port=5432,
        dbname="paper_tutor",
        user="tutor",
        password=os.getenv("POSTGRES_PASSWORD"),
    )


def save_papers(cur, topics, records, authors, settings):
    """Insert or update topics and papers; records are (raw OpenAlex paper, topic_id) pairs."""
    for topic_id, (name, area) in topics.items():
        cur.execute(UPSERT_TOPIC, (topic_id, name, area))
    for paper, topic_id in records:
        cur.execute(UPSERT_PAPER, paper_row(paper, topic_id, authors.get(paper["id"], []), settings))