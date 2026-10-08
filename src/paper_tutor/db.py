"""Database connection shared by all scripts."""

import os

import psycopg
from dotenv import load_dotenv
from psycopg.types.json import Jsonb

from paper_tutor.corpus import normalize_title, paper_row

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

PAPER_TITLES = "SELECT id, title FROM papers"

INSERT_REFRESHED = """
    INSERT INTO refreshed_raw (paper_id, topic_id, paper) VALUES (%s, %s, %s)
    ON CONFLICT (paper_id) DO NOTHING
"""

REFRESHED = "SELECT refreshed_on, topic_id, paper FROM refreshed_raw ORDER BY refreshed_on, paper_id"


def connect():
    """Connection settings come from the environment; the defaults are the local Docker database."""
    load_dotenv()
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "paper_tutor"),
        user=os.getenv("POSTGRES_USER", "tutor"),
        password=os.getenv("POSTGRES_PASSWORD"),
        sslmode=os.getenv("POSTGRES_SSLMODE", "prefer"),
    )


def save_papers(cur, topics, records, authors, settings):
    """Insert or update topics and papers; records are (raw OpenAlex paper, topic_id) pairs."""
    for topic_id, (name, area) in topics.items():
        cur.execute(UPSERT_TOPIC, (topic_id, name, area))
    for paper, topic_id in records:
        cur.execute(UPSERT_PAPER, paper_row(paper, topic_id, authors.get(paper["id"], []), settings))


def known_papers(cur):
    """Ids and normalised titles of every paper in the database, so a refresh can skip them."""
    cur.execute(PAPER_TITLES)
    rows = cur.fetchall()
    return {paper_id for paper_id, _ in rows}, {normalize_title(title) for _, title in rows} - {""}


def save_refreshed_raw(cur, records):
    """Keep the raw OpenAlex record of every refreshed paper in the database (data/raw/ is lost in the cloud)."""
    for paper, topic_id in records:
        cur.execute(INSERT_REFRESHED, (paper["id"], topic_id, Jsonb(paper)))
