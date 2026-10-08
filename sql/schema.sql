-- Safe to run again: every statement uses IF NOT EXISTS

  CREATE EXTENSION IF NOT EXISTS vector;

  CREATE TABLE IF NOT EXISTS topics (
      id           TEXT PRIMARY KEY,
      name          TEXT,
      area         TEXT NOT NULL
  );

  -- One row per paper that passed the credibility rules
  CREATE TABLE IF NOT EXISTS papers (
      id            TEXT PRIMARY KEY,            -- OpenAlex ID
      doi           TEXT,
      title         TEXT,
      abstract      TEXT,                        -- plain text, rebuilt from the inverted index
      year          INTEGER,
      type          TEXT,
      language      TEXT,
      venue         TEXT,                        -- journal or conference name
      venue_lists   TEXT[],                      -- e.g. {cwts-core, doaj}
      is_retracted  BOOLEAN NOT NULL DEFAULT FALSE,
      fwci          DOUBLE PRECISION,
      oa_url        TEXT,
      topic_id      TEXT REFERENCES topics(id),
      loaded_at     TIMESTAMPTZ NOT NULL DEFAULT now()
  );

  -- One row per paper and embedding model
    CREATE TABLE IF NOT EXISTS embeddings (
    paper_id   TEXT NOT NULL REFERENCES papers(id),
    model      TEXT NOT NULL,
    embedding  vector NOT NULL,
    PRIMARY KEY (paper_id, model)
);

-- Added later, so they are ALTER statements instead of being part of CREATE TABLE above

-- First author names in author order (#23)
ALTER TABLE papers ADD COLUMN IF NOT EXISTS authors TEXT[];

-- Full-text search on title (weight A) and abstract (weight B), kept up to date by Postgres (#2)
ALTER TABLE papers ADD COLUMN IF NOT EXISTS search_text tsvector GENERATED ALWAYS AS (
    setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
    setweight(to_tsvector('english', coalesce(abstract, '')), 'B')
) STORED;
CREATE INDEX IF NOT EXISTS papers_search_text_idx ON papers USING GIN (search_text);

-- Raw OpenAlex record of every paper added by the weekly refresh (#29). In the cloud, data/raw/ is
-- lost when the container restarts, so the database keeps them; scripts/16_pull_refreshed.py
-- copies them to data/raw/recent/, where 04_load.py expects them.
CREATE TABLE IF NOT EXISTS refreshed_raw (
    paper_id      TEXT PRIMARY KEY,            -- full OpenAlex URL, as in the raw files
    topic_id      TEXT NOT NULL,
    paper         JSONB NOT NULL,
    refreshed_on  DATE NOT NULL DEFAULT CURRENT_DATE
);
