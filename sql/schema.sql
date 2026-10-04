
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