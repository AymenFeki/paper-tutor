-- Q17: Full-text search: GIN index vs. no index vs. computing on the fly --

-- A) With the GIN index (normal case)
EXPLAIN ANALYZE
SELECT id, title FROM papers
WHERE search_text @@ plainto_tsquery('english', 'lasso regression');

-- B) Same query, index switched off for this session
SET enable_bitmapscan = off;
SET enable_indexscan = off;
EXPLAIN ANALYZE
SELECT id, title FROM papers
WHERE search_text @@ plainto_tsquery('english', 'lasso regression');
RESET enable_bitmapscan;
RESET enable_indexscan;

-- C) Without the stored column: build the text vector for every paper on the fly
EXPLAIN ANALYZE
SELECT id, title FROM papers
WHERE to_tsvector('english', coalesce(title, '') || ' ' || coalesce(abstract, ''))
      @@ plainto_tsquery('english', 'lasso regression');

-- Q18: Does an index on topic_id help? (tested inside a transaction, then undone) --
BEGIN;

EXPLAIN ANALYZE
SELECT COUNT(*) FROM papers WHERE topic_id = 'T10320';

CREATE INDEX papers_topic_id_idx ON papers (topic_id);
ANALYZE papers;

EXPLAIN ANALYZE
SELECT COUNT(*) FROM papers WHERE topic_id = 'T10320';

ROLLBACK;