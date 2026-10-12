-- Q8: What share of papers has no DOI or no open-access URL? --

SELECT
    COUNT(*) AS total,
    COUNT(*) FILTER (WHERE doi IS NULL) AS no_doi,
    COUNT(*) FILTER (WHERE oa_url IS NULL) AS no_oa_url,
    ROUND(COUNT(*) FILTER (WHERE doi IS NULL) * 100.0 / COUNT(*), 1) AS pct_no_doi,
    ROUND(COUNT(*) FILTER (WHERE oa_url IS NULL) * 100.0 / COUNT(*), 1) AS pct_no_oa_url
FROM papers;

-- Q9: Which papers have no year? --

SELECT id, title, venue
FROM papers
WHERE year IS NULL;

-- Q10: Are there topics with 0 papers? --

SELECT t.id, t.name, COUNT(p.id) AS n_papers
FROM topics t
LEFT JOIN papers p ON p.topic_id = t.id
GROUP BY t.id, t.name
ORDER BY n_papers ASC
LIMIT 10;

-- Q11: Which papers have no embedding for the active model? --
-- (expected: bge-small 0, the active model; qwen3-0.6b only
-- covers papers from before the cloud refresh) --

SELECT
    m.model,
    COUNT(*) AS n_missing
FROM papers p
CROSS JOIN (SELECT DISTINCT model FROM embeddings) m
LEFT JOIN embeddings e
    ON e.paper_id = p.id AND e.model = m.model
WHERE e.paper_id IS NULL
GROUP BY m.model;

-- Q12: Which papers have no qwen3-0.6b embedding? --

SELECT p.id, p.title, p.loaded_at
FROM papers p
LEFT JOIN embeddings e
    ON e.paper_id = p.id AND e.model = 'qwen3-0.6b'
WHERE e.paper_id IS NULL;

-- Q13: How many papers have no authors stored? --

SELECT COUNT(*) AS n_no_authors
FROM papers
WHERE authors = '{}' OR authors IS NULL;