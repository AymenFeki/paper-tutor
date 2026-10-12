-- What share of papers has no DOI or no open-access URL?
SELECT
    COUNT(*) AS total,
    COUNT(*) FILTER (WHERE doi IS NULL) AS no_doi,
    COUNT(*) FILTER (WHERE oa_url IS NULL) AS no_oa_url,
    ROUND(COUNT(*) FILTER (WHERE doi IS NULL) * 100.0 / COUNT(*), 1) AS pct_no_doi,
    ROUND(COUNT(*) FILTER (WHERE oa_url IS NULL) * 100.0 / COUNT(*), 1) AS pct_no_oa_url 
FROM papers;


-- Which papers have no year?
SELECT id, title, venue
FROM papers
WHERE year IS NULL;