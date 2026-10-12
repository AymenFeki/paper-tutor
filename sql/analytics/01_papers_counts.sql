-- How many papers are there, and how many of each type?
SELECT COUNT(*) AS n_papers FROM papers;

SELECT type, COUNT(*) AS n_papers
FROM papers
GROUP BY type
ORDER BY COUNT(*) DESC;