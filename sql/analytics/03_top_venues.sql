-- Which 10 venues have the most papers?
SELECT venue, COUNT(*) AS n_papers
FROM papers
WHERE venue IS NOT NULL
GROUP BY venue
ORDER BY n_papers DESC
LIMIT 10;