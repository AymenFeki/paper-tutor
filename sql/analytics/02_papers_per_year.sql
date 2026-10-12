-- How many papers per year?
SELECT year, COUNT(*) AS n_papers
FROM papers
GROUP BY year
ORDER BY year;