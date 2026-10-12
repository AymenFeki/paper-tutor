-- Q1: How many papers are there in total? --
SELECT COUNT(*) AS n_papers
FROM papers;

-- Q2: How many papers of each type? --
SELECT type, COUNT(*) AS n_papers
FROM papers
GROUP BY type
ORDER BY n_papers DESC;

-- Q3: How many papers per year? --

SELECT year, COUNT(*) AS n_papers
FROM papers
GROUP BY year
ORDER BY year;

-- Q4: Which 10 venues have the most papers? --

SELECT venue, COUNT(*) AS n_papers
FROM papers
WHERE venue IS NOT NULL
GROUP BY venue
ORDER BY n_papers DESC
LIMIT 10;