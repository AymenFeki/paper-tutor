-- Q14: Top 3 papers by FWCI in each area --

WITH ranked AS (
    SELECT
        t.area,
        p.title,
        p.year,
        ROUND(p.fwci::numeric, 1) AS fwci,
        ROW_NUMBER() OVER (PARTITION BY t.area ORDER BY p.fwci DESC) AS rank_in_area
    FROM papers p
    JOIN topics t ON p.topic_id = t.id
    WHERE p.fwci IS NOT NULL and p.year <= 2023
)
SELECT *
FROM ranked
WHERE rank_in_area <= 3
ORDER BY area, rank_in_area;

-- Q15: Papers per area per year since 2015, with change vs. previous year --

WITH per_year AS (
    SELECT t.area, p.year, COUNT(*) AS n_papers
    FROM papers p
    JOIN topics t ON p.topic_id = t.id
    WHERE p.year >= 2015
    GROUP BY t.area, p.year
)
SELECT
    area,
    year,
    n_papers,
    LAG(n_papers) OVER (PARTITION BY area ORDER BY year) AS prev_year,
    n_papers - LAG(n_papers) OVER (PARTITION BY area ORDER BY year) AS change
FROM per_year
ORDER BY area, year;

-- Q16: Which 10 authors appear most often in the corpus? --

SELECT unnest(authors) AS author, COUNT(*) AS n_papers
FROM papers
GROUP BY author
ORDER BY n_papers DESC
LIMIT 10;