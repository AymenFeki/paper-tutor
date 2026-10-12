-- Q5: How many papers per area? --

SELECT t.area, COUNT(*) AS n_papers
FROM papers p
JOIN topics t ON p.topic_id = t.id
GROUP BY t.area
ORDER BY n_papers DESC;

-- Q6: Per area: number of papers, average and median FWCI --
-- (FWCI is right-skewed, so the median is the better summary) --

SELECT
    t.area,
    COUNT(*) AS n_papers,
    ROUND(AVG(p.fwci)::numeric, 2) AS avg_fwci,
    ROUND(percentile_cont(0.5) WITHIN GROUP (ORDER BY p.fwci)::numeric, 2) AS median_fwci
FROM papers p
JOIN topics t ON p.topic_id = t.id
GROUP BY t.area
ORDER BY n_papers DESC;

-- Q7: Which 10 topics have the fewest papers? --

SELECT t.area, t.name, COUNT(*) AS n_papers
FROM papers p
JOIN topics t ON p.topic_id = t.id
GROUP BY t.area, t.name
ORDER BY n_papers ASC
LIMIT 10;