-- How many papers per area?
SELECT t.area, COUNT(*) AS n_papers
FROM papers p
JOIN topics t ON p.topic_id = t.id
GROUP BY t.area
ORDER BY n_papers DESC;