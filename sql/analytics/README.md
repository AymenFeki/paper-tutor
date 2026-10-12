# Corpus analytics (SQL)

[← Back to the README](../../README.md)

This folder holds plain SQL queries over the database, grouped by theme:

| File | Questions | What it answers |
|---|---|---|
| `01_corpus_overview.sql` | Q1–Q4 | Papers in total, per type and per year; top venues |
| `02_areas_and_topics.sql` | Q5–Q7 | Papers and FWCI per area; smallest topics |
| `03_data_quality.sql` | Q8–Q13 | Missing DOI, open-access link, year and authors; empty topics; missing embeddings |
| `04_trends.sql` | Q14–Q16 | Top papers per area; yearly growth; most frequent authors |
| `05_performance.sql` | Q17–Q18 | Query plans and timings with and without indexes (`EXPLAIN ANALYZE`) |

Run a file against the local database with:

```
docker compose exec -T db psql -U tutor -d paper_tutor < sql/analytics/01_corpus_overview.sql
```

## Findings

On the local database (7,186 papers, October 2026):

- **Composition**: 4,439 articles, 1,713 conference papers, 1,014 preprints and 20 reviews.
  arXiv is the largest venue (878 papers), followed by the Journal of Finance (190) and JASA (143).
- **Areas**: statistics 2,133, deep learning 1,701, finance 1,162, cloud infrastructure 699,
  machine learning 596, economics 419, agents 294, econometrics 182. Every topic has at least
  35 papers.
- **FWCI is right-skewed in every area** (deep learning: mean 69.2, median 24.5), so the median
  is the better summary. Agents has the highest median (38.7), because that area was built
  from the most-cited papers around five seed papers; statistics has the lowest (7.5).
- **FWCI is unreliable for recent papers.** Ranked by FWCI without a year limit, the top papers
  in machine learning and statistics were little-known 2024–2025 papers with FWCI above 1,000:
  for a new paper the expected citation count is tiny, so a few citations inflate the ratio.
  Limited to papers up to 2023, the top three are landmark papers (ImageNet, ResNet,
  Faster R-CNN in deep learning; lmerTest, Stan in statistics).
- **Some top-ranked records are mislabeled or not papers**, which a year limit does not fix:
  "Interactive Tree Of Life (iTOL)" (a biology tool) and "Lecture Notes in Computer Science
  1205" (a book-series record) rank in the top three of cloud infrastructure, and the top
  agents paper is about protein structure prediction.
- **Missing values**: 3.5% of papers have no DOI and 48.8% have no open-access link, so for
  about half of the corpus only the abstract is available. 4 papers have no year and 17 have
  no authors.
- **Embeddings**: all 7,186 papers have a `bge-small` embedding; 3 papers have no
  `qwen3-0.6b` embedding. All three were loaded on 2026-10-08, the first cloud refresh, which
  only embeds with the active model.
- **Growth**: the agents area jumps to 90 papers in 2023 and has none from 2026, because the
  weekly refresh does not cover that area.
- **Most frequent authors**: Michael I. Jordan (23), Donald B. Rubin (20), Mark E. J. Newman
  (19), Robert F. Engle (18). These are lower bounds, because only the first five authors of
  each paper are stored.
- **Performance** (`EXPLAIN ANALYZE`, local database): full-text search for "lasso
  regression" takes 0.09 ms with the GIN index on the stored `search_text` column, 8 ms
  without the index (sequential scan) and 493 ms when the text vector is computed on the
  fly, so the stored column and the index together make it more than 5,000× faster. An index
  on `topic_id` speeds up counting the papers of one topic from 1.5 ms to 0.03 ms (index-only
  scan), but it is not added, because no query in the app filters by topic.