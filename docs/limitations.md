# Known limitations

[← Back to the README](../README.md)

## Evaluation and answers

- **Most relevance judgments come from an LLM.** 75 of 444 are human; a hand-checked
  sample agreed 19/20 with the LLM labels, but the set is still small (34 questions).
- **The thresholds were chosen on the queries they are tested on.** 30 new held-out
  queries (including 5–6 words) are also all answered or refused correctly. But the
  similarity margin for longer queries is small (0.03–0.05 from the threshold on both
  sets), and a 6-word query with only 2 matching papers in the database is still
  answered (see [eval/README.md](../eval/README.md)).
- **Citations from the 8B model are not always faithful**, even with the strict prompt,
  and the faithfulness check uses the same 8B model as judge
  ([#7](https://github.com/AymenFeki/paper-tutor/issues/7)).
- **The clarification check is a simple word rule.** It catches "When does it fail?" but
  not every vague question.
- **The quiz's answer-key check uses the same model as the writer.** It catches keys the model
  disagrees with when it answers the question fresh (before the check, a question marked "KL
  divergence is a squared Euclidean distance" as correct), but a misconception the model holds
  both times still passes.

## Data

- **The agents area is ranked by citations**, so it leans towards widely cited application
  papers (medicine, chemistry); the keyword filter still lets a few hardware papers in
  (e.g. TPU v4, whose abstract mentions language models).
- **Area labels are noisy** (see [data-checks.md](data-checks.md)); 20% of papers are closer
  to another area, mostly at natural borders ([#21](https://github.com/AymenFeki/paper-tutor/issues/21)).
- **Short-abstract junk above 30 words remains**, e.g. reference strings like "Claes
  Fornell, David F. Larcker, Structural Equation Models with ..., Journal of ...".
- **Only the first five authors are stored**, so the sources cannot say whether a paper
  has more authors.
- **The trusted-server rule only checks preprints.** A Zenodo record typed as a conference
  paper still passes (one such paper is in the corpus), because conference papers are
  exempt from the venue list.
- **17 papers have no authors and 4 have no year** in OpenAlex; some records that pass the
  rules are not papers (e.g. "Lecture Notes in Computer Science 1205").

## Weekly refresh

- **Few LLM and agent papers make the top 20.** The agents area has refresh topics
  since [#25](https://github.com/AymenFeki/paper-tutor/issues/25), but in the 14 days up to
  2026-10-24 only 1 of its 38 candidates at 0.75 or more made the top 20 (the next one
  missed it by 0.001); the other areas' papers score slightly higher.
- **The refresh takes one top 20 across all areas**, so busy areas win: the first run
  added 7 finance, 7 statistics, 3 deep learning, 2 economics and 1 machine learning
  paper, none for cloud infrastructure. A quota per area would balance it.
- **Syllabus similarity measures topic, not quality.** The weakest refreshed papers are
  applied studies from broad journals (an India VIX-RSI market-efficiency study in
  F1000Research, a buzzword-heavy causal-inference framework in *Electronics*); they pass
  because the journals are on the CWTS list.
- **Refreshed papers are only embedded with the active model** (bge-small). Run
  `uv run python scripts/05_embed.py qwen3-0.6b` before comparing against Qwen3 again.
- **The local database does not update by itself.** The weekly refresh writes to the cloud
  database; `make pull-refreshed`, `make load` and `make embed` bring the new papers to the
  local one.

## Cloud and operations

- **GitHub Actions schedules are best effort**: runs can start several minutes late, the
  times shift by an hour with daylight saving time (cron is in UTC), and GitHub disables
  scheduled workflows in a repository without activity for 60 days.
- **The cloud refresh must finish within Azure's request timeout** (about 4 minutes per
  HTTP request). The first cloud run had 1,022 candidates and added 3 papers well within
  it, but a week with many more candidates could get closer.
- **Cold start**: after a period without traffic, the first request to the cloud API
  takes about a minute (container start and model loading). That is the cost of scaling
  to zero.
- **Deploying a new version is half manual**: GitHub Actions builds and pushes the image,
  but the Container App is switched to the new `sha-<commit>` tag by hand.
- **Conversation memory is in process**: `/chat` threads are lost when the container
  restarts or scales to zero.