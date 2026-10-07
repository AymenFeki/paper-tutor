# Evaluation

How well paper-tutor finds relevant papers, refuses off-topic questions and cites its sources. The data lives in this folder: test questions (`questions.yaml`, `out_of_scope.yaml`, `keyword_queries.yaml`), relevance judgments (`judgments.yaml`), retrieval results (`results/`) and faithfulness runs (`faithfulness/`). Back to the [main README](../README.md).

## Retrieval

34 test questions (`eval/questions.yaml`): 32 written for one specific paper each (4 per
area), plus 2 real-use failure cases about choosing priors. For each question the top 10
papers are retrieved and the rank of the relevant paper is recorded. Results are in
`eval/results/` (`uv run python scripts/07_evaluate.py`, add e.g. `hybrid=true` or
`embedding=qwen3-0.6b` to try a variant). All rows were measured on a corpus of 7,226
papers. The corpus now has 7,183 papers: the trusted-preprint rule removed 74, 11 agents
papers moved into the top 300, and the first weekly refresh added 20. The default variant
(bge-small + reranker + threshold) was re-run on it after each step, with exactly the same
numbers; none of the 20 refreshed papers appears in any top 10.

| Variant (n = 34) | hit@1 | hit@5 | hit@10 | MRR | hit@5 95% Wilson CI |
|---|---|---|---|---|---|
| bge-small, vector search | 0.32 | 0.47 | 0.65 | 0.41 | [0.31, 0.63] |
| bge-small + hybrid (full-text + RRF) | 0.21 | 0.47 | 0.56 | 0.30 | [0.31, 0.63] |
| **bge-small + reranker (default)** | **0.35** | **0.53** | **0.71** | **0.45** | **[0.37, 0.69]** |
| bge-small + hybrid + reranker | 0.29 | 0.56 | 0.65 | 0.41 | [0.39, 0.71] |
| Qwen3-Embedding-0.6B, vector search | 0.26 | 0.59 | 0.68 | 0.38 | [0.42, 0.74] |

The relevance threshold does not change these numbers: with it, no in-scope question
loses a hit. All the differences come from the 4 new agent questions:

| Variant | original 30 questions: hit@5 | MRR | 4 agent questions: hit@5 | MRR |
|---|---|---|---|---|
| bge-small, vector search | 0.50 | 0.44 | 1/4 | 0.17 |
| + hybrid | 0.40 | 0.28 | 4/4 | 0.47 |
| + reranker | 0.50 | 0.40 | 3/4 | 0.79 |
| + hybrid + reranker | 0.50 | 0.36 | 4/4 | 0.80 |
| Qwen3-Embedding-0.6B | 0.57 | 0.36 | 3/4 | 0.54 |

What this shows:

- On the original 30 questions nothing changed since the first baseline (hit@5 0.50,
  MRR 0.44): data cleanup and the new area did not disturb the old results.
- **Hybrid search and the reranker** look no better than vector search when only the one
  original paper counts, except on the 4 agent questions. Full-text search finds no
  relevant paper that vector search misses (a relevant paper is among the 30 candidates
  for 27 of 34 questions with or without it); it only changes the order. The decision
  was made with the pooled judgments below: the reranker is on, hybrid search is off
  ([#1](https://github.com/AymenFeki/paper-tutor/issues/1), [#2](https://github.com/AymenFeki/paper-tutor/issues/2), [#5](https://github.com/AymenFeki/paper-tutor/issues/5)).
- **Qwen3-Embedding-0.6B vs bge-small**, compared like with like (same corpus, same 34
  questions, no threshold) ([#6](https://github.com/AymenFeki/paper-tutor/issues/6)): Qwen3 finds the relevant paper in the top 5 more often
  (20 vs 16 of 34) but ranks it first less often (hit@1 0.26 vs 0.32, MRR 0.38 vs 0.41).
  bge-small stays active: the difference is within noise, the threshold is calibrated
  on bge-small similarities, and bge-small is about 18 times smaller (33M vs 0.6B
  parameters).
- n = 34, so differences of one or two questions are within noise (the intervals overlap).

## Retrieval with pooled judgments

Counting only one paper as relevant is strict, because other papers
can answer the same question. So the top 5 papers of every variant were pooled and judged
for relevance (`eval/judgments.yaml`: 444 judgments for 34 questions, 139 relevant;
`uv run python scripts/08_pooled_evaluate.py`). Unjudged papers count as not relevant,
so these are lower bounds:

| Variant | hit@5 (pooled) | 95% Wilson CI | MRR (lower bound) | unjudged papers in top 5 |
|---|---|---|---|---|
| bge-small, vector search | 25/34 | [0.57, 0.85] | 0.64 | 14 |
| + hybrid | 26/34 | [0.60, 0.88] | 0.50 | 0 |
| **+ reranker (default, with threshold)** | **30/34** | **[0.73, 0.95]** | **0.76** | **0** |
| + hybrid + reranker | 30/34 | [0.73, 0.95] | 0.72 | 0 |
| Qwen3-Embedding-0.6B | 29/34 | [0.70, 0.94] | 0.68 | 16 |

What this shows:

- **The reranker helps**: it finds a relevant paper in the top 5 for 30 of 34 questions
  (vector search: 25) and ranks it higher (MRR 0.76 vs 0.64). This is why it is on.
- **Hybrid search hurts on its own** (MRR 0.64 → 0.50): full-text search pushes up papers
  that share words with the question but answer something else. On top of the
  reranker it adds nothing (0.72 vs 0.76), so it stays off.
- The intervals still overlap: with 34 questions the reranker's gain is a clear signal,
  but the close variants cannot be ranked against each other.
- The bge-small and Qwen3 rows still have 14 and 16 unjudged top-5 papers (the 6 newer
  questions were never pooled for them), so their numbers may be a little too low.

How the judgments were made:

- **239 judgments by an LLM** (first pooling of bge-small and Qwen3): the judge saw
  titles and abstracts in shuffled order without model or rank labels, but had already
  seen the models' results earlier, so this judging was **not fully blind**.
- **205 judgments for the hybrid and reranker results**: 11 judged by hand in the judging
  app, the remaining 194 labelled by an LLM. 64 of these 194 were checked by hand (the 44
  borderline cases plus a random sample of 20; `judge: human` in the file): agreement was
  62/64 overall and **19/20 on the random sample**. In both disagreements the LLM had been
  too strict, so the pooled numbers lean low rather than high. The 130 unchecked labels
  are marked `judge: llm`.
- The test questions were also written by an LLM, while reading each abstract, so they
  may be easier than real student questions.

## Out-of-scope questions and keyword queries

10 out-of-scope questions (`eval/out_of_scope.yaml`: sourdough bread, football transfers,
the UI text "Searched for", ...) and 40 short keyword queries (`eval/keyword_queries.yaml`:
24 in scope like "Jeffreys prior", "GARCH", "retrieval augmented generation", 16 off topic
like "pasta recipe", "weather Munich") check whether the system answers or refuses
(`scripts/14_keyword_queries.py`). Best score among the 30 vector candidates:

| Queries | words | best cosine similarity | best reranker score |
|---|---|---|---|
| long, in scope (34) | 6–37 | 0.653 – 0.793 | 0.0009 – 0.9953 |
| long, off topic (10) | 2–10 | 0.467 – 0.660 | 0.0000 – 0.0907 |
| short, in scope (24) | 1–4 | 0.628 – 0.903 | 0.9822 – 0.9999 |
| short, off topic (16) | 2–4 | 0.449 – 0.660 | 0.0002 – 0.0907 |

The similarity separates long questions (except the 2-word "Searched for") but not short
queries; the reranker separates short queries by a wide margin but not long questions,
whose scores are spread from 0.0009 to 0.99. So the threshold uses the reranker for
queries of at most 5 words and the similarity for longer ones ([#22](https://github.com/AymenFeki/paper-tutor/issues/22)):

| Threshold rule | long in scope answered | long off topic refused | short in scope answered | short off topic refused |
|---|---|---|---|---|
| no threshold | 34/34 | 0/10 | 24/24 | 0/16 |
| similarity ≥ 0.62 for all queries (before) | 34/34 | 9/10 | 24/24 | 10/16 |
| reranker ≥ 0.5 for all queries | 6/34 | 10/10 | 24/24 | 16/16 |
| **≤ 5 words: reranker ≥ 0.5, longer: similarity ≥ 0.62 (default)** | **34/34** | **10/10** | **24/24** | **16/16** |

The similarity threshold 0.62 was chosen with `scripts/12_threshold.py`: every in-scope
question has a best similarity of at least 0.653 and every longer out-of-scope question at
most 0.577; 0.62 is in the middle of that gap. The thresholds were chosen on the same
queries they are reported on; the held-out queries below test them on new ones.

### Held-out queries

`eval/holdout_queries.yaml` has 30 new queries, written and labelled after the thresholds
were chosen: 10 in scope with 5–6 words (five of 5 words, judged by the reranker, and five
of 6 words, judged by the similarity), 10 longer in-scope questions, and 10 off topic
(five of 5–6 words, five longer). They were scored with the same rules as
`scripts/14_keyword_queries.py`.

| Threshold rule | in scope, 5–6 words: answered | in scope, long: answered | off topic: refused |
|---|---|---|---|
| similarity ≥ 0.62 for all queries (before) | 10/10 | 10/10 | 10/10 |
| reranker ≥ 0.5 for all queries | 9/10 | 5/10 | 10/10 |
| **≤ 5 words: reranker ≥ 0.5, longer: similarity ≥ 0.62 (default)** | **10/10** | **10/10** | **10/10** |

**The default rule answered or refused all 30 correctly.** For 5-word queries the
reranker margin is wide: in scope at least 0.972, off topic at most 0.103. For 6 words
and longer, the in-scope similarity is at least 0.661 and the off-topic similarity at most
0.570, both about 0.05 from the 0.62 threshold. This matches the calibration gap
(0.577 vs 0.653), so it is confirmed, not widened. Limits:

- **Small sample:** 10/10 per group still has a 95% Wilson interval of [0.72, 1.00].
- **The off-topic queries are easier than the existing ones:** even the old rule
  (similarity for all queries) refuses all 10. Their best similarity is at most 0.591,
  against up to 0.660 for bare nouns like "Harry Potter". So this set does not separate the
  old rule from the new one.
- **The 5/6-word boundary matters:** "serverless computing cold start latency problems"
  (6 words) is answered on similarity (0.702), but its best reranker score is 0.34, and its
  top papers are not about serverless cold starts. Only 2 papers in the database mention
  serverless. With `short_query_words: 6` it would have been refused, arguably the more
  useful answer. The threshold judges whether a query is on topic, not whether the corpus
  covers it well.

## Citation faithfulness

`scripts/11_faithfulness.py` asks the tutor 10 eval questions (every third one, the first
10), splits each answer into sentences, and for every citation asks the LLM whether the
cited paper's title and abstract support the sentence. All runs are in
`eval/faithfulness/`. The before runs (`basic.json`, `strict.json`) were measured on the
older corpus, before the agents area and the reranker were added. The script writes to
`eval/faithfulness/<prompt>.json`, so the newer runs were saved under the names below.

| Run | Answer prompt | retrieval | supported citations | 95% Wilson CI | sentences with a citation | file |
|---|---|---|---|---|---|---|
| before | basic | vector search, older corpus | 23/29 (0.79) | [0.62, 0.90] | 28/55 (0.51) | `basic.json` |
| before | strict | vector search, older corpus | 24/30 (0.80) | [0.63, 0.90] | 30/53 (0.57) | `strict.json` |
| reranker off | strict | vector search | 22/30 (0.73) | [0.56, 0.86] | 30/52 (0.58) | `strict_norerank.json` |
| | basic | + reranker | 25/28 (0.89) | [0.73, 0.96] | 28/56 (0.50) | `basic_rerank.json` |
| **default** | **strict** | **+ reranker** | **31/34 (0.91)** | **[0.77, 0.97]** | **34/49 (0.69)** | `strict_rerank.json` |
| repeat of the default | strict | + reranker | 32/35 (0.91) | [0.78, 0.97] | 35/47 (0.74) | `strict_rerank_repeat.json` |

The tutor and the judge run at temperature 0. **The reranker explains almost all of the
change:** with it off, 9 of 10 answers are identical to the before run; the tenth
retrieved different papers from the newer corpus (24/30 → 22/30). Generation is not
perfectly deterministic: a repeat of the default run changed one answer (31/34 → 32/35),
so expect about ±1 citation of noise per run, much less than the 22/30 → 31/34 gain from
the reranker. With the reranker, the strict prompt ("every sentence must be supported by
a cited source, no outside knowledge") cites in 69% of sentences against 50% for basic,
with about the same supported share (0.91 vs 0.89); before, the two prompts were
indistinguishable. The confidence intervals still overlap: 10 questions and about 30
citations are a signal, not proof.

**The 8B judge is too lenient.** A hand check of the 8 citations in two answers against
the cited abstracts agreed on all 3 in the medical-images answer, but in the data science
projects answer the judge accepted 4 of 5 while only 1–2 are supported: *MLOps as Enabler
of Trustworthy AI* is cited for specific roles and tools (MLOps engineers, CI/CD
pipelines, model registries) that its abstract does not mention. So on this sample the
judge overstates support by 2–3 of 8 citations, and the 0.91 is probably optimistic.
Other limits: qwen3:8b still writes uncited sentences, some with outside knowledge, which
the metric does not see, so fewer unsupported citations can just mean fewer citations;
the judge only sees abstracts, so a claim that is true in the paper body counts as
unsupported. The sample is the same 10 questions as before: `[::3][:10]` stops at
index 27, so the 4 agent questions and the two prior questions are not included.
