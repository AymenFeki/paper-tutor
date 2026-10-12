# Features

[← Back to the README](../README.md)

Every behaviour below can be switched in `config/syllabus.yaml`, so old and new settings
can be compared. The defaults are the best settings measured (see
[eval/README.md](../eval/README.md)).

## Building the corpus

- **Credibility rules** (`settings` in the config, `corpus.py`): keep only articles,
  reviews, preprints and conference papers; English only; drop retracted papers; journal
  articles must come from a venue on the CWTS core list (preprints and conference papers
  are exempt); the abstract must have at least 30 words. Below 30 words the "abstracts"
  in OpenAlex are mostly citation strings ("Technometrics, Vol. 31, No. 2, pp. 270-271"),
  publisher boilerplate or a single word; this rule removed about 200 papers.
- **Trusted preprint servers** (`preprint_servers` in the config): a preprint must come from
  arXiv, RePEc, SSRN or HAL. Open repositories are full of auto-generated spam: 589 of
  1,716 recent candidates were Zenodo "preprints" such as "LAB #2710 ... LEDGER BENCH".
  The rule removed 74 papers from the corpus (26 Zenodo, 18 without a source, 13 bioRxiv,
  7 Research Square, 6 Preprints.org, 4 others). None of them was a test question's
  paper, and the retrieval metrics did not change (see [eval/README.md](../eval/README.md)).
- **Deduplication**: papers with the same normalised title published within 3 years of
  each other are one paper, e.g. an arXiv preprint and its conference version, or a
  conference paper and its later journal version. The published version is kept, then
  the one with a DOI, a venue and the longer abstract. 75 duplicates are merged, 31 of
  them across years.
- **LLMs and agents area** ([#9](https://github.com/AymenFeki/paper-tutor/issues/9)): built
  around five seed papers (RAG, ReAct, Toolformer, Chain-of-Thought, GPT-3) instead of
  OpenAlex topics. Candidates are the 200 most-cited papers citing each seed plus the
  papers the seeds cite (999 papers). They go through the same credibility rules (802
  pass; conference papers and preprints are exempt from the venue list, which matters
  because ML is published on arXiv and at conferences), must mention an LLM keyword in
  title or abstract (573; without this filter the most-cited candidates were LSTM, GloVe,
  word2vec and Vision Transformers, which cite or are cited by GPT-3), and must not have a
  corrupted arXiv record (see below). The 300 most-cited are kept; 294 are new to the
  corpus, the others were already in another area.
- **Corrupted OpenAlex records**: some records keep the DOI and citation count of a famous
  arXiv paper but show an unrelated title, abstract and references. The arXiv DOIs of RAG,
  ReAct and Chain-of-Thought point to records titled "Affordance-Compiled Intelligence",
  "Distributing Accountability, Not Capability" and "BNAI, NO-TOKEN, and MIND-UNITY".
  `02_fetch.py` downloads arXiv's own titles, and records whose title differs are dropped
  (`check_arxiv_titles`); references are only taken from seed records with the right title.
- **Authors**: the first five author names per paper, shown in the sources and given to
  the LLM, which may name authors only when they appear in the sources.

## Keeping it up to date

- **Weekly refresh** (`refresh.py`, `refresh` in the config): `POST /refresh` (or
  `make refresh`) fetches the papers published in the last 14 days for every topic,
  applies the credibility rules, removes duplicates, and scores each paper by its highest
  cosine similarity to any learn query of the syllabus. The top 20 papers of the window
  with a similarity of at least 0.75 are added if they are not in the database yet (same
  OpenAlex id or same normalised title). 0.75 comes from the measured scores: the best
  papers score about 0.82-0.85, spam and off-topic papers 0.55-0.71. Because the top 20 is
  chosen before known papers are skipped, running it again adds nothing, and the next
  week only adds what has entered the top 20. New papers are written to
  `data/raw/recent/<date>.json` (and their authors to `data/raw/authors.json`) before they
  are saved and embedded, so the raw files stay the source of truth and `make load` keeps
  them. The raw records are also stored in the `refreshed_raw` table, because in the cloud
  container `data/raw/` is lost on every restart; `make pull-refreshed` copies them from the
  cloud database into `data/raw/recent/` ([#29](https://github.com/AymenFeki/paper-tutor/issues/29)).
  The first run had 1,124 candidates and added 20 papers; it takes about 45 seconds.
- **Automations** ([#14](https://github.com/AymenFeki/paper-tutor/issues/14),
  [#29](https://github.com/AymenFeki/paper-tutor/issues/29)): two scheduled jobs call the API
  and send the result to a Telegram bot. In the cloud they are GitHub Actions workflows
  (`.github/workflows/quiz.yml`, `refresh.yml`) that call the Azure API with the API key, so
  they run without any local machine; each first calls `/health` with retries to wake the API
  from scale-to-zero. Locally the same jobs exist as n8n workflows (`n8n/`).
  - *Daily quiz* (every day at 10:00): `POST /quiz` picks a random learn item from the
    syllabus, retrieves a matching paper and asks the LLM for one multiple-choice question
    (Pydantic model with Telegram's length limits; questions that mention "the paper" or
    contain Chinese characters are rejected and regenerated). A second LLM call at
    temperature 0 then answers the question from the abstract without seeing the answer key;
    if it picks a different option, or finds no single correct one, the question is thrown
    away and a new one is written (at most 3 attempts,
    [#24](https://github.com/AymenFeki/paper-tutor/issues/24)). The options are shuffled and
    the question is sent as a Telegram quiz poll, followed by the topic and the source paper.
  - *Weekly refresh* (Sundays at 20:00): `POST /refresh`, then a Telegram message listing
    the papers added and their scores.

## Search and answers

- **Embeddings**: title and abstract embedded with `BAAI/bge-small-en-v1.5`, searched
  by cosine distance in pgvector. `Qwen/Qwen3-Embedding-0.6B` is also stored for
  comparison; the active model is set in the config.
- **Retrieval** (`rag.py`, `retrieval` in the config):
  - vector search (default),
  - optional hybrid search: Postgres full-text search on title and abstract (GIN index),
    fused with the vector ranking by Reciprocal Rank Fusion,
  - reranking of 30 candidates with the cross-encoder `BAAI/bge-reranker-base` (on by default),
  - a relevance threshold. Queries of at most 5 words ("Jeffreys prior") are judged by the
    reranker (score at least 0.5), longer questions by the cosine similarity (at least
    0.62). If no paper passes, nothing is returned and the tutor says the database has no
    relevant papers instead of answering from irrelevant ones.
- **LLM provider** (`llm.provider` in the config, `llm.py`): `ollama` runs qwen3:8b locally;
  `groq` calls the Groq API (`qwen/qwen3.8-27b`, free tier, needs `GROQ_API_KEY`), which
  answers in seconds instead of about 20 s and is needed for a cloud deployment without a
  GPU. Every chat model (answers, tutor, quiz) is built by one function, `build_llm`, so
  switching is one line in the config. The evaluation numbers were measured with qwen3:8b.
  With Groq, the quiz limits are enforced while generating (text is cut at the limit), so
  the prompt asks for shorter options and explanations than the hard limits.
- **LangGraph tutor** (`tutor.py`):
  - A first question that points at something it doesn't name ("When does it fail?") is
    answered with a clarifying question instead of a search (conditional edge, rule-based
    check in `is_vague`).
  - Follow-up questions such as "how does it compare to ridge?" are rewritten into a
    standalone search query using the conversation.
  - If retrieval returns no papers, the tutor refuses instead of answering (second
    conditional edge); otherwise it answers with the conversation in context.
  - Two answer prompts: `basic`, and `strict` (every sentence must be backed by a cited
    source, no outside knowledge). Memory is kept in process (`InMemorySaver`).

## API and clients

- **API endpoints**:
  - `GET /health`: status, active embedding model and retrieval settings.
  - `POST /search`: up to k relevant papers with authors, no LLM.
  - `POST /ask`: a single cited answer from the top papers (no memory).
  - `POST /chat`: the tutor with memory per `thread_id`.
  - `POST /quiz`: one multiple-choice question about a paper for a random syllabus item.
  - `POST /refresh`: the weekly refresh; returns the number of papers added, their titles
    and scores, and the number of candidates.
- **API key** (`api.py`): when the environment variable `API_KEY` is set, every endpoint
  except `/health` requires it in the `X-API-Key` header and answers `401` otherwise. The
  key is compared in constant time (`secrets.compare_digest`). Locally `API_KEY` is not
  set, so the API stays open for the UI, the MCP server and n8n.
- **MCP tool** (`mcp_server.py`): exposes `search_papers` so Claude Desktop can search
  the database and cite real papers, with authors.

## Quality control

- **Evaluation** (`scripts/07`, `08`, `11`, `12`, `14`): retrieval metrics with confidence
  intervals, out-of-scope questions and keyword queries, threshold calibration and a
  citation faithfulness check. Results: [eval/README.md](../eval/README.md).
- **Judging app** (`app/judge.py`, `make judge`): shows unjudged (question, paper) pairs
  one at a time, without saying which retrieval variant found them, and appends the
  answer to `eval/judgments.yaml` with `judge: human` ([#5](https://github.com/AymenFeki/paper-tutor/issues/5)).
- **Data checks** (`scripts/13_data_checks.py`, `make data-checks`): a report on reviews,
  title-abstract mismatches, area labels and corrupted arXiv records; nothing is deleted.
  Results: [data-checks.md](data-checks.md).
- **SQL analytics** (`sql/analytics/`): corpus statistics, data-quality checks, trends and
  query performance. Findings: [sql/analytics/README.md](../sql/analytics/README.md).