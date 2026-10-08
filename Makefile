# One-command setup for paper-tutor. Run `make rebuild` to build the database from scratch.
# Needs: uv, Docker, Ollama with qwen3:8b, and a .env file with OPENALEX_API_KEY and POSTGRES_PASSWORD.

.PHONY: db-up schema fetch load embed rebuild refresh pull-refreshed evaluate threshold keyword-queries faithfulness data-checks judge api ui test lint

# Start Postgres + pgvector in Docker and wait until it accepts connections
db-up:
	docker compose up -d
	until docker compose exec -T db pg_isready -h localhost -U tutor -d paper_tutor; do sleep 1; done

# Create the tables (safe to run again: every table uses CREATE TABLE IF NOT EXISTS)
schema:
	docker compose exec -T db psql -U tutor -d paper_tutor -v ON_ERROR_STOP=1 < sql/schema.sql

# Download papers (by topic and around seed papers), authors and arXiv titles into data/raw (cached data is skipped)
fetch:
	uv run python scripts/02_fetch.py

# Apply the credibility rules, remove duplicates and write papers to Postgres
load:
	uv run python scripts/04_load.py

# Embed the papers that do not have an embedding yet
embed:
	uv run python scripts/05_embed.py

# Everything above, in order
rebuild: db-up schema fetch load embed

# Add the best recently published papers that match the syllabus (same as POST /refresh, called weekly by GitHub Actions)
refresh:
	uv run python scripts/15_refresh.py

# Copy the papers that cloud refreshes stored in the database into data/raw/recent/ (needs the cloud POSTGRES_* variables)
pull-refreshed:
	uv run python scripts/16_pull_refreshed.py

# Retrieval evaluation with the settings in config/syllabus.yaml, then the pooled judgments
evaluate:
	uv run python scripts/07_evaluate.py
	uv run python scripts/08_pooled_evaluate.py

# Table for choosing the relevance threshold (needs a result saved with threshold=false)
threshold:
	uv run python scripts/07_evaluate.py threshold=false
	uv run python scripts/12_threshold.py bge-small

# Threshold rules on long questions and short keyword queries (#22)
keyword-queries:
	uv run python scripts/14_keyword_queries.py

# Share of the tutor's citations that the LLM judge finds supported (needs Ollama, takes minutes)
faithfulness:
	uv run python scripts/11_faithfulness.py

# Data-quality report: reviews, title-abstract mismatches, area mismatches, wrong arXiv titles (no changes)
data-checks:
	uv run python scripts/13_data_checks.py

# Judge unjudged (question, paper) pairs by hand; answers go to eval/judgments.yaml (#5)
judge:
	uv run streamlit run app/judge.py

# Start the FastAPI service on http://127.0.0.1:8000
api:
	uv run uvicorn paper_tutor.api:app

# Start the Streamlit chat UI (needs the API running)
ui:
	uv run streamlit run app/streamlit_app.py

# Unit tests (no database, Ollama or model download needed)
test:
	uv run pytest

# Lint with ruff
lint:
	uv run ruff check