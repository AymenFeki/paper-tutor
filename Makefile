# One-command setup for paper-tutor. Run `make rebuild` to build the database from scratch.
# Needs: uv, Docker, Ollama with qwen3:8b, and a .env file with OPENALEX_API_KEY and POSTGRES_PASSWORD.

.PHONY: db-up schema fetch load embed rebuild api ui test lint

# Start Postgres + pgvector in Docker and wait until it accepts connections
db-up:
	docker compose up -d
	until docker compose exec -T db pg_isready -h localhost -U tutor -d paper_tutor; do sleep 1; done

# Create the tables (safe to run again: every table uses CREATE TABLE IF NOT EXISTS)
schema:
	docker compose exec -T db psql -U tutor -d paper_tutor -v ON_ERROR_STOP=1 < sql/schema.sql

# Download papers from OpenAlex into data/raw (topics already downloaded are skipped)
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
