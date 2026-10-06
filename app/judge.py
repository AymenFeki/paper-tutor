"""Judge relevance by hand (#5): one unjudged (question, paper) pair at a time.

Shows the top-5 papers of the reranker and hybrid result files that have no judgment yet,
without saying which variant found them or at which rank. Each answer is appended to
eval/judgments.yaml with `judge: human`. Skip only hides the pair until the page is reloaded.

Run with: uv run streamlit run app/judge.py   (needs the database running)
"""

import json
from pathlib import Path

import streamlit as st
import yaml

from paper_tutor.db import connect
from paper_tutor.evaluation import judgment_yaml, unjudged_pairs

JUDGMENTS = Path("eval/judgments.yaml")
RESULT_FILES = ["*+rerank*.json", "*+hybrid*.json"]  # in eval/results/


def load_pairs():
    paths = sorted({path for pattern in RESULT_FILES for path in Path("eval/results").glob(pattern)})
    results = [json.loads(path.read_text()) for path in paths]
    judgments = yaml.safe_load(JUDGMENTS.read_text())
    return unjudged_pairs(results, judgments, depth=5)


def load_paper(paper_id):
    """Year, venue, authors and abstract from the database."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT year, venue, authors, abstract FROM papers WHERE id = %s", (paper_id,))
        return cur.fetchone()


def save(pair, relevant):
    """Append one judgment to eval/judgments.yaml."""
    with open(JUDGMENTS, "a") as f:
        f.write(judgment_yaml(pair["question"], pair["paper_id"], relevant))


st.set_page_config(page_title="paper-tutor judging", page_icon="⚖️")
st.title("Relevance judgments")
st.caption("Relevant = a student asking this question would find the paper genuinely useful for answering it.")

if "skipped" not in st.session_state:
    st.session_state.skipped = set()

pairs = [p for p in load_pairs() if (p["question"], p["paper_id"]) not in st.session_state.skipped]
if not pairs:
    st.success("Nothing left to judge.")
    st.stop()

pair = pairs[0]
st.write(f"{len(pairs)} pairs left")

st.subheader("Question")
st.markdown(f"> {pair['question']}")

st.subheader("Paper")
row = load_paper(pair["paper_id"])
if row is None:
    st.warning(f"{pair['paper_id']} is no longer in the database; skip it.")
    year, venue, authors, abstract = None, None, [], None
else:
    year, venue, authors, abstract = row
url = f"https://openalex.org/{pair['paper_id']}"
st.markdown(f"**[{pair['title']}]({url})** ({year}, {venue or 'unknown venue'})")
if authors:
    st.caption(", ".join(authors))
st.write(abstract or "(no abstract)")

relevant, not_relevant, skip = st.columns(3)
if relevant.button("Relevant", type="primary", use_container_width=True):
    save(pair, True)
    st.rerun()
if not_relevant.button("Not relevant", use_container_width=True):
    save(pair, False)
    st.rerun()
if skip.button("Skip", use_container_width=True):
    st.session_state.skipped.add((pair["question"], pair["paper_id"]))
    st.rerun()
