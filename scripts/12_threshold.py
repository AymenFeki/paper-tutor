"""Show what different minimum scores would do, to choose the retrieval threshold.

Reads a result file saved by 07_evaluate.py with the threshold off, e.g.
    uv run python scripts/12_threshold.py bge-small
and prints, for a range of thresholds, how many in-scope questions would get no papers,
how many top-5 hits survive and how many out-of-scope questions would get no papers.
A good threshold refuses the out-of-scope questions without losing in-scope hits.
"""

import json
import sys
from pathlib import Path

from paper_tutor.evaluation import threshold_effect

name = sys.argv[1] if len(sys.argv) > 1 else "bge-small"
result = json.loads(Path(f"eval/results/{name}.json").read_text())
if result["retrieval"]["threshold"] or result["retrieval"]["rerank"]:
    raise SystemExit("Use a result saved with threshold=false and rerank=false: the threshold for longer "
                     "questions is on the similarity, and the scores must not be cut off yet.")

questions, out_of_scope = result["questions"], result["out_of_scope"]
hits5 = sum(q["rank"] is not None and q["rank"] <= 5 for q in questions)

# Candidate thresholds: every score seen, so no interesting value is skipped
scores = {q["retrieved"][0]["score"] for q in questions} | {q["top_score"] for q in out_of_scope}

print(f"\n{name}: threshold on the cosine similarity; {len(questions)} in-scope questions "
      f"({hits5} top-5 hits without a threshold), {len(out_of_scope)} out-of-scope questions\n")
print(f"{'min score':>10}{'in-scope refused':>18}{'top-5 hits kept':>17}{'out-of-scope refused':>22}")
for min_score in sorted(scores):
    e = threshold_effect(questions, out_of_scope, min_score)
    print(f"{min_score:>10.4f}{e['in_scope_refused']:>18}{e['hits5_kept']:>17}{e['out_of_scope_refused']:>22}")
