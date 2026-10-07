"""Embed the papers that do not have an embedding yet (optionally: scripts/05_embed.py qwen3-0.6b)."""

import sys

from paper_tutor.corpus import load_config
from paper_tutor.embed import embed_missing

model_key = sys.argv[1] if len(sys.argv) > 1 else None
print(embed_missing(load_config(), model_key), "embeddings saved")