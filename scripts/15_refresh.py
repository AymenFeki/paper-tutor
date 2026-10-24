"""Weekly refresh from the command line: add the best recent papers that match the syllabus (same as POST /refresh).

    uv run python scripts/15_refresh.py             # add them to data/raw/recent/ and the database
    uv run python scripts/15_refresh.py --dry-run   # only show what would be added
"""

import json
import sys
import time

from paper_tutor.corpus import load_config
from paper_tutor.embed import active_model, load_model
from paper_tutor.refresh import refresh

config = load_config()
_, model_cfg = active_model(config)
start = time.perf_counter()
result = refresh(config, load_model(model_cfg), model_cfg, dry_run="--dry-run" in sys.argv[1:])
print(json.dumps(result, indent=2))
print(f"{time.perf_counter() - start:.0f} s")
