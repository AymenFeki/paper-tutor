"""Copy the refreshed papers stored in a database into data/raw/recent/, so `make load` keeps them.

The weekly refresh runs in the cloud, where data/raw/ does not survive a restart; every refreshed
paper's raw record is kept in the refreshed_raw table instead. Run this against the cloud database:

    POSTGRES_HOST=<azure host> POSTGRES_PASSWORD=<azure password> POSTGRES_SSLMODE=require make pull-refreshed
"""

from collections import defaultdict

from paper_tutor.db import REFRESHED, connect
from paper_tutor.refresh import add_recent

with connect() as conn, conn.cursor() as cur:
    cur.execute(REFRESHED)
    rows = cur.fetchall()

by_day = defaultdict(list)
for refreshed_on, topic_id, paper in rows:
    by_day[refreshed_on.isoformat()].append({"topic_id": topic_id, "paper": paper})

added = sum(add_recent(entries, day)[0] for day, entries in by_day.items())
print(f"{len(rows)} refreshed papers in the database, {added} new in data/raw/recent/")
