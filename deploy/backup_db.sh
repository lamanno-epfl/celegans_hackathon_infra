#!/usr/bin/env bash
# SQLite online backup via Python (no sqlite3 CLI required).
# Safe while the orchestrator is writing. Keeps the last 14 copies.
set -euo pipefail

SRC=/home/fusar/claude/celegans_hackathon_infra/runtime/orchestrator.db
DST_DIR=/home/fusar/claude/celegans_hackathon_infra/runtime/backups
PY=/home/fusar/claude/celegans_hackathon_infra/.venv/bin/python
mkdir -p "$DST_DIR"

STAMP=$(date +%Y%m%d-%H%M%S)
DST="$DST_DIR/orchestrator-$STAMP.db"

"$PY" - "$SRC" "$DST" <<'EOF'
import sqlite3, sys
src, dst = sys.argv[1], sys.argv[2]
with sqlite3.connect(src) as s, sqlite3.connect(dst) as d:
    s.backup(d)
EOF
gzip -f "$DST"

ls -1t "$DST_DIR"/orchestrator-*.db.gz 2>/dev/null | tail -n +15 | xargs -r rm --
echo "backup ok: $DST.gz"
