#!/bin/bash
# Isolated pytest run for the social port: unique log path, explicit target.
source /opt/arena_ws/source >/dev/null 2>&1
ROOT=/opt/arena_ws/data/_fixwave/roots/social/arena_evaluation
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$ROOT" || exit 1
LOG=/tmp/pytest_${SLUG_LOG}.txt
python3 -m pytest "$TARGET" -q -p no:cacheprovider --tb=line > "$LOG" 2>&1
RC=$?
tail -120 "$LOG"
echo "PYTEST_EXIT=$RC"
