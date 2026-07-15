#!/usr/bin/env bash
# Syncs raw results off the GPU host before terminating an instance.
# Env-var configurable. Refuses to delete anything, local or remote — only
# ever adds files at the destination (rsync without --delete).
set -euo pipefail

: "${KVCOT_SYNC_DEST:?Set KVCOT_SYNC_DEST, e.g. export KVCOT_SYNC_DEST=user@host:/path/to/backup/}"
SRC="${KVCOT_SYNC_SRC:-results/}"

echo "== syncing $SRC -> $KVCOT_SYNC_DEST =="
echo "(no --delete: this script only ever adds files, never removes them)"

rsync -avz --progress "$SRC" "$KVCOT_SYNC_DEST"

echo "== sync complete =="
