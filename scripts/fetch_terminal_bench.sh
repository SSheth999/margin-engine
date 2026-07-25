#!/bin/bash
# Fetch Terminal-Bench 2.1 task definitions into tasks/terminal_bench/.
# Sparse-checkout of just tasks/ (the prebuilt Docker images live on a registry and
# are wrapped into E2B templates at run time, so we only need the task defs + tests here).
set -euo pipefail

REPO_URL="https://github.com/harbor-framework/terminal-bench-2-1.git"
DEST="$(cd "$(dirname "$0")/.." && pwd)/tasks/terminal_bench"

if [ -d "$DEST/.git" ]; then
  echo "updating existing clone at $DEST"
  git -C "$DEST" pull --ff-only
  exit 0
fi

echo "cloning $REPO_URL (sparse: tasks/) -> $DEST"
git clone --filter=blob:none --sparse "$REPO_URL" "$DEST"
git -C "$DEST" sparse-checkout set tasks
echo "done. discovered $(find "$DEST/tasks" -maxdepth 1 -mindepth 1 -type d | wc -l | tr -d ' ') task dirs."
