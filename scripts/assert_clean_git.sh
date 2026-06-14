#!/usr/bin/env bash
set -euo pipefail

if [[ -n "$(git status --porcelain)" ]]; then
  echo "ERROR: working tree is dirty. Commit or stash changes before running experiment."
  git status --short
  exit 1
fi

echo "Git commit: $(git rev-parse HEAD)"