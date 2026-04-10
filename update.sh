#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_NAME="${REMOTE_NAME:-origin}"
BRANCH="${1:-main}"

cd "$REPO_ROOT"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "Not inside a git repository: $REPO_ROOT" >&2
  exit 1
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Refusing to update because the working tree has local changes." >&2
  git status --short
  exit 1
fi

echo "Fetching $REMOTE_NAME/$BRANCH"
git fetch --prune "$REMOTE_NAME"

LOCAL_SHA="$(git rev-parse HEAD)"
REMOTE_SHA="$(git rev-parse "${REMOTE_NAME}/${BRANCH}")"
BASE_SHA="$(git merge-base HEAD "${REMOTE_NAME}/${BRANCH}")"

if [[ "$LOCAL_SHA" == "$REMOTE_SHA" ]]; then
  echo "Already up to date on $BRANCH"
  exit 0
fi

if [[ "$LOCAL_SHA" != "$BASE_SHA" ]]; then
  echo "Local branch has commits that are not a fast-forward to ${REMOTE_NAME}/${BRANCH}." >&2
  echo "Review with: git log --oneline --decorate --graph HEAD..${REMOTE_NAME}/${BRANCH}" >&2
  exit 1
fi

echo "Fast-forwarding to ${REMOTE_NAME}/${BRANCH}"
git pull --ff-only "$REMOTE_NAME" "$BRANCH"
git submodule update --init --recursive
echo "Update complete"
