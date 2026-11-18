#!/usr/bin/env bash
# Commits and pushes the project; then enable GitHub Pages (Settings -> Pages -> main /docs) or let gh do it.
set -euo pipefail
cd "$(dirname "$0")"
git add -A
git status --short | grep -E "keystore|\.apk|node_modules|venv/" && { echo "REVIEW: secrets/build files staged"; exit 1; } || true
git commit -m "TensorView VR: WebXR build (GitHub Pages) + native Quest build" || true
git branch -M main
git push -u origin main
if command -v gh >/dev/null 2>&1; then
  REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || true)
  if [ -n "$REPO" ]; then
    gh api -X POST "repos/$REPO/pages" -f "source[branch]=main" -f "source[path]=/docs" >/dev/null 2>&1 \
      && echo "Pages enabled." || echo "Pages already enabled (or enable it in Settings -> Pages: main / docs)."
    echo "Live URL: https://${REPO%%/*}.github.io/${REPO##*/}/"
  fi
else
  echo "Now: GitHub -> Settings -> Pages -> Deploy from branch -> main / docs"
fi
