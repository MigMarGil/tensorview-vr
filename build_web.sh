#!/usr/bin/env bash
# Rebuilds the IWSDK WebXR app (web/) into docs/ (what GitHub Pages serves). Needs Node >= 20.19 / 22.12.
set -euo pipefail
cd "$(dirname "$0")/web"
npm install
npm run build
echo "Built -> docs/. Now run ./deploy.sh"
