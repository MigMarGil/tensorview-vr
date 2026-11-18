#!/usr/bin/env bash
# Local test server for the WebXR build (localhost counts as a secure context, so WebXR works).
cd "$(dirname "$0")/docs" && echo "Open http://localhost:8080 in Chrome (production build of the IWSDK app) (with Meta's Immersive Web Emulator)" && python3 -m http.server 8080
