#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"

git config core.hooksPath "$ROOT_DIR/.githooks"
chmod +x "$ROOT_DIR/.githooks/pre-commit"

echo "Git hooks installed (core.hooksPath -> .githooks)."
