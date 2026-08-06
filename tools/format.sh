#!/usr/bin/env bash
# 一键全仓重排：ruff check --fix + ruff format，末了双校验保证彻底清 0。
# 用法：bash tools/format.sh（或在仓库根直接 ./tools/format.sh）
set -euo pipefail
cd "$(dirname "$0")/.."

uv run --extra dev ruff check --fix .
uv run --extra dev ruff format .
uv run --extra dev ruff format --check .
uv run --extra dev ruff check .

echo "✅ ruff check 0 error，format 全绿"
