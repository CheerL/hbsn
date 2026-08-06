---
name: ruff-no-file-lines-rule
description: ruff 无"文件最大行数"规则（C0302 invalid）——CI 兜底 ≤500 + tools/format.sh 一键重排
metadata:
  type: reference
---

ruff 没有"单文件最大行数"的内置 lint 规则：`ruff rule C0302`（pylint 的
too-many-lines）直接报 invalid rule，`[tool.ruff]` 配置也无法表达文件级行数上限。

**Why:** 2026-08-06 想把"长文件拆分"固化为自动门禁，发现 ruff 原生做不到。

**How to apply:**
- 文件行数门禁在 CI 兜底：`.github/workflows/ci.yml` 的 "File line-count cap (≤500)"
  步骤（heredoc python，扫 hbsn/scripts/tools/tests 四个根，超 500 退出 1）
- 一键全仓重排：`bash tools/format.sh`（ruff check --fix + ruff format + 双校验，
  末了 ruff check 0 error + format 全绿）
- CI 另有 `ruff format --check .` 步骤防 format 回归
- 当前最大文件 `scripts/eval_coco.py` 495 行（format 重排后 414→495），500 上限仅剩
  5 行余量，再增长就该拆分（如抽 compare 部分到 scripts/ 模块）

相关：[[loguru-capsys-stderr]]、[[uv-extra-dev]]
