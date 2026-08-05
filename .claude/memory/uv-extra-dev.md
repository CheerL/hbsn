---
name: uv-extra-dev
description: uv run 默认不装 [project.optional-dependencies] extras，pytest/ruff 需 uv run --extra dev
metadata:
  type: feedback
---

`uv run` 默认**只安装主依赖**，不会装 `[project.optional-dependencies]` 里的 extras。
本仓库 pytest/ruff 在 `dev` extras 里，直接 `uv run python -m pytest` / `uv run ruff check .`
会报 `No module named pytest` / `Failed to spawn: ruff`。

**Why:** 2026-08-06 在 worktree 里首次跑 `uv run ruff check .` 失败（uv 只同步了主依赖，
先装完 torch 等 73 个包后找不到 ruff）；`uv run --extra dev` 才装上 pytest/ruff。

**How to apply:**
- 测试：`uv run --extra dev python -m pytest`
- lint：`uv run --extra dev ruff check .`
- `--extra dev` 会补齐 dev extras 后复用已装的缓存，增量安装很快（本会话实测 ~55ms 补 ruff）
- 任何把命令写进 CLAUDE.md/README/skill 的地方都要带 `--extra dev`

相关：[[eval-coco-legacy-device]]（同属 2026-08-06 重构会话的开发陷阱）
