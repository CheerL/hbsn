# 项目记忆（Project Memory）

本目录存放 HBSN 项目的**跨会话**记忆文件，由 Claude Code 自动读写。每条记忆一个文件，
遵循 frontmatter 规范：

```markdown
---
name: <short-kebab-case-slug>
description: <一句话摘要 — 用于判断是否相关>
metadata:
  type: project | user | feedback | reference
---

<事实内容；feedback/project 类附 **Why:** 与 **How to apply:**；
相关记忆用 [[slug]] 互链。>
```

## 已有记忆

- [syncthing-stignore-bidirectional](syncthing-stignore-bidirectional.md) — .stignore 不随 Syncthing 同步，两边必须手动镜像
- [syncthing-deletable-prefix](syncthing-deletable-prefix.md) — Syncthing (?d) 前缀用法（尾斜杠、顺序、遍历边界）
- [uv-extra-dev](uv-extra-dev.md) — uv run 默认不装 extras，pytest/ruff 需 uv run --extra dev
- [eval-coco-legacy-device](eval-coco-legacy-device.md) — 迁移 ckpt 存档 device 可能失效（cuda:2），评估需强制覆盖

## 约定

- **不记录**：代码结构、过去的修复、git 历史、CLAUDE.md 已承载的内容（仓库自己记录的
  东西不需要记忆）。
- **记录**：非显然的决策、踩坑、外部资源、跨会话约束。
- 索引在 `~/.claude/projects/-home-nnb-projects-HBSN-python/memory/MEMORY.md`
  （本项目的自动记忆索引，本目录是可提交到 git 的项目级补充）。

## 常用相关命令

```bash
uv run --extra dev python -m pytest   # 测试（--extra dev 装 pytest/ruff，见 uv-extra-dev）
uv run --extra dev ruff check .       # lint
uv run python -m hbsn.train ...       # 训练（主依赖即可）
```
