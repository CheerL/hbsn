---
name: eval-coco-legacy-device
description: 迁移 ckpt 存档 config 可能带失效 device（如 cuda:2），评估脚本需强制覆盖
metadata:
  type: feedback
---

旧 checkpoint 的 `config.net.device` 字段**可能指向不存在的 GPU**（远程训练时是
`cuda:2` 之类，本机没有）。`spec.net.factory(net_cfg)` 按存档 device 建模型会直接
`RuntimeError: CUDA error: invalid device ordinal`。

**Why:** 2026-08-06 给 eval_coco 加 `--compare` 多模型对比时，加载 deeplab 迁移 ckpt 崩溃——
其存档 config 是 `device: cuda:2`（笔记本在远程 GPU 跑过）。对比模式最初没强制覆盖
device，崩溃后才修复。

**How to apply:**
- eval_coco 的 `compare_evaluate` / `single_image_infer` 都要强制 `device`（默认 cpu，
  `--device` 可覆盖）——不能信 checkpoint 存档值
- `_load_compare_nets` 通过 `{"device": device}` 覆盖构建，正是为此
- 新写评估/加载脚本时，加载迁移 ckpt 前先想 device 字段是否可信

相关：[[uv-extra-dev]]（同属 2026-08-06 重构会话的开发陷阱）
