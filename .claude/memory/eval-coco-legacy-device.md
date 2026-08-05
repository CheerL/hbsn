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
device，崩溃后才修复。**同一天单模型模式（`--model + --checkpoint`）又踩一次**：
`evaluate()` 在 `args.device` 为 None 时保留 ckpt 存档 device，跑 deeplab ckpt 同样崩，
需 `device=args.device or "cpu"` 强制覆盖。

**How to apply:**
- eval_coco 的 `evaluate` / `compare_evaluate` / `single_image_infer` 三个入口都要强制
  `device`（默认 cpu，`--device` 可覆盖）——不能信 checkpoint 存档值
- `_build_configs` 对 `None` 值跳过覆盖，所以"不传 device" = "用存档值" = 隐患
- 迁移 deeplab/tpsn 等远程训练 ckpt 时默认 cpu，除非明确本机 GPU 可用

相关：[[uv-extra-dev]]（同属 2026-08-06 重构会话的开发陷阱）
