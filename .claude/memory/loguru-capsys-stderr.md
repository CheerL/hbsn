---
name: loguru-capsys-stderr
description: loguru 默认 sink 绑定 import 时原始 stderr，pytest capsys 捕获不到——测 loguru 输出用临时 sink
metadata:
  type: feedback
---

loguru 的默认 stderr sink 在 `import loguru` 时绑定当时的 `sys.stderr` 对象。pytest 的
`capsys` fixture 在测试内替换 `sys.stderr`，但 loguru 仍写向旧对象 → `capsys.readouterr()`
拿不到 loguru 输出（不报错，断言静默失效）。

**Why:** 2026-08-06 把 `scripts/eval_coco.py` 的 `print(..., file=sys.stderr)` 统一成
`logger.warning` 后，`test_build_configs_unknown_override_warns`（原 capsys 断言 stderr）
静默失效——测试不报错但断言永远不触发。

**How to apply:**
- 断言 loguru 输出不要用 `capsys`/`caplog`（caplog 走标准 logging，loguru 也不经过）
- 用临时 sink 拦截：
  `records=[]; id=logger.add(records.append, level="WARNING"); ...; logger.remove(id)`，
  再断言 `any("目标文本" in str(r) for r in records)`
- 项目内先例：`tests/test_eval_coco.py::test_build_configs_unknown_override_warns`

相关：[[ruff-no-file-lines-rule]]（同属 2026-08-06 工程收尾会话）
