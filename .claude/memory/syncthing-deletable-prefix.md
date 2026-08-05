---
name: syncthing-deletable-prefix
description: Syncthing (?d) 前缀用法——让被忽略文件在父目录删除时被一并删除
metadata:
  type: reference
---

## 问题场景

远端删除了一个目录，但本地该目录下含有被 `.stignore` 忽略的文件（如 `.venv`、`__pycache__`），Syncthing 报错：
```
syncing: delete dir: directory has been deleted on a remote device
but contains ignored files (see ignore documentation for (?d) prefix)
```

## 解决方案：`(?d)` 前缀

`(?d)` 标记被忽略文件为"可删除"——当父目录被远端删除时，允许 Syncthing 一并删除本地被忽略的文件。

### 正确用法

```gitignore
// 忽略 .venv 目录本身
.venv
// 标记 .venv 目录内容可删（尾斜杠很重要！）
(?d).venv/
```

两条规则配合：
- `.venv`（无斜杠）：忽略目录条目本身，不参与同步
- `(?d).venv/`（有斜杠）：匹配目录**内容**，标记为可删除

### 关键细节

1. **尾斜杠 `dir/` 匹配目录内容，不是目录本身**——无斜杠的 `.venv` 匹配目录条目，`(?d).venv/` 匹配其内容文件
2. **规则顺序敏感**：`(?d)` 规则必须放在 `.venv` 等宽泛规则**之前**，否则宽泛规则先匹配 → `(?d)` 永不生效
3. **`(?d)` 只对文件生效**，非空目录本身不会被 `(?d)` 删除——但文件被删后目录变空，Syncthing 可正常删除
4. **Syncthing 删除大量文件时有遍历边界**：3814 个文件的 site-packages 删到 85M 后停止，剩余部分需手动 `rm -rf` 补齐

### 验证是否生效

```bash
# 查看 .venv 大小是否在缩小
du -sh <dir>/.venv
# 查看 syncthing 日志
journalctl --user -u syncthing | grep -iE "delete|ignore"
```

**Why:** 2026-08-05 修复 AIS 同步错误时花了大量时间调试 `(?d)` 语法。关键误解：`(?d).venv` 单独一行对非空目录无效，必须 `.venv` + `(?d).venv/` 两行配合，且顺序必须 `(?d)` 在前。

**How to apply:** 见上面的代码块。`(?d)` 规则置于对应忽略规则之前，尾斜杠匹配目录内容。大量文件时 Syncthing 可能删不完，手动补齐。参考 [[syncthing-stignore-bidirectional]] 同步规则到对端。