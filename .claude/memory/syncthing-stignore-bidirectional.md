---
name: syncthing-stignore-bidirectional
description: .stignore 不随 Syncthing 同步，两边必须手动镜像修改
metadata:
  type: feedback
---

.stignore 文件自身被 `.stignore` 第一条规则排除（`.stignore` 条目），因此 **不会通过 Syncthing 同步到对端**。修改一端后，另一端必须手动同步修改。

**Why:** 2026-08-05 修复 HBSN 同步时，本机 .stignore 加了 `.mindfs/` 规则，但 NAS 侧对此不知，导致 `.mindfs/` 运行时 DB 被同步到 NAS。修复前已同步的文件残留需手动清理（ignore 规则生效后 Syncthing 会从远端删除已同步的残留——前提是两端 ignore 规则一致）。

**How to apply:**
- 修改 .stignore 后，ssh 到 NAS 容器或服务器，在对应路径做相同修改
- 修改后在本机重启 syncthing（`systemctl --user restart syncthing`）或触发 rescan，使新规则生效
- 对于 Docker 容器运行的 syncthing（如 NAS），`docker restart <container>` 或 `docker exec` 触发 rescan
- 已同步到对端的文件在 ignore 后会被自动删除（对端 global 移除后本地跟随删除）