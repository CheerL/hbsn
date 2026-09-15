# scripts/exp/ —— osc family 实验脚本阅读地图

「摆动滑移三角形族（osc family）」上 HBSN finetune 的完整工具链：
**定义族 → 建缓存 → 造 gt → 训练 → 按标准评估 → 出图**。

族：底边 `[-0.7, 0.7] × {0}`、高 `H_SLIDE`、顶点水平滑动 `x(psi) = X0 + AMP·sin(psi)`，
一个周期内古典 HBS 翻转 4 次（接缝在 `FLIPS` 里）。

## 执行顺序

```
osc_family.py ─────────────┬─→ build_osc_cache.py ──→ runs/cache_osc/（场 + 图像缓存）
                           │
classical_arcs.py ─────────┴─→ build_gt_arcs.py ─────→ gt（训练目标）

eval_osc_standard.py ──┬─→ ideal_gt_osc.py ────→ 理想连续 gt
                       ├─→ eval_osc_min.py / eval_osc_compare.py   （评估与汇总）
                       └─→ select_osc_ckpt.py                     （按标准选 ckpt）

finetune_osc.py ──→ runs/<run>/checkpoints/    （训练）

出图：e1_deform_osc_*.py
      e1_deform_osc_gt_compare.py 导出 _raw_slide / _nearest_valid，其余出图脚本依赖它
```

## 各脚本一句话

| 脚本 | 行数 | 做什么 | 读 | 写 |
|---|---:|---|---|---|
| `osc_family.py` | 79 | 族的**唯一定义**（几何、psi 采样、接缝） | — | — |
| `classical_arcs.py` | 89 | 古典 HBS 的接缝测量 + 逐段 180° 重标号（把不连续变成连续） | — | — |
| `build_osc_cache.py` | 111 | 整族 0..360°/0.25° 的**场 + 渲染图像缓存** | `osc_family` | `runs/cache_osc/*.npz` |
| `build_gt_arcs.py` | 114 | 用重标号后的古典场造**训练 gt** | `classical_arcs`/`eval_osc_standard` | `runs/cache_osc/gt_arcs_*.npz` |
| `ideal_gt_osc.py` | 383 | 造「理想连续 HBS」gt（另一种目标形态） | `eval_osc_standard` | `figures/hbsn/candidates/ideal_gt_osc*` |
| `eval_osc_standard.py` | 367 | **评估标准**：F(保真) + C(连续性) + O(参照系) | 缓存 npz | `runs/cache_osc/eval_standard_<tag>.npz` |
| `eval_osc_min.py` | 163 | 最小标准 F + C（去掉 O，见下） | 同上 | 同上 |
| `eval_osc_compare.py` | 79 | 跨候选汇总表 | `eval_standard_*.npz` | stdout |
| `select_osc_ckpt.py` | 133 | **按标准重选 checkpoint**（不是训练时的 `_quick_eval`） | run 目录 | stdout |
| `finetune_osc.py` | 808 | 主训练脚本（band loss、分组学习率、frozen-BN 等） | 缓存 npz | `runs/<run>/checkpoints/` |
| `train_osc_aligned.py` | ~130 | 对照训练（branch alignment） | `img/osc_merged` 等 | `runs/<run>/` |
| `e1_deform_osc.py` | 298 | 主面板图（半圆扫掠结构） | 缓存 npz | `figures/hbsn/candidates/*.png` |
| `e1_deform_osc_preview.py` | 500 | 预览图（`--ckpt`/`--out` 可指定） | 缓存 npz | 同上 |
| `e1_deform_osc_ft_compare.py` | 208 | base vs finetuned 对照 | 缓存 npz | 同上 |
| `e1_deform_osc_gt_compare.py` | 295 | 6 行对照：输入 / 古典 / 古典+翻转 / gt / base HBSN / ft | 缓存 npz | 同上 |
| `e1_deform_osc_arcs.py` | 252 | 3 行：输入 / 古典 / 古典+弧段翻转 | 缓存 npz | 同上 |
| `e1_deform_osc_frame_branch.py` | 213 | 展示 base ↔ igtA16 的 180° 分支差 | 缓存 npz | 同上 |
| `e1_deform_osc_final.py` | 163 | **最终交付图**（只有输入 + 模型输出两行） | 缓存 npz | 同上 |

## 最短复现路径

```bash
cd <repo 根>
python scripts/exp/build_osc_cache.py      # 1. 建缓存（下游全依赖它）
python scripts/exp/build_gt_arcs.py        # 2. 造训练 gt
python scripts/exp/finetune_osc.py --help  # 3. 训练（参数最多，先看 --help）
python scripts/exp/eval_osc_standard.py    # 4. 按 F+C+O 评估
python scripts/exp/select_osc_ckpt.py      # 5. 按标准挑 checkpoint
python scripts/exp/e1_deform_osc_final.py  # 6. 出终图
```

## 三条约定（改之前必读）

1. **路径一律相对仓库根推导**（`HERE`/`REPO` + `os.path.join`），不要再写 `/home/...` 绝对路径
   （2026-09-15 统一修掉 10 处；`.worktree/task-*/runs` 是指向主仓库 `runs/` 的软链，
   派生路径在 worktree 里同样成立）。
2. **评估标准是 F + C，不是训练时的 `_quick_eval`**：训练保存的 `best.pth` 按 11 个展示列
   的 mean F 选；标准是 616 个 psi + 连续性门限。要挑 ckpt 用 `select_osc_ckpt.py`，别信
   `best.pth` 的名字。
3. **族的定义只有一个**：`osc_family.py`。接缝中点（`FLIPS`，写在 `finetune_osc.py` /
   `build_gt_arcs.py` 里）是**对当前族的实测值**，改了几何就要重新测。

## 两条已定性的结论（省得后来人重推）

- **古典 HBS 的不连续只是"标签翻转"**：跨接缝 raw 场跳 ~0.30，但 `R_180` 之后只跳 ~0.009
  ——形状连续、分支标签翻转。这就是「理想连续 gt」的根据（`ideal_gt_osc.py`）。
- **F/C 看不见参照系错误**：F 会对每个 psi 取最优旋转、C 只看连续性，两者都对全局 180°
  旋转免疫，所以旧标准另加 O 门限；改用新目标后 O 可以去掉（见 `eval_osc_min.py`）。
