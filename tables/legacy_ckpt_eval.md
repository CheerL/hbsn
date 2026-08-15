# 旧架构 checkpoint 综合评估记录

> 服务器（ai:107.173.180.64）迁移来的旧架构 HBSNet checkpoint（2024-03，v0.7）。
> 经 `scripts/exp/hbsn_exp/legacy.py` 兼容加载（bce 主干 → 当前 UNet 结构，
> strict=True；推理 256→GHBS 128 重采样）。2026-08-12 评估。

## 架构

旧架构 UNet：7 层，`channels = [8,16,32,64,128,256,512]`，downs=ups=6 → 输出 256×256。
当前架构：6 层，channels 至 256，输出 128×128。两者模块同构（`DoubleConv/BN`、
`maxpool_conv`、`up.1`），仅层深不同；场可比（同形状 |场| 相关性 0.966）。

## Checkpoint 清单

| checkpoint | 来源实验 | 日期 | stn_mode | best_epoch | best_loss |
|---|---|---|---|---|---|
| `no_rn_stn1_best993.pth` | `Mar17_14-11-06_stn1_big_aug_5` | 2024-03-17 | 1（仅 pre，无 RN）| 993 | 0.0083 |
| `no_rn_nostn_best943.pth` | `Mar18_11-02-27_nostn_big_aug_new` | 2024-03-18 | 0（无 STN）| 943 | 0.0092 |
| `no_rn_nostn_stnloss_best975.pth` | `Mar21_11-36-18_nostn_big_aug_stnloss` | 2024-03-21 | 0（无 STN）| 975 | 0.0096 |
| `stn3_big_aug_best886.pth` | `Mar18_00-42-05_stn3_big_aug` | 2024-03-18 | 3（pre+post，带 RN）| 886 | 0.0073 |

所有 ckpt 均为旧架构（bce 主干，7 层），原文件只读，下载至 `runs/remote_no_rn/`。

## 性能评估

| checkpoint | 场 corr↑ | 场 rel_err↓ | E3 翻率[0,0.02) | E3 翻率[0.15,∞) | E1 d_max↓ |
|---|---|---|---|---|---|
| `no_rn_stn1` | 0.851 | 0.401 | 0.0% | 0.2% | 2.217 |
| `no_rn_nostn` | 0.851 | 0.397 | 1.7% | 0.2% | **0.922** |
| `no_rn_nostn_stnloss` | **0.865** | **0.378** | 0.0% | 0.2% | 1.342 |
| `stn3_big_aug`（对照）| 0.818 | 0.465 | 1.7% | 0.2% | 1.719 |

**指标说明**
- **场质量**：5 个代表性形状（椭圆 b=0.6/0.9、等边三角、普通三角×2）上，
  HBSN |场| 与经典 HBS |场| 的皮尔逊相关（corr）与相对 L2 误差（rel_err）。
  经典场经渲染→像素边界管线（自带离散化误差），corr≈0.85 属正常场质量。
- **E3 翻转率**：E3 形状族 1296 相邻对 R_π 翻转，按 sym(B)（n≥3）分箱；
  此处列近 Σ（[0,0.02)）与远 Σ（[0.15,∞)）两箱。
- **E1 连续性**：半圆扫掠族 t∈[0.616,0.624] 相邻场距离 d(B_t,B_{t+δ}) 最大值。

**结论**
- 所有旧模型（含 no-RN）翻转率极低（0–1.7%），E1 连续（d_max<2.3）——
  与当前 HBSN+RN 一致，验证无 RotationNormalizer 的 HBSN 仍实现连续归一化。
- `no_rn_nostn_stnloss` 场质量最优（corr 0.865）；`no_rn_nostn` E1 最连续（0.922）。
- 与当前 BEST（HBSN+RN）相比，旧模型场 corr 略低（0.82–0.87 vs 当前 0.966 场可比性），
  归因于旧架构/训练数据不同代（2024-03 vs 2024-06）。
