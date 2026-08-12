# HBSN 待办清单（2026-08-11 更新）

> 状态：E1 半圆扫掠族 / E3 去 no-RN 完成；E2 noise 已移除；E4 invariance 完成。

## 实验（E1–E4）

- [x] **E1 连续形变（半圆扫掠族）**：底边 A=(-0.7,0),B=(0.7,0) 固定、上顶点 C=(cos θ, sin θ)
      沿上半单位圆扫过，θ∈[110.9°,112.3°]，翻转 θ*≈111.10°（Im I₂ 穿越实轴）。
      经典在 θ* 处 R_π 翻转 1 次 / HBSN 0 次。
      **图语义**：a 图 y 轴 = d(B_θ, B_θ*)（每场与翻转点场 L2 距离）→ 经典呈**阶跃**
      （θ<θ* 侧 ≈0、θ>θ* 侧 ≈52）、HBSN 平坦（d_max≈3.4）；b 图 3 行×4 列快照
      （θ*-2ε/θ*-ε/θ*+ε/θ*+2ε，ε=6×10⁻⁴；行=黑底白三角形+C 顶点坐标 4 位小数/经典/HBSN），
      行标签竖排无 bbox，标题各自居中。产出 `figures/hbsn/f_deform.png`，回填 §5.1。
- [x] **E3 对称分层（去 no-RN + sym 修正）**：1296 对形状（椭圆/等边族三角形/base=0.7 三角形/
      扰动五边形）。**sym 关键修正**：几何 sym 的 n 集合去掉 n=2——理论 n≥3 旋转对称 →
      (1-e^{-2πi/n})I₂=0 → I₂=0 恰落实轴墙；n=2 椭圆不强制（e^{-2iθ}=1 前缀消失），
      若计入会把椭圆误归低 sym 箱破坏单调（实测排除后单调恢复）。
      经典翻率随 sym **单调降** 31%/23%/16%/8%/4%；HBSN 全程 0 翻转。
      **场对称残差不可用**：ops.sym 对完全对称形状噪声 0.13-0.72（网络/离散化），
      高阶插值无改善，故仍用几何极径代理。
      产出 `tab_flip_counts.tex` + `f_flip_strat_dual.png`（单面板 sym，纵轴 %，柱顶频率），
      回填 §5.2。
      **方法修正**：推理时置 post_stn=None 的 no-RN 是假消融（网络仍被 rotation loss 训练过），
      **已删除**；真·no-RN 需重训（见下方待办）。
- [x] **E4 不变性**：20 形状 × 平移/缩放/旋转轨道，中位±IQR 方差 ~2-4（场范数 5-9%），
      两法高度不变，HBSN 略优。产出 `tables/tab_invariance.tex`，回填 §5.2。
      **发现**：经典图像管线旋转时对个别近墙形状翻转（2/20），中位数稳健。
- [x] **E2 noise 完全移除**：paper.tex 删 `\subsubsection{Robustness to noise}` 整块；
      删 `e2_noise.py`、`f_noise.png`、`tab_noise.tex`（python + latex 两侧）；grep 无残留。
- [x] **回填**：paper.tex §5.1/5.2 四个 \todo 替换为图 + 表 + 结果文字。

## 待办

- [ ] **重训真·no-RN HBSN**：`net.stn_mode=1`（去 post_stn，stn_loss 自动归零；同数据/超参
      1000 epoch，RTX 4070 SUPER ~0.5–1 天）。E3 要对比"HBSN 无 RotationNormalizer"必须重训，
      推理期置 None 不算数（本网络被 rotation loss 训练过）。

## hbs fork 稳健化 — 已完成 ✅

`~/projects/hbs_python`（独立仓库）：
- [x] `hbs/hbs.py` 归一化 while True → 30 次上限（Σhbs≈0 振荡快速抛错）
- [x] `hbs/conformal_welding.py` y_post_norm → 200 次上限
- [x] `hbs/utils/zipper.py` 入口纯虚数微旋防护（MATLAB 版无检查，噪声边界含实部=0 点）
- [x] `bench/compare.py` 对拍保持（shape_1 的 corr 低是 MATLAB 侧饱和 0.9999 固有，非回归）
- [x] hbsn 侧 `_compat.py` 撤销 zipper patch（fork 已自愈），保留幂等 #1/#2

## 经典 HBS 管线（现状）

- [x] **渲染-提取绕行仍需保留**：fork zipper 对几何精确三角形仍不收敛（Beltrami 饱和），
      `classic.py` 的 渲染→像素边界→get_hbs 是必要 workaround。
- [x] **重建（HBS→形状）缺陷已定性**：恒等测试 3-6% 误差（尖角）为 **LSQC 分片线性近似的
      固有极限**——全局 4 倍加密误差不变、局部加密 Delaunay 崩、μ=0 重建精确、eps 正则无效。
      非网格/参数可调，需高阶元或非线性求解（另开大工程）。

## 实验脚本（scripts/exp/）

- `e1_deform.py` / `e3_flip_strat.py` / `e4_invariance.py`：E1/E3/E4 入口（`e2_noise.py` 已删）
- `hbsn_exp/`：classic（fork + 超时 + 重试）、grid（GHBS 网格）、nets（绝对路径 ckpt）、
  ops（R_π 翻转）、shapes（形状生成）、safe（SIGALRM 超时）、_compat（幂等 patch）
- 经典 HBS 基线全部走 hbs_python fork（`hbsn_exp/__init__.py` sys.path 优先，可用 HBS_PYTHON_DIR 覆盖）
