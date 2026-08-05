# HBSN 项目地图（Claude 速览）

研究项目：Harmonic Beltrami Signature Network（论文在 `../latex/`，独立 git 仓库）。
本目录 `python/` 是代码仓库根（原 `src`，2026-08 重写并改名）。

## 结构

```
pyproject.toml        hatchling；torch 2.13.0/tv 0.28.0/smp 0.3.3 精确钉版
hbsn/
  train.py            hydra 入口 + 自定义训练循环（不引入 lightning）
  registry.py         MODEL_REGISTRY：模型↔schema↔dataset 三元组
  config/schemas.py   dataclass schema（字段名与旧 Config 逐字一致）
  config/validate.py  未知 key 报错 + 跨字段断言（masked_size↔输出尺寸等）
  nets/               base/hbsn/stn/unet(双变体合一)/segmentation/deeplab/unetpp/maskrcnn/tpsn
  data/               dataset(BaseDataset+TransformSubset)/hbsn(.npy)/coco/transforms
  recorder.py         SummaryWriter+loguru+checkpoint 目录+结果图（三份 add_output 合一）
  conf/               hydra 配置（net/dataset/run/recorder 四组 + model 选择组）
  _legacy/pickle_shim.py  旧 ckpt pickle 反序列化桩（仅迁移工具用）
tools/                convert_mat_to_npy / migrate_checkpoints / 离线数据生成
scripts/eval_coco.py  test_full/test_script 合一评估 + --compare 多模型对比 + --single-image 单图推理
tests/                pytest 覆盖套件（含 test_registry/test_eval_coco/numerics 黄金测试）
.claude/              项目脚手架（settings.json 权限白名单 + memory/ 项目记忆）
```

## 关键约束（改动前必读）

1. **state_dict 键不可随意改**：`pre_stn.*/post_stn.*/backbone.*`（HBSNet）、
   `hbsn.*`（SegHBSNNet 子网，旧 ckpt 运行时语义为丢弃）、`model.*`
   （smp 0.3.3 / torchvision 0.28 键集，升级版本会静默换键）、
   `mask_conv.*/weight_layer.*/lap_loss.weight`。类名可改，属性名勿动。
2. **config 字段名与旧 Config 属性逐字一致**（pickle 桩反序列化的硬约束）。
3. **OmegaConf 合并后 dataclass @property 不可用**：派生值用
   `hbsn/nets/base.py` 的 `torch_dtype()`/`output_size()` 辅助函数
   （schemas.py 里不要再加 property，加了也不会被 OmegaConf 暴露）。
4. **浮点等价验证**：GPU 训练轨迹不可逐位复现；以 iteration-0 一致 +
   分布一致为准（见 README「已知行为说明」）。
5. **数据**：训练读 `.npy`（float32 CHW）；`.mat` 源已删除（远程服务器
   有备份），`.npy` 为唯一本地数据，均不入 git/Syncthing。
6. 本机仅 15GB 内存：`pin_memory` 默认 false（WSL2 CachingHostAllocator
   无界增长）；DataLoader worker 内存 ~1.2GB/个。
7. hydra 配置文件的完整默认值不可删（override 依赖 key 存在）。

## 开发流程（提交前门禁）

```bash
uv run --extra dev python -m pytest   # 必须全部 passed，覆盖 ≥80%（--extra dev 装 pytest/ruff）
uv run --extra dev ruff check .       # 必须 All checks passed（0 error）
```

- 测试已 hermetic：`tests/conftest.py` 合成数据，worktree/任意目录可跑；
  coco 数据依赖测试在无数据时自动 skip（test_coco_dataset_shapes）。
- ruff 已在 `[dev]` 依赖；ruff.toml 只设 `line-length = 80`，其余用默认规则集
  （含 PEP585/604 类型现代化 UP 系列——改类型标注时用 `list[int]`/`X | None`）。
- 提交用 Conventional Commits（fix/refactor/chore/docs/test）。

## 仓库状态

- `master` 当前 HEAD：重构审阅后（fix pickle_shim/tpsn + refactor + ruff 清零）。
- 论文证据：12 个旧 ckpt 已迁移到 `runs/migrated/`（原文件只读，tar 备份已按
  用户授权删除）。
- 数据：`img/**/*.npy` 16,799 个（唯一本地训练数据）；`coco/` 本机只有 val2017。

## 环境与陷阱

- **tokenless 已禁用**（`tokenless style off`）：该输出压缩插件有 reducer bug
  （破坏 git 参数、可静默吞命令）。若新会话它又启用，用 `tokenless style off` 关闭。
- **venv 在 `python/.venv`**（重建过，path 与新目录绑定）；旧 `.venv_old` 已删。
- Bash 工具 shell 是 **zsh**（变量不分词、`cd` 不持久跨调用）；重命名/移动文件后
  shebang 会断，需重建 venv。

## 与本仓库共存的其他目录（HBSN 根，git 外）

- `latex/`：论文源（独立 git 仓库；`paper_arxiv.tex` 永久冻结，勿改）。
- `outputs/`：hydra 运行输出（`python/outputs`，已被 .gitignore）。
- `img/`、`coco/`、`runs/`：数据与实验产物（git/Syncthing 均排除）。
