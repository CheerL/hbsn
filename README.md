# HBSN — Harmonic Beltrami Signature Network

HBSN 是配论文的 PyTorch 研究代码库：**调和 Beltrami 签名网络**（Harmonic
Beltrami Signature Network），用于从带噪声的二维形变观测中学习**旋转不变的表征**
与**形变场**。核心思想：用 UNet backbone + 空间变换网络（STN，旋转归一化）+
圆盘掩码提取每个点的 Beltrami 签名，分割模型（DeepLab/UNet++/MaskRCNN/TPSN）
内嵌 HBSN 子网作为正则/辅助头。

> 论文源码在 `../latex/`（独立 git 仓库）。本文档只描述 `python/` 代码仓库
> （git 仓库根，原 `src`，2026-08 激进现代化重写：fire→hydra、双 UNet 合并、
> 死代码删除、checkpoint 迁移）。

## 特性

- 5 种模型统一注册：`hbsn` | `deeplab` | `unetpp` | `maskrcnn` | `tpsn`
- hydra/omegaconf 配置：未知 key 报错（不再静默忽略）+ 跨字段约束校验
- 数据管线：.mat → float32 `.npy`（CHW），有界缓存 + 多 worker 透传
- 旧 checkpoint 迁移工具（pickle 桩 + state_dict 键映射），论文指标可复现
- 类型标注全覆盖，ruff clean，15 个 pytest

## 目录结构

```
python/                       # git 仓库根
├── pyproject.toml            # hatchling；精确钉版 torch/tv/smp
├── ruff.toml                 # line-length = 80，默认规则集
├── hbsn/                     # Python 包
│   ├── train.py              # hydra 入口 + 自定义训练循环
│   ├── registry.py           # MODEL_REGISTRY：模型↔schema↔dataset 三元组
│   ├── config/               # dataclass schema + 跨字段校验
│   ├── nets/                 # base/hbsn/stn/unet/segmentation/deeplab/unetpp/maskrcnn/tpsn
│   ├── data/                 # dataset/hbsn(.npy)/coco/transforms
│   ├── recorder.py           # SummaryWriter + loguru + checkpoint + 结果图
│   ├── conf/                 # hydra 配置（net/dataset/run/recorder + model 选择组）
│   ├── geometry/             # geodesic welding（离线数据生成用）
│   └── _legacy/pickle_shim.py # 旧 ckpt 反序列化桩（仅迁移工具用）
├── tools/                    # convert_mat_to_npy / migrate_checkpoints / 离线数据生成
├── scripts/eval_coco.py      # test_full+test_script 合一评估
├── tests/                    # 15 个 pytest
└── .claude/                  # Claude Code 脚手架（settings + memory）
```

数据与实验产物（git/Syncthing 均排除）：`img/`（.npy 训练数据）、`coco/`
（COCO 分割数据）、`runs/`（checkpoint 与 tensorboard 输出）。

## 快速开始

```bash
# 1. 环境（Python 3.10 + uv）
uv venv --python 3.10 && uv pip install -e ".[dev]"

# 2. 数据准备（若 img/**/*.npy 不存在）
python tools/convert_mat_to_npy.py --root img   # 从 .mat 转 .npy（幂等、断点续转）

# 3. 训练 hbsn（1 epoch 冒烟）
python -m hbsn.train model=hbsn run.total_epoches=1 net.device=cuda:0

# 4. 测试 + lint
python -m pytest && uv run ruff check .
```

`model` 可选：`hbsn` | `deeplab` | `unetpp` | `maskrcnn` | `tpsn`。
`model` 决定 net/dataset 配对（hbsn↔合成 HBS 数据，分割模型↔COCO）。

## 配置（hydra）

- `hbsn/conf/config.yaml`：顶层（model 选择组 → net/dataset 配对）
- `hbsn/conf/net/{model}.yaml`：网络参数（seg 类共享 `seg.yaml` 基座）
- `hbsn/conf/dataset/{model}.yaml`：数据集参数（COCO 类共享 `coco.yaml`）
- `hbsn/conf/run/default.yaml`：训练循环参数（epochs/lr/…）
- `hbsn/conf/recorder/default.yaml`：tensorboard/loguru 参数

字段类型与默认值在 `hbsn/config/schemas.py`（dataclass）定义；**未知 key 在
compose 时报错**（不再静默忽略）；跨字段约束（stn_mode 合法值、channels 层数
关系、`masked_size` 与输出尺寸匹配、数据路径存在）在 `hbsn/config/validate.py`。

```bash
# tpsn 完整示例
python -m hbsn.train model=tpsn net.stn_mode=3 \
  net.hbsn_checkpoint=runs/migrated/hbsn/Jun11_13-01-53_big_ns/checkpoints/best.pth \
  dataset.cat_ids='[16]' dataset.connected=true dataset.single_instance=true \
  run.batch_size=16 run.lr=5e-5 run.total_epoches=1000
```

### 旧 CLI（fire）→ 新 CLI（hydra）对照

| 旧（fire） | 新（hydra） |
|---|---|
| `--type tpsn` | `model=tpsn` |
| `--device cuda:0` | `net.device=cuda:0` |
| `--total_epoches 1000` | `run.total_epoches=1000` |
| `--batch_size 64` | `run.batch_size=64` |
| `--data_dir coco/val2017` | `dataset.data_dir=coco/val2017` |
| `--cat_ids '[16]'` | `dataset.cat_ids=[16]` |
| `--hbsn_checkpoint ...` | `net.hbsn_checkpoint=...` |
| `--comment xxx` | `recorder.comment=xxx` |

## 数据

- `img/**/*.npy`：训练数据（float32 CHW (2,256,256)），由
  `tools/convert_mat_to_npy.py` 从 `.mat`（float64）转换（本机 .mat 已删除，
  远程服务器有备份）。`.npy` 为当前唯一本地数据，不入 git、不随 Syncthing
  同步（`.gitignore`/`.stignore`）。
- `coco/`：COCO 分割数据（本机 train2017 为空，默认配置指向 val2017）。
- 数据生成：`tools/build_image.py`（合成形状图像）+ 旧版 MATLAB 工程（HBS 字段）。

### 数据准备步骤（从 .mat 到 .npy）

```bash
python tools/convert_mat_to_npy.py --root img
# 递归 img/**/*.mat → float32 CHW (2,256,256) .npy
# 原子写（tmp+rename）、幂等（已存在跳过=断点续转）、末尾写 manifest.json（sha256）
```

## Checkpoint

- **格式**：`config` 键为纯 dict（net/dataset/recorder/run 四节），无类路径
  依赖；optimizer 状态已弃（重写后参数顺序变化，续训语义不可保）。
- **旧 ckpt 迁移**：`python tools/migrate_checkpoints.py "runs/*/*/checkpoints/*.pth" --out-dir runs/migrated`
  （只读原文件，输出镜像原结构；hbsn.* 键按旧运行时语义丢弃）。
- **评估**：`python scripts/eval_coco.py --model unetpp --checkpoint runs/migrated/unetpp/May17_10-17-34_hbs0.05_all_c/checkpoints/epoch_350.pth --data-dir coco/val2017 --annotation-path coco/annotations/instances_val2017.json --set connected=true --set single_instance=true`

### 论文基准（coco/val2017，connected+single_instance，batch 1）

| 模型 | checkpoint | F1 | IoU | 来源 |
|---|---|---|---|---|
| unetpp（hbsn 0.05） | May17_10-17-34/epoch_350.pth | 0.7831 → **0.7820** | 0.7110 → **0.7098** | test_full.py 记录 vs 迁移后复现 |

## 测试与开发

```bash
uv run python -m pytest   # 15 个：nets(5 模型冒烟)/config/dataset/geometry/migrate
uv run ruff check .       # 0 error（默认规则集，含 PEP585/604）
```

- pytest **必须从仓库根 `python/`** 跑（测试用相对路径 `img/`、`coco/`）。
- 提交用 Conventional Commits；提交前保证 pytest 全绿 + ruff clean。

## 已知行为说明（有意保留的旧语义）

- `is_soft_label`（默认 true）：SoftLabel 变换作用于**输入图像**（语义上应作用
  于标签，但旧行为如此，保留以保证数值等价）。
- `pin_memory` 默认 false：WSL2+torch 2.13 下 CachingHostAllocator 不复用，
  主进程 RSS 无界增长（Phase A 实测 36MB/迭代）。
- 训练轨迹在 GPU 上不可逐位复现（cuDNN/调度噪声）：等价性验证以
  iteration-0 逐位一致 + 全程统计分布一致为标准。

## Troubleshooting

| 症状 | 原因 / 处理 |
|---|---|
| pytest 报 FileNotFoundError（img/、coco/） | 从仓库根 `python/` 跑；或数据未准备（见「数据准备」） |
| 训练 OOM / RSS 无界增长 | `pin_memory` 保持 false；减 `num_workers` 或 batch_size |
| 未知 key 报错 | hydra override 的 key 必须在对应 yaml/schema 中存在 |
| 加载迁移 ckpt 报 strict 错误 | 旧 ckpt 的 config 可能缺新字段，用 `eval_coco --set`/schema 过滤 |
| 升级 torch/tv/smp 后 load 失败 | 精确钉版；`maskrcnn`/`deeplab` 手抄内部 API、unetpp 键集绑定版本 |
| 跨文件移动后 `.venv` 失效 | `rm -rf .venv && uv venv && uv pip install -e ".[dev]"` |

## 环境钉版

torch==2.13.0+cu130 / torchvision==0.28.0 / segmentation-models-pytorch==0.3.3
（maskrcnn/deeplab 手抄 torchvision 内部 API、unetpp ckpt 键集与 smp 版本绑定，
升级前务必核对）。numpy>=2 / matplotlib>=3.8 / opencv>=4.10（numpy 2 兼容下限）。

## 引用

论文待发表（`../latex/`）。代码基于旧仓库重写；历史 checkpoint 在
`runs/migrated/` 保留为论文实验证据。
