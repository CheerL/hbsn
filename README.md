# HBSN — Harmonic Beltrami Signature Network

论文配套代码库（PyTorch）。目录原为 `src`，2026-08 激进现代化重写后为
`python/`（git 仓库根）：flat-layout 包 `hbsn/` + hydra 配置 + pyproject。

## 快速开始

```bash
uv venv --python 3.10 && uv pip install -e ".[dev]"
python -m hbsn.train model=hbsn run.total_epoches=100 net.device=cuda:0
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
python -m hbsn.train model=tpsn net.stn_mode=3 net.hbsn_checkpoint=runs/migrated/hbsn/Jun11_13-01-53_big_ns/checkpoints/best.pth \
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

## 已知行为说明（有意保留的旧语义）

- `is_soft_label`（默认 true）：SoftLabel 变换作用于**输入图像**（语义上应作用
  于标签，但旧行为如此，保留以保证数值等价）。
- `pin_memory` 默认 false：WSL2+torch 2.13 下 CachingHostAllocator 不复用，
  主进程 RSS 无界增长（Phase A 实测 36MB/迭代）。
- 训练轨迹在 GPU 上不可逐位复现（cuDNN/调度噪声）：等价性验证以
  iteration-0 逐位一致 + 全程统计分布一致为标准。

## 测试

```bash
python -m pytest   # 15 个：nets（5 模型冒烟）/config/dataset/geometry/migrate
```

## 环境钉版

torch==2.13.0+cu130 / torchvision==0.28.0 / segmentation-models-pytorch==0.3.3
（maskrcnn/deeplab 手抄 torchvision 内部 API、unetpp ckpt 键集与 smp 版本绑定，
升级前务必核对）。numpy>=2 / matplotlib>=3.8 / opencv>=4.10（numpy 2 兼容下限）。
