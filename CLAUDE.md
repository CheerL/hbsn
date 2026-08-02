# HBSN 项目地图（Claude 速览）

研究项目：Harmonic Beltrami Signature Network（论文在 `../latex/`，独立 git 仓库）。
本目录 `Python/` 是代码仓库根（原 `src`，2026-08 重写）。

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
scripts/eval_coco.py  test_full/test_script 合一评估
tests/                15 个 pytest
```

## 关键约束（改动前必读）

1. **state_dict 键不可随意改**：`pre_stn.*/post_stn.*/backbone.*`（HBSNet）、
   `hbsn.*`（SegHBSNNet 子网，旧 ckpt 运行时语义为丢弃）、`model.*`
   （smp 0.3.3 / torchvision 0.28 键集，升级版本会静默换键）、
   `mask_conv.*/weight_layer.*/lap_loss.weight`。类名可改，属性名勿动。
2. **config 字段名与旧 Config 属性逐字一致**（pickle 桩反序列化的硬约束）。
3. **OmegaConf 合并后 dataclass @property 不可用**：派生值用
   `hbsn/nets/base.py` 的 `torch_dtype()`/`output_size()` 辅助函数。
4. **浮点等价验证**：GPU 训练轨迹不可逐位复现；以 iteration-0 一致 +
   分布一致为准（见 README「已知行为说明」）。
5. **数据**：训练读 `.npy`（float32 CHW），`.mat` 是备份不删除；
   `.mat`/`.npy` 都已被 git/Syncthing 排除。
6. 本机仅 15GB 内存：`pin_memory` 默认 false（WSL2 CachingHostAllocator
   无界增长）；DataLoader worker 内存 ~1.2GB/个。
7. hydra 配置文件的完整默认值不可删（override 依赖 key 存在）。

## 常用命令

```bash
python -m hbsn.train model=hbsn run.total_epoches=1 net.device=cuda:0   # 训练
python scripts/eval_coco.py --model unetpp --checkpoint <glob> --set connected=true --set single_instance=true  # 评估
python tools/migrate_checkpoints.py "runs/*/*/checkpoints/*.pth" --out-dir runs/migrated  # ckpt 迁移
python -m pytest
```

## 仓库状态

- `master`：Phase A/B/C 后的旧代码基线（可回滚）；`rewrite`：重写后的新代码。
- 论文证据：12 个旧 ckpt 已迁移到 `runs/migrated/`（原文件只读 + tar 备份在
  `/home/nnb/projects/HBSN/.backup/ckpt-2026-08-02.tar.gz`）。
- `.venv_old`：旧 venv（Phase A 修复后代码可用），Phase E 验证通过后可删。
