"""类型化配置 schema（dataclass）。

字段名与旧版 `config/` 的类属性**逐字一致**——旧 checkpoint 里 pickle 的 Config
对象通过 tools/migrate_checkpoints.py 的 _legacy 桩反序列化，属性名对不上就失败。
"""

from dataclasses import dataclass, field


@dataclass
class BaseNetSchema:
    device: str = "cpu"
    dtype: str = "float32"  # torch dtype 名，代码里 getattr(torch, dtype)
    height: int = 256
    width: int = 256
    input_channels: int = 1
    output_channels: int = 2
    load_strict: bool = True
    is_freeze: bool = False
    finetune_rate: float = 1


@dataclass
class HBSNetSchema(BaseNetSchema):
    stn_rate: float = 0.1
    grad_rate: float = 0.0
    stn_mode: int = 3  # 0-无 STN, 1-仅 pre, 2-仅 post, 3-两者
    radius: int = 50
    channels_down: list = field(default_factory=lambda: [8, 8, 16, 32, 64, 128])
    channels_up: list = field(default_factory=lambda: [8, 16, 32, 64, 128])
    is_skip: bool = True


@dataclass
class SegNetSchema(BaseNetSchema):
    dice_rate: float = 0.1
    iou_rate: float = 0
    hbs_loss_rate: float = 1.0
    mask_scale: float = 10
    hbsn_checkpoint: str = ""
    input_channels: int = 3
    # 分割输出为单通道 mask（hbsn 子网输入 1 通道；旧默认 2 是 maskrcnn 潜伏缺陷）
    output_channels: int = 1
    is_freeze: bool = True  # 冻结内嵌 HBSNet
    hbsn_config: HBSNetSchema = field(default_factory=HBSNetSchema)


@dataclass
class MaskRcnnNetSchema(SegNetSchema):
    select_num: int = 10
    weight_hidden_size: int = 20


@dataclass
class TpsnNetSchema(SegNetSchema):
    mse_rate: float = 1
    qc_loss_rate: float = 0.01
    lap_loss_rate: float = 0.0001


@dataclass
class BaseDatasetSchema:
    data_dir: str = ""
    test_data_dir: str = ""
    is_augment: bool = False
    augment_rotation: float = 180
    augment_scale: list = field(default_factory=lambda: [0.8, 1.2])
    augment_translate: list = field(default_factory=lambda: [0.1, 0.1])
    is_soft_label: bool = True
    # pin_memory 默认 False：WSL2+torch 2.13 下 CachingHostAllocator 不复用，
    # 主进程 RSS 无界增长（Phase A 实测）
    pin_memory: bool = False
    num_workers: int = 4


@dataclass
class HbsnDatasetSchema(BaseDatasetSchema):
    data_dir: str = "img/generated"
    test_data_dir: str = "img/gen2"
    augment_rotation: float = 180
    augment_scale: list = field(default_factory=lambda: [0.5, 2])
    augment_translate: list = field(default_factory=lambda: [0.5, 0.5])
    masked_size: int = 64


@dataclass
class CocoDatasetSchema(BaseDatasetSchema):
    data_dir: str = "coco/train2017"
    test_data_dir: str = ""
    annotation_path: str = "coco/annotations/instances_train2017.json"
    test_annotation_path: str = ""
    height: int = 256
    width: int = 256
    img_ids: list = field(default_factory=list)
    cat_ids: list = field(default_factory=list)
    connected: bool = False
    single_instance: bool = False
    resize_rate: float = 1.5
    min_area: float = 500
    augment_rotation: float = 30
    augment_scale: list = field(default_factory=lambda: [0.8, 1.2])
    augment_translate: list = field(default_factory=lambda: [0.1, 0.1])


@dataclass
class RecorderSchema:
    log_dir: str = ""
    log_base_dir: str = "runs"
    comment: str = ""
    is_add_graph: bool = False


@dataclass
class RunSchema:
    checkpoint_path: str = ""
    version: str = "0.0"
    total_epoches: int = 1000
    batch_size: int = 64
    weight_norm: float = 1e-5
    moments: float = 0.9
    lr: float = 1e-4
    lr_decay_rate: float = 0.5
    lr_decay_steps: list = field(default_factory=lambda: [50, 100])
