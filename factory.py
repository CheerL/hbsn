from typing import Any, Dict, Tuple

from torch.utils.data import DataLoader

from config import Config, RunConfig
from data.coco_dataset import CocoDataset, CocoDatasetConfig
from data.hbsn_dataset import HBSNDataset, HBSNDatasetConfig
from net.deeplab import DeepLab
from net.hbsn import HBSNet, HBSNetConfig
from net.maskrcnn import MaskRCNN, MaskRCNNConfig
from net.seg_hbsn_net import SegHBSNNetConfig
from net.tpsn import TPSN, TPSNConfig
from net.unetpp import UnetPP
from recorder import (
    CocoHBSNRecorder,
    HBSNRecorder,
    RecorderConfig,
    TPSNRecorder,
)

TYPE_DICT = {
    "hbsn": (
        HBSNet,
        HBSNetConfig,
        HBSNDataset,
        HBSNDatasetConfig,
        HBSNRecorder,
    ),
    "maskrcnn": (
        MaskRCNN,
        MaskRCNNConfig,
        CocoDataset,
        CocoDatasetConfig,
        CocoHBSNRecorder,
    ),
    "deeplab": (
        DeepLab,
        SegHBSNNetConfig,
        CocoDataset,
        CocoDatasetConfig,
        CocoHBSNRecorder,
    ),
    "unetpp": (
        UnetPP,
        SegHBSNNetConfig,
        CocoDataset,
        CocoDatasetConfig,
        CocoHBSNRecorder,
    ),
    "tpsn": (
        TPSN,
        TPSNConfig,
        CocoDataset,
        CocoDatasetConfig,
        TPSNRecorder,
    ),
    "hbsnseg": (
        None,
        SegHBSNNetConfig,
        CocoDataset,
        CocoDatasetConfig,
        CocoHBSNRecorder,
    )
}


def type_check(type_: str):
    assert type_ in TYPE_DICT, "Invalid `type_`"


def config_factory(type_: str, config_dict: Dict[str, Any]) -> Config:
    type_check(type_)
    _, NetConfig, _, DatasetConfig, _ = TYPE_DICT[type_]

    net_config = NetConfig(config_dict)
    dataset_config = DatasetConfig(config_dict)
    recorder_config = RecorderConfig(config_dict)
    run_config = RunConfig(config_dict)

    return Config(net_config, dataset_config, recorder_config, run_config)


def net_factory(type_: str, config: Config):
    type_check(type_)
    Net, NetConfig, _, _, _ = TYPE_DICT[type_]
    assert isinstance(
        config.net_config, NetConfig
    ), f"{NetConfig.__name__} is required"
    net = Net.factory(config.net_config)
    return net


def dataset_factory(
    type_: str, config: Config
) -> Tuple[DataLoader, DataLoader]:
    type_check(type_)
    _, _, Dataset, DatasetConfig, _ = TYPE_DICT[type_]
    assert isinstance(
        config.dataset_config, DatasetConfig
    ), f"{DatasetConfig.__name__} is required"
    dataset = Dataset(config.dataset_config)
    if config.dataset_config.test_data_dir:
        test_dataset = Dataset(config.dataset_config, is_test=True)
        train_dataloader, _ = dataset.get_dataloader(
            batch_size=config.run_config.batch_size, split_rate=1, pin_memory=config.dataset_config.pin_memory
        )
        _, test_dataloader = test_dataset.get_dataloader(
            batch_size=config.run_config.batch_size, split_rate=0, pin_memory=config.dataset_config.pin_memory
        )
    else:
        test_dataset = None
        train_dataloader, test_dataloader = dataset.get_dataloader(
            batch_size=config.run_config.batch_size, pin_memory=config.dataset_config.pin_memory
        )

    assert train_dataloader is not None, "Train dataloader is None"
    assert test_dataloader is not None, "Test dataloader is None"
    return train_dataloader, test_dataloader


def recorder_factory(
    type_: str, config: Config, train_size: int, test_size: int
):
    type_check(type_)
    _, _, _, _, Recorder = TYPE_DICT[type_]
    config.recorder_config.log_base_dir = f"runs/{type_}"
    recorder = Recorder(
        config.recorder_config,
        train_size,
        test_size,
        config.run_config.total_epoches,
        config.run_config.batch_size,
    )

    return recorder
