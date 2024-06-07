from re import T
from typing import Any, Dict, Tuple

from torch.utils.data import DataLoader

from config import Config, RunConfig
from data.coco_dataset import CocoDataset, CocoDatasetConfig
from data.hbsn_dataset import HBSNDataset, HBSNDatasetConfig
from net.deeplab import DeepLab
from net.hbsn import HBSNet, HBSNetConfig
from net.maskrcnn import MaskRCNN, MaskRCNNConfig
from net.seg_hbsn_net import SegHBSNNetConfig
from net.unetpp import UnetPP
from net.tpsn import TPSN
from recoder import CocoHBSNRecoder, HBSNRecoder, RecoderConfig


def config_factory(type_: str, config_dict: Dict[str, Any]):
    if type_ == "hbsn":
        net_config = HBSNetConfig(config_dict)
        dataset_config = HBSNDatasetConfig(config_dict)
    else:
        dataset_config = CocoDatasetConfig(config_dict)
        if type_ == "maskrcnn":
            net_config = MaskRCNNConfig(config_dict)
        else:
            net_config = SegHBSNNetConfig(config_dict)

    recorder_config = RecoderConfig(config_dict)
    run_config = RunConfig(config_dict)

    config = Config(net_config, dataset_config, recorder_config, run_config)
    return config


def net_factory(type_: str, config: Config):
    if type_ == "hbsn":
        assert isinstance(config.net_config, HBSNetConfig), "HBSNetConfig required"
        net = HBSNet.factory(config.net_config)
        config.recorder_config.log_base_dir = "runs/hbsn"
    elif type_ == "maskrcnn":
        assert isinstance(config.net_config, MaskRCNNConfig), "MaskRCNNConfig required"
        net = MaskRCNN.factory(config.net_config)
        config.recorder_config.log_base_dir = "runs/maskrcnn"
    elif type_ == "deeplab":
        assert isinstance(
            config.net_config, SegHBSNNetConfig
        ), "SegHBSNNetConfig required"
        net = DeepLab.factory(config.net_config)
        config.recorder_config.log_base_dir = "runs/deeplab"
    elif type_ == "unetpp":
        assert isinstance(
            config.net_config, SegHBSNNetConfig
        ), "SegHBSNNetConfig required"
        net = UnetPP.factory(config.net_config)
        config.recorder_config.log_base_dir = "runs/unetpp"
    elif type_ == 'tpsn':
        net = TPSN.factory(config.net_config)
        config.recorder_config.log_base_dir = "runs/tpsn"

    return net


def dataset_factory(type_: str, config: Config) -> Tuple[DataLoader, DataLoader]:
    if type_ == "hbsn":
        assert isinstance(
            config.dateset_config, HBSNDatasetConfig
        ), "HBSNDatasetConfig required"
        dataset = HBSNDataset(config.dateset_config)
        if config.dateset_config.test_data_dir:
            test_dataset = HBSNDataset(config.dateset_config, is_test=True)
        else:
            test_dataset = None
    else:
        assert isinstance(
            config.dateset_config, CocoDatasetConfig
        ), "CocoDatasetConfig required"
        dataset = CocoDataset(config.dateset_config)
        if (
            config.dateset_config.test_data_dir
            and config.dateset_config.test_annotation_path
        ):
            test_dataset = CocoDataset(config.dateset_config, is_test=True)
        else:
            test_dataset = None

    if test_dataset:
        train_dataloader, _ = dataset.get_dataloader(
            batch_size=config.run_config.batch_size, split_rate=1
        )
        _, test_dataloader = test_dataset.get_dataloader(
            batch_size=config.run_config.batch_size, split_rate=0
        )
    else:
        train_dataloader, test_dataloader = dataset.get_dataloader(
            batch_size=config.run_config.batch_size
        )

    assert train_dataloader is not None, "Train dataloader is None"
    assert test_dataloader is not None, "Test dataloader is None"
    return train_dataloader, test_dataloader


def recoder_factory(type_: str, config: Config, train_size: int, test_size: int):
    if type_ == "hbsn":
        recoder = HBSNRecoder(
            config.recorder_config,
            train_size,
            test_size,
            config.run_config.total_epoches,
            config.run_config.batch_size,
        )
    elif type_ in ["maskrcnn", "deeplab", "unetpp"]:
        recoder = CocoHBSNRecoder(
            config.recorder_config,
            train_size,
            test_size,
            config.run_config.total_epoches,
            config.run_config.batch_size,
        )

    return recoder
