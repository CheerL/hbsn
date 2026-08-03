"""跨字段配置校验：compose 后立即执行，配错立即报错而不是静默带病训练。"""
import os

from omegaconf import DictConfig, OmegaConf

from hbsn.config.schemas import CocoDatasetSchema, HbsnDatasetSchema


def validate_config(cfg: DictConfig, net_schema_cls, dataset_schema_cls) -> tuple:
    """合并 + 跨字段校验，返回 (merged_net, merged_dataset)。"""
    # 类型化合并：未知 key 在 OmegaConf.structured 合并时抛 ValidationError
    merged_net = OmegaConf.merge(OmegaConf.structured(net_schema_cls), cfg.net)
    merged_dataset = OmegaConf.merge(
        OmegaConf.structured(dataset_schema_cls), cfg.dataset
    )
    net, dataset = merged_net, merged_dataset

    # 跨字段约束
    assert net.stn_mode in {0, 1, 2, 3}, f"stn_mode must be 0..3, got {net.stn_mode}"
    assert len(net.channels_up) == len(net.channels_down) - 1, (
        f"len(channels_up)={len(net.channels_up)} must equal "
        f"len(channels_down)-1={len(net.channels_down)-1}"
    )

    if dataset_schema_cls is HbsnDatasetSchema:
        # masked_size 与 HBSNet 输出降采样（256→128）的隐式耦合显式化
        rate = 2 ** (len(net.channels_up) - len(net.channels_down))
        output_height = int(net.height * rate)
        assert dataset.masked_size * 2 < net.height, "masked_size too large"
        assert net.height - 2 * dataset.masked_size == output_height, (
            f"masked_size={dataset.masked_size} 与输出尺寸不匹配："
            f"输入 {net.height} 裁剪后 {net.height - 2 * dataset.masked_size}，"
            f"但网络输出 {output_height}"
        )
        for d in (dataset.data_dir, dataset.test_data_dir):
            if d:
                assert os.path.isdir(d), f"data_dir 不存在: {d}"

    if dataset_schema_cls is CocoDatasetSchema:
        assert os.path.isfile(dataset.annotation_path), (
            f"annotation_path 不存在: {dataset.annotation_path}"
        )
        if dataset.test_annotation_path:
            assert os.path.isfile(dataset.test_annotation_path)

    # 覆盖后的值写回，保证后续代码拿到的就是校验后的配置
    return merged_net, merged_dataset
