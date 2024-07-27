from typing import Any, Dict

import torch

from config.base import BaseConfig


class BaseNetConfig(BaseConfig):
    device = "cpu"
    dtype = torch.float32
    height = 256
    width = 256
    input_channels = 1
    output_channels = 2
    load_strict = True

    # If `is_freeze` is True, the `fixable_layers` will be freezed
    # and only the other layers will be trained
    #
    # Otherwise, `fixable_layers` will be trained with
    # a smaller lr = lr * finetune_rate
    is_freeze = False
    finetune_rate = 1

    @property
    def finetune(self):
        return "freeze" if self.is_freeze else self.finetune_rate

    @property
    def _except_keys(self):
        return super()._except_keys + ["is_freeze", "finetune_rate"]


class HBSNetConfig(BaseNetConfig):
    stn_rate = 0.1
    grad_rate = 0.0

    stn_mode = 3
    # stn_mode: 0 - no stn,
    #           1 - pre stn
    #           2 - post stn
    #           3 - both stn

    radius = 50
    channels_down = [8, 8, 16, 32, 64, 128]
    channels_up = [8, 16, 32, 64, 128]
    is_skip = True

    def __init__(self, config_dict: dict = {}):
        super().__init__(config_dict)
        self._output_rate = 2 ** (
            len(self.channels_up) - len(self.channels_down)
        )
        self.output_height = int(self.height * self._output_rate)
        self.output_width = int(self.width * self._output_rate)


class SegHBSNNetConfig(BaseNetConfig):
    dice_rate = 0.1
    iou_rate = 0
    hbs_loss_rate = 1.0
    mask_scale = 10
    hbsn_checkpoint = ""

    # `is_freeze` is set to True by default
    # since we want to freeze the inside HBSNet.
    is_freeze = True

    def __init__(
        self,
        config_dict: Dict[str, Any] = {},
        hbsn_config: HBSNetConfig | None = None,
    ):
        super().__init__(config_dict)
        if hbsn_config:
            self.hbsn_config = hbsn_config
        else:
            self.hbsn_config = HBSNetConfig()

    @property
    def _except_keys(self):
        return super()._except_keys + ["hbsn_config"]

    def get_config(self):
        config = super().get_config()
        config.update(self.hbsn_config.get_config())
        return config


class MaskRCNNConfig(SegHBSNNetConfig):
    select_num = 10
    weight_hidden_size = 20


class TPSNConfig(SegHBSNNetConfig):
    qc_loss_rate: float = 0.01
    lap_loss_rate: float = 0.0001
