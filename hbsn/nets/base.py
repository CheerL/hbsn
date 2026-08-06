"""BaseNet：统一的 save/load/initialize/参数分组逻辑。"""

import torch
from torch import nn


def torch_dtype(config) -> torch.dtype:
    """配置里的 dtype 是字符串名（OmegaConf 合并后无 dataclass property）。"""
    return getattr(torch, config.dtype)


def output_size(config) -> tuple[int, int]:
    """HBSNet 输出尺寸：由 channels_up/down 层数差的降采样率决定（256→128）。"""
    rate = 2 ** (len(config.channels_up) - len(config.channels_down))
    return int(config.height * rate), int(config.width * rate)


class BaseNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @property
    def uninitializable_layers(self) -> nn.Module:
        return nn.Module()

    @property
    def fixable_layers(self) -> nn.Module:
        return nn.Module()

    def forward(self, img):
        raise NotImplementedError()

    def loss(
        self, predict, ground_truth
    ) -> tuple[dict[str, torch.Tensor], tuple[torch.Tensor, ...]]:
        raise NotImplementedError()

    def initialize(self):
        non_initializable_ids = {
            id(m) for m in self.uninitializable_layers.modules()
        }
        for m in self.modules():
            if (
                isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d))
                and id(m) not in non_initializable_ids
            ):
                nn.init.xavier_uniform_(m.weight)

    def get_input_shape(self, batch_size=1):
        return (
            batch_size,
            self.config.input_channels,
            self.config.height,
            self.config.width,
        )

    def save(
        self, path, epoch, best_epoch, best_loss, config=None, optimizer=None
    ):
        torch.save(
            {
                "state_dict": self.state_dict(),
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_loss": best_loss,
                "config": config,
                "optimizer": optimizer.state_dict() if optimizer else None,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(
            path, map_location=self.config.device, weights_only=False
        )
        self.load_state_dict(checkpoint["state_dict"], self.config.load_strict)
        return (
            checkpoint["epoch"],
            checkpoint["best_epoch"],
            checkpoint["best_loss"],
            checkpoint.get("optimizer"),
        )

    @staticmethod
    def load_model(path, device="cpu"):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        return (
            checkpoint,
            checkpoint["config"],
            checkpoint["epoch"],
            checkpoint["best_epoch"],
            checkpoint["best_loss"],
        )

    def freeze_fixable_layers(self):
        for param in self.fixable_layers.parameters():
            param.requires_grad = False

    def get_param_dict(self, lr):
        fixable_param_ids = {
            id(param) for param in self.fixable_layers.parameters()
        }
        params = [
            param
            for param in self.parameters()
            if id(param) not in fixable_param_ids
        ]
        param_dict = [{"params": params, "initial_lr": lr}]
        if self.config.is_freeze:
            self.freeze_fixable_layers()
        else:
            param_dict.append(
                {
                    "params": list(self.fixable_layers.parameters()),
                    "initial_lr": lr * self.config.finetune_rate,
                }
            )
        return param_dict
