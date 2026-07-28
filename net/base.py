from typing import Dict, Tuple

import torch
import torch.nn as nn

from config import BaseNetConfig

class BaseNet(nn.Module):
    def __init__(self, config: BaseNetConfig):
        super().__init__()
        self.config = config

    @property
    def uninitializable_layers(self):
        return nn.Module()

    @property
    def fixable_layers(self):
        return nn.Module()

    def forward(self, img):
        raise NotImplementedError()

    def loss(
        self, predict, ground_truth
    ) -> Tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]:
        raise NotImplementedError()

    def initialize(self):
        non_initializable_modules = list(self.uninitializable_layers.modules())

        for m in self.modules():
            if (
                isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d))
                and m not in non_initializable_modules
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
        self,
        path,
        epoch,
        best_epoch,
        best_loss,
        config={},
        optimizer=None,
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

    def _handle_checkpoint(self, checkpoint):
        return checkpoint

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.config.device, weights_only=False)
        checkpoint = self._handle_checkpoint(checkpoint)
        self.load_state_dict(checkpoint["state_dict"], self.config.load_strict)
        epoch = checkpoint["epoch"]
        best_epoch = checkpoint["best_epoch"]
        best_loss = checkpoint["best_loss"]
        optimizer_data = (
            checkpoint["optimizer"] if "optimizer" in checkpoint else {}
        )

        return epoch, best_epoch, best_loss, optimizer_data

    @staticmethod
    def load_model(path, device="cpu"):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        epoch = checkpoint["epoch"]
        best_epoch = checkpoint["best_epoch"]
        best_loss = checkpoint["best_loss"]
        return checkpoint, config, epoch, best_epoch, best_loss

    def freeze_fixable_layers(self):
        for param in self.fixable_layers.parameters():
            param.requires_grad = False

    def get_param_dict(self, lr):
        fixable_param_ids = [
            id(param) for param in self.fixable_layers.parameters()
        ]
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
