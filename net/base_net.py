import torch
import torch.nn as nn

DTYPE = torch.float32

class BaseNet(nn.Module):
    def __init__(self, height, width, input_channels, output_channels, device="cpu", dtype=DTYPE, config=None):
        super().__init__()
        self.height = height
        self.width = width
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.dtype = dtype
        if config:
            self.device = config.device
        else:
            self.device = device
            

    @property
    def uninitializable_layers(self):
        return nn.Module()
    
    @property
    def fixable_layers(self):
        return nn.Module()


    def forward(self, x):
        raise NotImplementedError()

    def loss(self):
        raise NotImplementedError()

    def initialize(self):
        non_initializable_modules = list(self.uninitializable_layers.modules())

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)) and m not in non_initializable_modules:
                nn.init.xavier_uniform_(m.weight)
                
    def get_input_shape(self, batch_size=1):
        return (batch_size, self.input_channels, self.height, self.width)

    def save(self, path, epoch, best_epoch, best_loss, config={}, optimizer=None):
        torch.save({
            "state_dict": self.state_dict(),
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_loss": best_loss,
            'config': config,
            'optimizer': optimizer.state_dict() if optimizer else None
        }, path)
        
    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["state_dict"])
        epoch = checkpoint["epoch"]
        best_epoch = checkpoint["best_epoch"]
        best_loss = checkpoint["best_loss"]
        optimizer_data = checkpoint["optimizer"] if "optimizer" in checkpoint else {}

        return epoch, best_epoch, best_loss, optimizer_data
    
    @staticmethod
    def load_model(path, device='cpu'):
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        epoch = checkpoint["epoch"]
        best_epoch = checkpoint["best_epoch"]
        best_loss = checkpoint["best_loss"]
        return checkpoint, config, epoch, best_epoch, best_loss

    def freeze_fixable_layers(self):
        for param in self.fixable_layers.parameters():
            param.requires_grad = False

    def get_param_dict(self, lr, is_freeze=False, finetune_rate=1):
        fixable_param_ids = [id(param) for param in self.fixable_layers.parameters()]
        params = [
            param for param in self.parameters()
            if id(param) not in fixable_param_ids
        ]
        param_dict = [
            {'params': params, 'initial_lr': lr}
        ]
        if is_freeze:
            self.freeze_fixable_layers()
        else:
            param_dict.append(
                {'params': list(self.fixable_layers.parameters()), 'initial_lr': lr * finetune_rate}
            )
        # print(len(param_dict))
        return param_dict