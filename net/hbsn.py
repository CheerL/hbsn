import torch
import torch.nn as nn
import torch.nn.functional as F

from net.bce import BCENet
from net.stn import STN

DTYPE = torch.float32

class HBSNet(nn.Module):
    def __init__(
        self, height, width,
        input_channels=1, output_channels=2,
        channels=[16, 32, 64, 128], radius=50,
        device="cpu", dtype=DTYPE, stn_mode=0, stn_rate=0.0
        ):
        # stn_mode: 0 - no stn, 
        #           1 - pre stn
        #           2 - post stn
        #           3 - both stn
        super(HBSNet, self).__init__()
        self.dtype = dtype
        self.device = device
        self.height = height
        self.width = width
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stn_mode = stn_mode
        self.stn_rate = stn_rate
        
        if stn_mode == 1 or stn_mode == 3:
            self.pre_stn = STN(input_channels, height, width, dtype=dtype, is_rotation_only=False)
        
        if stn_mode == 2 or stn_mode == 3:
            self.post_stn = STN(output_channels, height, width, dtype=dtype, is_rotation_only=True)
        
        self.bce = BCENet(input_channels, output_channels, channels, bilinear=True, dtype=dtype)
        self.mask = self.create_mask(radius)
        self.to(device)

    def create_mask(self, r):
        x, y = torch.meshgrid(torch.arange(self.width),torch.arange(self.height), indexing='ij')
        x = (x - self.width/2) / r
        y = (y - self.height/2) / r
        mask = (x**2 + y**2) <= 1
        mask.requires_grad = False
        mask = mask.to(self.device)
        return mask

    def forward(self, x):
        if self.stn_mode == 1 or self.stn_mode == 3:
            x, _ = self.pre_stn(x)
            
        x = self.bce(x)
        
        if self.stn_mode == 2 or self.stn_mode == 3:
            x, theta = self.post_stn(x)

        x = torch.masked_fill(x, ~self.mask, 0.0)
        return x, theta

    def loss(self, predict, label, is_mask=True, stn_rate=0.0):
        if self.stn_mode == 2 or self.stn_mode == 3:
            label, theta = self.post_stn(label)
            double_stn_predict, double_theta = self.post_stn(predict)
            stn_loss = F.mse_loss(double_stn_predict, label, reduction="mean")
        else:
            stn_loss = 0

        if is_mask:
            hbs_loss= F.mse_loss(
                torch.masked_select(predict, self.mask), 
                torch.masked_select(label, self.mask), 
                reduction="mean")
        else:
            hbs_loss = F.mse_loss(predict, label, reduction="mean")
            
        loss = hbs_loss + stn_loss * self.stn_rate
        return loss, hbs_loss, stn_loss

    def initialize(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)

    def save(self, path, epoch, best_epoch, best_loss, config={}):
        torch.save({
            "state_dict": self.state_dict(),
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_loss": best_loss,
            'config': config
        }, path)
        
    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["state_dict"])
        epoch = checkpoint["epoch"]
        best_epoch = checkpoint["best_epoch"]
        best_loss = checkpoint["best_loss"]

        return epoch, best_epoch, best_loss
    
    @staticmethod
    def load_model(path, device=None):
        from data.dataset import HBSNDataset 
        
        # config = {
        #     "data_dir": data_dir,
        #     "device": device,
        #     "total_epoches": total_epoches,
        #     "lr": lr,
        #     "weight_norm": weight_norm,
        #     "moments": moments,
        #     "batch_size": batch_size,
        #     "channels": channels,
        #     "version": version,
        #     "lr_decay_rate": lr_decay_rate,
        #     "lr_decay_steps": lr_decay_steps,
        #     "stn_mode": stn_mode,
        #     "is_augment": (augment_rotation, augment_scale, augment_translate) if is_augment else False,
        #     "radius": radius
        # }
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        
        if config['is_augment']:
            augment_rotation, augment_scale, augment_translate = config['is_augment']
            dataset = HBSNDataset(
                config['data_dir'], is_augment=True,
                augment_rotation=augment_rotation, 
                augment_scale=augment_scale, 
                augment_translate=augment_translate
            )
        else:
            dataset = HBSNDataset(config['data_dir'], is_augment=False)
        H, W, C_input, C_output = dataset.get_size()
        train_dataloader, test_dataloader = dataset.get_dataloader(batch_size=config['batch_size'])
        
        # summary_writer = HBSNSummary(
        #     config, len(train_dataloader), len(test_dataloader), 
        #     log_dir=log_dir, comment=comment
        #     )
        # summary_writer.init_logger()
        
        net = HBSNet(
            height=H, width=W, input_channels=C_input, output_channels=C_output, 
            channels=config['channels'], device=device or config['device'], 
            dtype=DTYPE, stn_mode=config['stn_mode'], radius=config['radius'],
            stn_rate=config['stn_rate']
            )
        net.load_state_dict(checkpoint["state_dict"])
        epoch = checkpoint["epoch"]
        best_epoch = checkpoint["best_epoch"]
        best_loss = checkpoint["best_loss"]
        return net, epoch, best_epoch, best_loss, dataset, train_dataloader, test_dataloader
