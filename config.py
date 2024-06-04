import os
from datetime import datetime


class BaseConfig:
    load=''
    version='0.0'
    
    log_dir=''
    log_base_dir='runs'
    comment=''
    
    weight_norm=1e-5
    moments=0.9
    lr=1e-3
    lr_decay_rate=0.5
    lr_decay_steps=[50,100]
    
    batch_size=32
    device='cuda:0'
    total_epoches=1000
    
    is_augment=False
    augment_rotation=180.0
    augment_scale=[0.8,1.2]
    augment_translate=[0.1,0.1]

    is_freeze=False
    finetune_rate=1
    
    def __init__(self, config_dict):
        for attr in dir(self):
            if attr in config_dict:
                self.__setattr__(attr, config_dict[attr])

        if not self.log_dir:
            current_time = datetime.now().strftime("%b%d_%H-%M-%S")
            self.log_dir = os.path.join(
                self.log_base_dir, current_time
            )
            if self.comment:
                self.log_dir += f"_{self.comment}"
                
        self.checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
                
    @property
    def augment(self):
        return (self.augment_rotation, self.augment_scale, self.augment_translate) if self.is_augment else False
    
    @property
    def finetune(self):
        return 'freeze' if self.is_freeze else self.finetune_rate
    
    @property
    def _except_keys(self):
        return [
            'log_base_dir', 'comment', 'is_augment', 'is_freeze', 'finetune_rate',
            'augment_rotation', 'augment_scale', 'augment_translate', 'checkpoint_dir',
        ]
    
    @property
    def _show_keys(self):
        return [
            attr for attr in dir(self)
            if (
                not attr.startswith('_') and 
                not attr.startswith('get') and 
                attr not in self._except_keys
            )
        ]
        
    def get_config(self):
        return {
            attr: self.__getattribute__(attr)
            for attr in self._show_keys
        }

    def get_config_str_list(self):
        return [f'{k}: {v}' for k, v in self.get_config().items()]


class HBSNetConfig(BaseConfig):
    data_dir="img/generated"
    test_data_dir="img/gen2"
    
    channels=[8,16,32,64,128,256]
    stn_rate=0.1
    grad_rate=0.0
    log_base_dir='runs/hbsn'
    
    stn_mode=0
    radius=50
    
    is_soft_label=True
    channels_down=[8,8,16,32,64,128]
    channels_up=[8,16,32,64,128]
    

class SegNetConfig(BaseConfig):
    coco_root='coco/train2017/'
    coco_annotation='coco/annotations/instances_train2017.json'
    cat_ids=[16]
    resize_rate=1.5
    min_area=500
    connected=True
    single_instance=True
    
    dice_rate=0.1
    iou_rate=0.0
    hbs_loss_rate=1.0
    hbsn_checkpoint='runs/hbsn/Apr05_09-38-40_stn3_loog3/checkpoints/best_1481.pth'
    hbsn_version=1
    mask_scale=100
    
    hbsn_channels=[64, 128, 256, 512]
    hbsn_radius=50
    hbsn_stn_mode=0
    hbsn_stn_rate=0.0
    

    log_base_dir='runs/maskrcnn'

    
    @property
    def _except_keys(self):
        return super()._except_keys + [
            'hbsn_channels', 'hbsn_radius', 'hbsn_stn_mode', 'hbsn_stn_rate'
        ] if self.hbsn_checkpoint else ['hbsn_checkpoint']