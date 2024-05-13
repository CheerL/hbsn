from config import SegNetConfig
from data.coco_dataset import CocoDataset
from net.deeplab import DeepLab
from recoder import CocoHBSNRecoder
from train.base_train import training


def main(
    coco_root='coco/train2017/', 
    coco_annotation='coco/annotations/instances_train2017.json', 
    cat_ids=[16],
    resize_rate=1.5, 
    is_augment=False, 
    augment_rotation=30.0, 
    augment_scale=[0.8,1.2], 
    augment_translate=[0.1,0.1], 
    dice_rate=0.1, 
    iou_rate=0.0, 
    hbs_loss_rate=1.0,
    hbsn_checkpoint='runs/hbsn/Apr05_09-38-40_stn3_loog3/checkpoints/best_1481.pth',  
    mask_scale=100,
    weight_norm=1e-5, 
    moments=0.9, 
    lr=1e-3, 
    lr_decay_rate=1, 
    lr_decay_steps=[],
    is_freeze=False, 
    finetune_rate=1,
    log_dir='', 
    log_base_dir='runs/deeplab', 
    comment='',
    device='cuda:0', 
    total_epoches=1000, 
    batch_size=32, 
    load='',
    version=0.1
    ):
    args = locals()
    config = SegNetConfig(args)
    dataset = CocoDataset(config=config, connected=True, single_instance=True)
    train_dataloader, test_dataloader = dataset.get_dataloader(batch_size=config.batch_size)
    recoder = CocoHBSNRecoder(
        config, len(train_dataloader), len(test_dataloader), 
    )
    net = DeepLab(
        height=dataset.height, width=dataset.width, config=config
    )

    training(
        net, recoder, config,
        train_dataloader, test_dataloader
    )
