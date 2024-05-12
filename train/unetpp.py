import os

import click
import torch

from data.coco_dataset import CocoDataset
from net.unetpp import UnetPP
from recoder import CocoHBSNRecoder
from train.base_train import list_para_handle, training

COCO_ROOT = 'coco/train2017/'
COCO_ANNOTATION = 'coco/annotations/instances_train2017.json'
CAT_IDS = [16]

RESIZE_RATE = 1.5
AUGMENT_ROTATION = 30.0
AUGMENT_SCALE = [0.8, 1.2]
AUGMENT_TRANSLATE = [0.1, 0.1]

DICE_RATE = 0.1
IOU_RATE = 0
MASK_SCALE = 100
HBS_LOSS_RATE = 1
HBSN_CHECKPOINT = 'runs/hbsn/Apr05_09-38-40_stn3_loog3/checkpoints/best_1481.pth'

WEIGHT_NORM = 1e-5
MOMENTS = 0.9
LR = 1e-3
LR_DECAY_RATE = 0.5
LR_DECAY_STEPS = [50,100]
FINETUNE_RATE = 0.1

LOG_BASE_DIR = 'runs/unetpp'

DEVICE = "cuda:0"
TOTAL_EPOCHES = 1000
BATCH_SIZE = 64

VERSION = "0.1"
# VERSION 0.1
# initial version


@click.command()
# COCO settings
@click.option("--coco_root", default=COCO_ROOT, help="")
@click.option("--coco_annotation", default=COCO_ANNOTATION, help="")
@click.option("--cat_ids", default=str(CAT_IDS), help="")
# Dataset settings
@click.option("--resize_rate", default=RESIZE_RATE, help="Resize rate for data augmentation")
@click.option("--is_augment", is_flag=True, help="Use data augmentation or not")
@click.option("--augment_rotation", default=AUGMENT_ROTATION, help="Rotation range for data augmentation")
@click.option("--augment_scale", default=str(AUGMENT_SCALE), help="Scale range for data augmentation")
@click.option("--augment_translate", default=str(AUGMENT_TRANSLATE), help="Translate range for data augmentation")
# Model settings
@click.option("--dice_rate", default=DICE_RATE, help="Dice loss rate")
@click.option("--iou_rate", default=IOU_RATE, help="IoU loss rate")
@click.option("--hbsn_checkpoint", default=HBSN_CHECKPOINT, help="HBSN checkpoint path")
@click.option("--hbs_loss_rate", default=HBS_LOSS_RATE)
@click.option("--mask_scale", default=MASK_SCALE, help="Mask scale for sigmoid function")
# Optimizer settings
@click.option("--weight_norm", default=WEIGHT_NORM, help="Weight decay")
@click.option("--moments", default=MOMENTS, help="Momentum")
@click.option("--lr", default=LR, help="Learning rate")
@click.option("--lr_decay_rate", default=LR_DECAY_RATE, help="Learning rate decay rate")
@click.option("--lr_decay_steps", default=str(LR_DECAY_STEPS), help="Learning rate decay steps", type=str)
@click.option("--is_freeze", is_flag=True, help="Freeze layers or not")
@click.option("--finetune_rate", default=FINETUNE_RATE, help="Fixable layers finetune rate")
# Recoder settings
@click.option("--log_dir", default='', help="Log directory")
@click.option("--log_base_dir", default=LOG_BASE_DIR)
@click.option("--comment", default='')
# Training settings
@click.option("--device", default=DEVICE, help="Device to run the training")
@click.option("--total_epoches", default=TOTAL_EPOCHES, help="Total epoches to train")
@click.option("--batch_size", default=BATCH_SIZE, help="Batch size")
@click.option("--load", default="", help="Load model from checkpoint")
# Other settings
@click.option("--version", default=VERSION, help="Version of the model")
def main(
    coco_root, coco_annotation, cat_ids,
    resize_rate, is_augment, augment_rotation, augment_scale, augment_translate, 
    dice_rate, iou_rate, hbsn_checkpoint, hbs_loss_rate, mask_scale,
    weight_norm, moments, lr, lr_decay_rate, lr_decay_steps,
    is_freeze, finetune_rate,
    log_dir, log_base_dir, comment,
    device, total_epoches, batch_size, load,
    version
    ):
    lr_decay_steps = list_para_handle(lr_decay_steps)
    augment_scale = list_para_handle(augment_scale, float)
    augment_translate = list_para_handle(augment_translate, float)
    cat_ids = list_para_handle(cat_ids, int)
    
    config = {
        "coco_root": coco_root,
        "coco_annotation":coco_annotation,
        "device": device,
        "total_epoches": total_epoches,
        "lr": lr,
        "weight_norm": weight_norm,
        "moments": moments,
        "batch_size": batch_size,
        "version": version,
        "lr_decay_rate": lr_decay_rate,
        "lr_decay_steps": lr_decay_steps,
        "finetune_rate": 'freezed' if is_freeze else finetune_rate,
        "is_augment": (augment_rotation, augment_scale, augment_translate) if is_augment else False,
        "dice_rate": dice_rate,
        "iou_rate": iou_rate,
        "hbs_loss_rate": hbs_loss_rate,
        "hbsn_checkpoint": hbsn_checkpoint,
        "mask_scale": mask_scale
    }
    dataset = CocoDataset(
        coco_root, coco_annotation, 
        cat_ids=cat_ids, connected=True, single_instance=True,
        is_augment=is_augment, augment_rotation=augment_rotation,
        augment_scale=augment_scale, augment_translate=augment_translate,
        resize_rate=resize_rate
    )
    train_dataloader, test_dataloader = dataset.get_dataloader(batch_size=batch_size)
    
    recoder = CocoHBSNRecoder(
        config, len(train_dataloader), len(test_dataloader), 
        log_dir=log_dir, log_base_dir=log_base_dir ,comment=comment
        )
    
    net = UnetPP(
        height=dataset.height, width=dataset.width, 
        hbsn_checkpoint=hbsn_checkpoint, hbs_loss_rate=hbs_loss_rate,
        dice_rate=dice_rate, iou_rate=iou_rate, mask_scale=mask_scale, 
        device=device
    )

    param_dict = net.get_param_dict(lr, is_freeze, finetune_rate)
    optimizer = torch.optim.Adam(param_dict, lr=lr, weight_decay=weight_norm, betas=(moments, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_decay_steps, gamma=lr_decay_rate)
    
    training(
        net, recoder, optimizer, scheduler,
        train_dataloader, test_dataloader, load
    )
