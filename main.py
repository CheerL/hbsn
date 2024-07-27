import fire

from train import train
from validate import validate

if __name__ == "__main__":
    '''
    run this file with command:
    python main.py train|validate --type TYPE [--configs CONFIGS]
    
    TYPE: str, required
        type of model, should be one of [
            "hbsn",
            "maskrcnn",
            "deeplab",
            "unetpp",
            "tpsn"
        ]
        
    CONFIGS: There are many configurations for the model, containing 4 parts:
        net_config: configuration for the network
            For `BaseNet`: 
                --device cpu
                --dtype torch.float32
                --height 256
                --width 256
                --input_channels 1
                --output_channels 2
                --load_strict True
                --is_freeze False
                --finetune_rate 1
            For `HBSNet`:
                --stn_rate 0.1
                --grad_rate 0.0
                --stn_mode 3
                --radius 50
                --channels_down '[8, 8, 16, 32, 64, 128]'
                --channels_up '[8, 16, 32, 64, 128]'
                --is_skip True
            For `SegHBSNNet`:
                --dice_rate 0.1
                --iou_rate 0
                --hbs_loss_rate 1.0
                --mask_scale 10
                --is_freeze True
                --hbsn_checkpoint ""
                (if `hbsn_checkpoint` is not empty, 
                a new HBSNet will be created 
                and configs for `HBSNet` are required)
            For `MaskRCNN`:
                --select_num 10
                --weight_hidden_size 20
            For `TPSN`:
                --qc_loss_rate 0.01
                --lap_loss_rate 0.0001
        
        dataset_config: configuration for the dataset
            For `BaseDataset`:
                --data_dir ""
                --test_data_dir ""
                --is_augment False
                --augment_rotation 180
                --augment_scale '[0.8, 1.2]'
                --augment_translate '[0.1, 0.1]'
                --is_soft_label True
            For `HBSNDataset`:
                --data_dir "img/generated"
                --test_data_dir "img/gen2"
                --augment_scale '[0.5, 2]'
                --augment_translate '[0.5, 0.5]'
                --masked_size 64
            For `CocoDataset`:
                --data_dir "coco/train2017"
                --test_data_dir "coco/val2017"
                --annotation_path "coco/annotations/instances_train2017.json"
                --test_annotation_path "coco/annotations/instances_val2017.json"
                --height 256
                --width 256
                --img_ids []
                --cat_ids []
                --connected False
                --single_instance False
                --resize_rate 1.5
                --min_area 500
                --augment_rotation 30
                --augment_scale '[0.8, 1.2]'
                --augment_translate '[0.1, 0.1]'
        
        recorder_config: configuration for the recorder
                --log_base_dir "runs"
                --log_dir ""
                --comment ""
        
        run_config: dict, configuration for the run
                --checkpoint_path ""
                --version 0.0
                --total_epoches 1000
                --batch_size 64
                --weight_norm  1e-5
                --moments  0.9
                --lr 1e-4
                --lr_decay_rate  0.5
                --lr_decay_steps '[50, 100]'
    '''
    fire.Fire({"train": train, "validate": validate})
