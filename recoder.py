import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from config import BaseConfig, SegNetConfig


def get_random_index(num, size):
    num = min(num, size)
    index = np.random.choice(size, num, replace=False)
    return index, num


class BaseRecoder(SummaryWriter):
    def __init__(
        self, config: BaseConfig, train_size, test_size
    ):
        super().__init__(log_dir=config.log_dir, comment=config.comment)
        self.config = config
        self.total_epoches = config.total_epoches
        self.batch_size = config.batch_size
        self.checkpoint_dir = config.checkpoint_dir
        self.train_size = train_size
        self.test_size = test_size

        self.train_loss = torch.zeros(self.train_size)
        self.test_loss = torch.zeros(self.test_size)
        
        self.best_epoch = -1
        self.best_loss = 1e10
        
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        
        logger.add(os.path.join(self.log_dir, 'log.log'), level="INFO")

    def init_recoder(self, net=None):
        config_str_list = self.config.get_config_str_list()
        
        config_info = '\n\t'.join(config_str_list)
        logger.info(f'''
        Start training with config:
        {config_info}
        ''')
        
        
        max_width = max([len(s) for s in config_str_list]) * 0.1
        max_height = len(config_str_list) * 0.25
        fig = plt.figure(figsize=(max_width, max_height), dpi=100)
        plt.text(
            0.5, 0.5,
            '\n'.join(config_str_list),
            ha='center', va='center', multialignment='left', fontsize=12
            )
        plt.axis('off')
        self.add_figure('config', fig)
        
        try:
            empty_input = torch.rand(net.get_input_shape(), requires_grad=False, device=net.device, dtype=net.dtype)
            self.add_graph(net, empty_input)
        except Exception as e:
            pass
    
    def add_loss(self, epoch, iteration, loss_dict, is_train=True):
        prefix, size, total_iteration, logger_func = self._get_info(epoch, iteration, is_train=is_train)
        
        if is_train:
            self.train_loss[iteration] = loss_dict['loss']
        else:
            self.test_loss[iteration] = loss_dict['loss']

        for loss_name, loss_value in loss_dict.items():
            self.add_scalar(f"{loss_name}/{prefix}", loss_value, total_iteration)

        self.flush()
        
        loss_info = ', '.join([f'{loss_name}={loss_value:.4f}' for loss_name, loss_value in loss_dict.items()])
        logger_func(
            f"{prefix}: epoch={epoch}/{self.total_epoches}, iteration={iteration}/{size}, {loss_info}"
        )
        
    def add_epoch_loss(self, epoch, is_train=True):
        prefix, _, _, logger_func = self._get_info(epoch, is_train=is_train)
        
        if is_train:
            loss = self.train_loss.mean().item()
        else:
            loss = self.test_loss.mean().item()

        self.add_scalar(f"loss/{prefix}_epoch", loss, epoch)
        self.flush()

        logger_func(
            f"{prefix} epoch total: epoch={epoch}/{self.total_epoches}, total loss={loss}"
        )
        
        
    def add_output(self, epoch, iteration, output, is_train=True):
        raise NotImplementedError()
    
    def update_best(self, epoch):
        test_epoch_loss = self.test_loss.mean().item()
        if test_epoch_loss < self.best_loss:
            self.best_loss = test_epoch_loss
            self.best_epoch = epoch
            return True
        return False
    
    def _get_info(self, epoch, iteration=0, is_train=True):
        if is_train:
            prefix = "Train"
            size = self.train_size
            logger_func = logger.info
        else:
            prefix = "Test"
            size = self.test_size
            logger_func = logger.warning

        total_iteration = epoch * size + iteration
        return prefix, size, total_iteration, logger_func


class HBSNRecoder(BaseRecoder):
    def add_output(self, epoch, iteration, input_data, output_data, is_train=True, num=10):
        prefix, _, total_iteration, _ = self._get_info(epoch, iteration, is_train)
        k, num = get_random_index(num, self.batch_size)

        img, hbs = input_data
        img_k = img[k].detach().cpu().numpy()
        output_k = output_data[k].detach().cpu().numpy()
        hbs_k = hbs[k].detach().cpu().numpy()

        subfigure_size = 2
        fig = plt.figure(figsize=(num*subfigure_size, 3*subfigure_size))
        fig.subplots_adjust(hspace=0.05, wspace=0.05)
        for i in range(num):
            plt.subplot(3, num, i+1)
            plt.imshow(img_k[i,0], cmap='gray')
            plt.axis('off')

            plt.subplot(3, num, num+i+1)
            plt.imshow(np.linalg.norm(hbs_k[i], axis=0), cmap='jet')
            plt.axis('off')

            plt.subplot(3, num, 2*num+i+1)
            plt.imshow(np.linalg.norm(output_k[i], axis=0), cmap='jet')
            plt.axis('off')

        self.add_figure(f"result/{prefix}", fig, total_iteration)
        self.flush()


class CocoRecoder(BaseRecoder):
    def add_output(self, epoch, iteration, input_data, output_data, is_train=True, num=10):
        prefix, _, total_iteration, _ = self._get_info(epoch, iteration, is_train)
        k, num = get_random_index(num, self.batch_size)

        img, mask = input_data
        img_k = img[k].detach().cpu().numpy().transpose(0, 2, 3, 1)
        output_k = output_data[k].detach().cpu().numpy()
        mask_k = mask[k].detach().cpu().numpy()

        subfigure_size = 2
        fig = plt.figure(figsize=(num*subfigure_size, 3*subfigure_size))
        fig.subplots_adjust(hspace=0.05, wspace=0.05)
        for i in range(num):
            plt.subplot(3, num, i+1)
            plt.imshow(img_k[i])
            plt.axis('off')

            plt.subplot(3, num, num+i+1)
            plt.imshow(output_k[i, 0], cmap='gray')
            plt.axis('off')

            plt.subplot(3, num, 2*num+i+1)
            plt.imshow(mask_k[i, 0], cmap='gray')
            plt.axis('off')

        self.add_figure(f"result/{prefix}", fig, total_iteration)
        self.flush()
        

class CocoHBSNRecoder(BaseRecoder):
    def __init__(
        self, config: SegNetConfig, train_size, test_size
    ):
        super().__init__(config, train_size, test_size)
        
        self.train_iou = torch.zeros(self.train_size)
        self.test_iou = torch.zeros(self.test_size)
        self.train_dice = torch.zeros(self.train_size)
        self.test_dice = torch.zeros(self.test_size)
    
    def add_output(self, epoch, iteration, input_data, output_data, is_train=True, num=10):
        prefix, _, total_iteration, _ = self._get_info(epoch, iteration, is_train)
        k, num = get_random_index(num, self.batch_size)

        img, mask = input_data
        predict_mask, predict_hbs, hbs = output_data
        
        img_k = img[k].detach().cpu().numpy().transpose(0, 2, 3, 1)
        mask_k = mask[k].detach().cpu().numpy()
        predict_mask_k = predict_mask[k].detach().cpu().numpy()
        hbs_k = hbs[k].detach().cpu().numpy()
        predict_hbs_k = predict_hbs[k].detach().cpu().numpy()

        subfigure_size = 2
        fig = plt.figure(figsize=(num*subfigure_size, 5*subfigure_size))
        fig.subplots_adjust(hspace=0.05, wspace=0.05)
        for i in range(num):
            plt.subplot(5, num, i+1)
            plt.imshow(img_k[i])
            plt.axis('off')

            plt.subplot(5, num, num+i+1)
            plt.imshow(predict_mask_k[i, 0], cmap='gray')
            plt.axis('off')

            plt.subplot(5, num, 2*num+i+1)
            plt.imshow(mask_k[i, 0], cmap='gray')
            plt.axis('off')
            
            plt.subplot(5, num, 3*num+i+1)
            plt.imshow(np.linalg.norm(predict_hbs_k[i], axis=0), cmap='jet')
            plt.axis('off')

            plt.subplot(5, num, 4*num+i+1)
            plt.imshow(np.linalg.norm(hbs_k[i], axis=0), cmap='jet')
            plt.axis('off')

        self.add_figure(f"result/{prefix}", fig, total_iteration)
        self.flush()
        
    def add_loss(self, epoch, iteration, loss_dict, is_train=True):
        super().add_loss(epoch, iteration, loss_dict, is_train)
        
        if is_train:
            self.train_iou[iteration] = loss_dict['iou']
            self.train_dice[iteration] = loss_dict['dice']
        else:
            self.test_iou[iteration] = loss_dict['iou']
            self.test_dice[iteration] = loss_dict['dice']
        self.flush()
        
    def add_epoch_loss(self, epoch, is_train=True):
        super().add_epoch_loss(epoch, is_train)
        prefix, _, _, _ = self._get_info(epoch, is_train=is_train)
        
        if is_train:
            iou = self.train_iou.mean().item()
            dice = self.train_dice.mean().item()
        else:
            iou = self.test_iou.mean().item()
            dice = self.test_dice.mean().item()

        self.add_scalar(f"iou/{prefix}_epoch", iou, epoch)
        self.add_scalar(f"dice/{prefix}_epoch", dice, epoch)
        self.flush()
