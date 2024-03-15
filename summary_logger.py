import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter


class HBSNSummary(SummaryWriter):
    def __init__(
        self, config, train_size, test_size,
        log_dir=None, comment="", purge_step=None, 
        max_queue=10, flush_secs=120, filename_suffix=""
    ):
        
        assert 'total_epoches' in config, 'total_epoches is required'
        assert 'batch_size' in config, 'batch_size is required'

        self.config = config

        if not log_dir:
            # config_str = "+".join([
            #     f"{k}={v.replace('/', '.')}" 
            #     if isinstance(v, str) and k == 'data_dir'
            #     else f"{k}={v}"
            #     for k, v 
            #     in self.config.items()
            #     ])
            current_time = datetime.now().strftime("%b%d_%H-%M-%S")
            log_dir = os.path.join(
                "runs", current_time
            )
            if comment:
                log_dir += f"_{comment}"

        super().__init__(log_dir, comment, purge_step, max_queue, flush_secs, filename_suffix)
        
        self.total_epoches = config['total_epoches']
        self.batch_size = config['batch_size']
        self.train_size = train_size
        self.test_size = test_size

        self.train_loss = torch.zeros(self.train_size)
        self.test_loss = torch.zeros(self.test_size)
        

    def init_logger(self):
        logger.add(os.path.join(self.log_dir, 'log.log'), level="INFO")
        config_info = '\n\t\t'.join([f'{k}: {v}' for k, v in self.config.items()])
        logger.info(f'''
        Start training with config:
        {config_info}
        ''')
        
    def init_summary(self, net):
        empty_input = torch.zeros((1, net.input_channels, net.height, net.width), requires_grad=False).to(net.device, dtype=net.dtype)
        self.add_graph(net, empty_input)
        fig = plt.figure(figsize=(6, 2.7), dpi=100)
        plt.text(
            0.5, 0.5,
            '\n'.join([f'{k}: {v}' for k, v in self.config.items()]),
            ha='center', va='center', multialignment='left', fontsize=12
            )
        # plt.title('Config')
        plt.axis('off')
        self.add_figure('config', fig)
        self.flush()
    
    def add_loss(self, epoch, iteration, loss, is_train=True):
        loss = loss.item()

        if is_train:
            prefix = "Train"
            size = self.train_size
            self.train_loss[iteration] = loss
            logger_func = logger.info
        else:
            prefix = "Test"
            size = self.test_size
            self.test_loss[iteration] = loss
            logger_func = logger.warning
            
        total_iteration = epoch * size + iteration
        self.add_scalar(f"loss/{prefix}", loss, total_iteration)
        self.flush()
        
        logger_func(
            f"{prefix}: epoch={epoch}/{self.total_epoches}, iteration={iteration}/{size}, loss={loss}"
        )
        
    def add_epoch_loss(self, epoch, is_train=True):
        if is_train:
            prefix = "Train"
            loss = self.train_loss.mean().item()
            logger_func = logger.info
        else:
            prefix = "Test"
            loss = self.test_loss.mean().item()
            logger_func = logger.warning

        self.add_scalar(f"loss/{prefix}_epoch", loss, epoch)
        self.flush()

        logger_func(
            f"{prefix} epoch total: epoch={epoch}/{self.total_epoches}, total loss={loss}"
        )
        return loss
        
    def add_output(self, epoch, iteration, img, hbs, output, is_train=True, m=10):
        if is_train:
            prefix = "Train"
            size = self.train_size
        else:
            prefix = "Test"
            size = self.test_size
        total_iteration = epoch * size + iteration
        # k = np.random.randint(0, img.shape[0])
        
        # img_k = img[k].detach().cpu().numpy()
        # output_k = output[k].detach().cpu().numpy()
        # hbs_k = hbs[k].detach().cpu().numpy()
        
        # fig = plt.figure(figsize=(15, 5))
        # plt.subplot(131)
        # plt.imshow(img_k[0], cmap='gray')
        # plt.title("Input")
        # plt.subplot(132)
        # plt.imshow(np.linalg.norm(hbs_k, axis=0), cmap='jet')
        # plt.title("HBS")
        # plt.subplot(133)
        # plt.imshow(np.linalg.norm(output_k, axis=0), cmap='jet')
        # plt.title("Output")

        k = np.random.choice(self.batch_size, m, replace=False)

        img_k = img[k].detach().cpu().numpy()
        output_k = output[k].detach().cpu().numpy()
        hbs_k = hbs[k].detach().cpu().numpy()

        subfigure_size = 2
        fig = plt.figure(figsize=(m*subfigure_size, 3*subfigure_size))
        fig.subplots_adjust(hspace=0.05, wspace=0.05)
        for i in range(m):
            plt.subplot(3, m, i+1)
            plt.imshow(img_k[i,0], cmap='gray')
            plt.axis('off')

            plt.subplot(3, m, m+i+1)
            plt.imshow(np.linalg.norm(hbs_k[i], axis=0), cmap='jet')
            plt.axis('off')

            plt.subplot(3, m, 2*m+i+1)
            plt.imshow(np.linalg.norm(output_k[i], axis=0), cmap='jet')
            plt.axis('off')

        # self.add_image(f"{prefix}/input", img_k, total_iteration)
        # self.add_figure(f"{prefix}/hbs", plot_mu(hbs_k), total_iteration)
        self.add_figure(f"result/{prefix}", fig, total_iteration)
        self.flush()