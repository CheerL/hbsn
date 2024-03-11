import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter


class HBSNLogger(SummaryWriter):
    def __init__(
        self, total_epoches, train_size, test_size, batch_size,
        log_dir=None, comment="", purge_step=None, max_queue=10, flush_secs=120, filename_suffix=""
    ):
        super().__init__(log_dir, comment, purge_step, max_queue, flush_secs, filename_suffix)
        self.total_epoches = total_epoches
        self.train_size = train_size
        self.test_size = test_size
        self.batch_size = batch_size
        
        self.train_loss = []
        self.test_loss = []
        
        
    def init_summary(self, net):
        empty_input = torch.zeros((1, net.input_channels, net.height, net.width), requires_grad=False).to(net.device, dtype=net.dtype)
        self.add_graph(net, empty_input)
        self.flush()
    
    def add_loss(self, epoch, iteration, loss, is_train=True):
        loss = loss.item()

        if is_train:
            prefix = "Train"
            size = self.train_size
            self.train_loss.append(loss)
            total_iteration = len(self.train_loss)
            logger_func = logger.info
        else:
            prefix = "Test"
            size = self.test_size
            self.test_loss.append(loss)
            total_iteration = len(self.test_loss)
            logger_func = logger.warning

        self.add_scalar(f"loss/{prefix}", loss, total_iteration)
        self.flush()
        
        logger_func(
            f"{prefix}: epoch={epoch}/{self.total_epoches}, iteration={iteration}/{size}, loss={loss}"
        )
        
    def add_output(self, img, hbs, output, is_train=True):
        if is_train:
            prefix = "Train"
            total_iteration = len(self.train_loss)
        else:
            prefix = "Test"
            total_iteration = len(self.test_loss)
        
        k = np.random.randint(0, self.batch_size)
        
        img_k = img[k].detach().cpu().numpy()
        output_k = output[k].detach().cpu().numpy()
        hbs_k = hbs[k].detach().cpu().numpy()
        
        fig = plt.figure()
        plt.subplot(131)
        plt.imshow(img_k[0], cmap='gray')
        plt.title("Input")
        plt.subplot(132)
        plt.imshow(np.linalg.norm(hbs_k, axis=0), cmap='jet')
        plt.title("HBS")
        plt.subplot(133)
        plt.imshow(np.linalg.norm(output_k, axis=0), cmap='jet')
        plt.title("Output")

        
        # self.add_image(f"{prefix}/input", img_k, total_iteration)
        # self.add_figure(f"{prefix}/hbs", plot_mu(hbs_k), total_iteration)
        self.add_figure(f"result/{prefix}", fig, total_iteration)
        self.flush()