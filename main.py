from utils import read_image
from net import ConformalNet, check_inside_unit_disk
import torch
import torch.nn as nn
import scipy.io
import matplotlib.pyplot as plt

# import torch_directml

from torch.utils.tensorboard import SummaryWriter


k = 1.0
alpha = 1.0
beta = 0.01
total_epochs = 50000
lr = 5e-5
device = "cuda:0"

DTYPE = torch.float32
IMAGE_INTERVAL = 50


def get_data():
    mat = scipy.io.loadmat("nmap.mat")
    imgs = []
    f_maps = []
    for i in range(5):
        img = read_image(
            f"img/tp{i+1}.png",
            gray=True,
            binary_threshold=127,
            noramlize=True,
            CHW=True,
        )
        C, H, W = img.shape
        img = torch.from_numpy(img).reshape(C, H, W)
        imgs.append(img)
        f_map = torch.from_numpy(mat[f"nmap{i+1}"])
        f_map = f_map.reshape(W, H, 2).transpose(0, 1)
        f_maps.append(f_map)

    img = torch.stack(imgs, dim=0).type(DTYPE)
    f_map = torch.stack(f_maps, dim=0).type(DTYPE)
    return img, f_map


def plot_scatter(x):
    plt.scatter(x[:, :, 0], x[:, :, 1], s=0.5)
    return plt.gcf()


def init_summary(log_writer: SummaryWriter, img, f_map, net, k=0):
    label_k = f_map[k].detach().cpu().numpy()
    log_writer.add_image("train_img/original", img[k])
    log_writer.add_figure("train_map/label", plot_scatter(label_k))
    log_writer.add_graph(net, img[k : k + 1])
    log_writer.flush()


def summary_loss(log_writer: SummaryWriter, epoch, loss):
    # log_writer.add_scalars("train_loss/loss", loss, epoch)
    log_writer.add_scalar("train_loss/total", loss['total_loss'], epoch)
    log_writer.add_scalar("train_loss/img", loss['img_loss'], epoch)
    log_writer.add_scalar("train_loss/mu", loss['mu_loss'], epoch)
    log_writer.add_scalar("train_loss/label", loss['label_loss'], epoch)
    log_writer.add_scalar("train_loss/area", loss['area_loss'], epoch)
    log_writer.flush()


def summary_output(log_writer: SummaryWriter, epoch, output, k=0):
    output_map_k = output[k].detach().cpu().numpy()
    log_writer.add_figure("train_map/output", plot_scatter(output_map_k), epoch)
    is_inside = check_inside_unit_disk(output[k:k+1])
    log_writer.add_image("train_img/inside_unit", is_inside, epoch)
    log_writer.flush()


def main():
    img, f_map = get_data()
    img = img[0:1]
    f_map = f_map[0:1]
    N, C, H, W = img.shape
    img = img.to(device)
    f_map = f_map.to(device)
    net = ConformalNet(H, W, k, alpha, beta, device=device, dtype=DTYPE)
    net.initialize()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    log_writer = SummaryWriter()
    init_summary(log_writer, img, f_map, net)

    train_loss = []
    valid_loss = []

    ## Training
    for epoch in range(total_epochs):
        net.train()
        output = net(img)
        optimizer.zero_grad()
        total_loss, img_loss, mu_loss, label_loss, area_loss = net.loss(output, img, f_map)
        total_loss.backward()
        optimizer.step()

        loss = {
            "total_loss": total_loss.item(),
            "img_loss": img_loss.item(),
            "mu_loss": mu_loss.item(),
            "label_loss": label_loss.item(),
            "area_loss": area_loss.item(),
        }
        train_loss.append(loss)

        summary_loss(log_writer, epoch, loss)
        if not epoch % IMAGE_INTERVAL:
            summary_output(log_writer, epoch, output)
            
        # if not epoch % 50:
        #     print(output)

        print(
            f"epoch={epoch}/{total_epochs}, total_loss={loss['total_loss']}, img_loss={loss['img_loss']}, mu_loss={loss['mu_loss']}, label_loss={loss['label_loss']}, area_loss={loss['area_loss']}"
        )


if __name__ == "__main__":
    main()
