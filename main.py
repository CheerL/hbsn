from utils import read_image
from net import ConformalNet
import torch
import scipy.io
# import torch_directml


mat = scipy.io.loadmat('nmap.mat')
imgs = []
f_maps = []
for i in range(1, 6):
    img = read_image(f'img/tp{i}.png', gray=True, binary_threshold=127, noramlize=True, CHW=True)
    C, H, W = img.shape
    img = torch.from_numpy(img).reshape(C, H, W).float()
    imgs.append(img)
    f_map = torch.from_numpy(mat[f'nmap{i}'])
    f_map = f_map.reshape(W, H, 2).transpose(0,1)
    f_maps.append(f_map)
    
img = torch.stack(imgs, dim=0)
f_map = torch.stack(f_maps, dim=0)

alpha = 1.
beta = 1.
total_epochs = 500
lr = 1e-5

train_loss = []
valid_loss = []
train_epochs_loss = []
valid_epochs_loss = []


device = 'cpu'
# device = torch_directml.device()
img = img.to(device)
net = ConformalNet(H, W, alpha, beta, device=device)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)


for epoch in range(total_epochs):
    net.train()
    train_epoch_loss = []

    output = net(img)
    optimizer.zero_grad()
    loss = net.loss(output.double(), img, f_map)
    loss.backward()
    optimizer.step()
    
    train_epoch_loss.append(loss.item())
    train_loss.append(loss.item())
    print(f"epoch={epoch}/{total_epochs}, loss={loss.item()}")
