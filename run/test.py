import torch
from net.base_net import BaseNet
from torch.utils.data import DataLoader

def test(net: BaseNet, dataloader: DataLoader, evaluate_func):
    net.eval()
    with torch.no_grad():
        results = []
        for img, ground_truth in dataloader:
            img = img.to(net.device, dtype=net.dtype)
            ground_truth = ground_truth.to(net.device, dtype=net.dtype)
            output_data = net(img)
            result = evaluate_func(net, output_data, ground_truth)
            results.append(result)
            
    return torch.cat(results).cpu().numpy()