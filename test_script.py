#!/usr/bin/env python
# coding: utf-8

# In[1]:


from factory import config_factory, net_factory, dataset_factory, recorder_factory
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

cv2.setNumThreads(0)


config_dict = {
    'device': 'cuda:2',
    # 'hbsn_checkpoint': 'runs/hbsn/Jun11_13-01-53_big_ns/checkpoints/best.pth',
    'connected': True,
    'single_instance': True,
    'batch_size': 1,
    'pin_memory': False,
    # 'hbsn_loss_rate': 0.05,
    # 'dice_loss_rate': 1,
    # 'mse_loss_rate': 0.05,
    # 'iou_loss_rate': 0.05,
    'cat_ids': [],
    'data_dir': 'coco/val2017',
    'annotation_path': 'coco/annotations/instances_val2017.json',
    'test_data_dir': 'coco/val2017',
    'test_annotation_path': 'coco/annotations/instances_val2017.json',
    # 'checkpoint_path': 'runs/unetpp/May17_10-17-34_hbs0.05_all_c/checkpoints/best.pth',
}

type_ = 'hbsnseg'

config = config_factory(type_, config_dict)
train_dataloader, test_dataloader = dataset_factory(type_, config)
recorder = recorder_factory(
    type_, config, len(train_dataloader), len(test_dataloader)
)
# optimizer, scheduler = initialization(net, recorder, config)
# init_epoch = load_checkpoint(net, recorder, optimizer, scheduler, config)
# net.eval()
# net.initialize()
# net.train()
# torchsummary.summary(net, (1, 256, 256))
def load_net(type_, checkpoint_path, config_dict):
    config = config_factory(type_, {'checkpoint_path': checkpoint_path,**config_dict})
    net = net_factory(type_, config)
    net.load(config.run_config.checkpoint_path)
    net.eval()
    return net

def segnet_evaluate_func(net, input_data, ground_truth):
    predict_mask, predict_hbs = net(input_data)
    predict_mask = net.get_hard_mask(predict_mask)
    f1, iou = net.get_metrics(predict_mask, ground_truth)
    metrics = torch.stack([f1, iou], dim=1)
    return metrics

def test_checkpoint(type_, checkpoint_path, config_dict):
    net = load_net(type_, checkpoint_path, config_dict)
    net.eval()

    results = []
    with torch.no_grad():
        for img, mask in test_dataloader:
            img = img.to(net.config.device, dtype=net.config.dtype)
            mask = mask.to(net.config.device, dtype=net.config.dtype)

            metrics = segnet_evaluate_func(net, img, mask)
            results.append(metrics.cpu().numpy())
            
    results = np.concatenate(results, axis=0)
    return results.mean(axis=0)


# In[ ]:


# unet with hbsn
# runs/unetpp/May17_10-17-34_hbs0.05_all_c/checkpoints/epoch_350.pth [0.78309405 0.71100384]
type_ = 'unetpp'
checkpoint_dir = 'runs/unetpp/May17_10-17-34_hbs0.05_all_c/checkpoints'
for checkpoint in os.listdir(checkpoint_dir):
    if checkpoint.endswith('.pth'):
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
        mean_metrics = test_checkpoint(type_, checkpoint_path, config_dict)
        print(checkpoint_path, mean_metrics)


# In[ ]:


# unet without hbsn
# runs/unetpp/May14_19-52-45_hbs0_all/checkpoints/best.pth [0.779487  0.7047704]
type_ = 'unetpp'
checkpoint_dir = 'runs/unetpp/May14_19-52-45_hbs0_all/checkpoints'
for checkpoint in os.listdir(checkpoint_dir):
    if checkpoint.endswith('.pth'):
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
        mean_metrics = test_checkpoint(type_, checkpoint_path, config_dict)
        print(checkpoint_path, mean_metrics)


# In[ ]:


# deeplab with hbsn
# runs/deeplab/May23_21-02-42_hbs0.05_all/checkpoints/epoch_195.pth [0.78288734 0.704551  ]
type_ = 'deeplab'
checkpoint_dir = 'runs/deeplab/May24_12-58-07_hbs0.05_all_newhbsn/checkpoints'
for checkpoint in os.listdir(checkpoint_dir):
    if checkpoint.endswith('.pth'):
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
        mean_metrics = test_checkpoint(type_, checkpoint_path, config_dict)
        print(checkpoint_path, mean_metrics)


# In[ ]:


# deeplab without hbsn
# runs/deeplab/May15_11-02-15_hbs0_all_c/checkpoints/best.pth [0.7664594 0.6875632]

type_ = 'deeplab'
checkpoint_dir = 'runs/deeplab/May15_11-02-15_hbs0_all_c/checkpoints'
for checkpoint in os.listdir(checkpoint_dir):
    if checkpoint.endswith('.pth'):
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
        mean_metrics = test_checkpoint(type_, checkpoint_path, config_dict)
        print(checkpoint_path, mean_metrics)


# In[ ]:


unet_hbsn = load_net('unetpp', 'runs/unetpp/May17_10-17-34_hbs0.05_all_c/checkpoints/epoch_350.pth', config_dict)
unet = load_net('unetpp', 'runs/unetpp/May14_19-52-45_hbs0_all/checkpoints/best.pth', config_dict)
deeplab_hbsn = load_net('deeplab', 'runs/deeplab/May23_21-02-42_hbs0.05_all/checkpoints/epoch_195.pth', config_dict)
deeplab = load_net('deeplab', 'runs/deeplab/May15_11-02-15_hbs0_all_c/checkpoints/best.pth', config_dict)


# In[ ]:





# In[ ]:


import glob
import os
from PIL import Image
import numpy as np

# Get all image files from img/hbs_seg directory
image_dir = 'img/hbs_seg'
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
image_paths = []

for ext in image_extensions:
    image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    image_paths.extend(glob.glob(os.path.join(image_dir, ext.upper())))

# Filter out images ending with 'g.png'
image_paths = [path for path in image_paths if not path.endswith('g.png')]

print(f"Found {len(image_paths)} images in {image_dir}")

# Process each image
for i, img_path in enumerate(image_paths):
    print(f"Processing {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
    
    try:
        # Load and preprocess image
        img_pil = Image.open(img_path).convert('RGB')
        img_array = np.array(img_pil)
        
        # Resize to model input size (assuming 256x256)
        img_resized = cv2.resize(img_array, (256, 256))
        
        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
        # img_tensor = img_tensor.unsqueeze(0).to(unet.config.device)
        
        # gt = Image.open(img_path[:-4] + 'g.png').convert('L')
        # gt_array = np.array(gt)
        # gt_resized = cv2.resize(gt_array, (256, 256))
        # gt_tensor = torch.from_numpy(gt_resized).float().unsqueeze(0).unsqueeze(0) / 255.0
        # gt_tensor = gt_tensor.to(unet.config.device)
        
        # with torch.no_grad():
            # UNet prediction
            # unet_mask, _ = unet(img_tensor)
            # unet_pred = unet_mask[0,0].cpu().numpy()
            # # unet_pred = unet.get_hard_mask(unet_mask)[0, 0].cpu().numpy()
            
            # # UNet+HBSN prediction
            # unet_hbsn_mask, _ = unet_hbsn(img_tensor)
            # unet_hbsn_pred = unet_hbsn_mask[0,0].cpu().numpy()
            # # unet_hbsn_pred = unet_hbsn.get_hard_mask(unet_hbsn_mask)[0, 0].cpu().numpy()
            
            # # DeepLab prediction
            # deeplab_mask, _ = deeplab(img_tensor)
            # deeplab_pred = deeplab_mask[0,0].cpu().numpy()
            # # deeplab_pred = deeplab.get_hard_mask(deeplab_mask)[0, 0].cpu().numpy()
            
            # # DeepLab+HBSN prediction
            # deeplab_hbsn_mask, _ = deeplab_hbsn(img_tensor)
            # deeplab_hbsn_pred = deeplab_hbsn_mask[0,0].cpu().numpy()
            # # deeplab_hbsn_pred = deeplab_hbsn.get_hard_mask(deeplab_hbsn_mask)[0, 0].cpu().numpy()
        
        # Display results
        fig, axes = plt.subplots(1, 6, figsize=(20, 4))
        fig.suptitle(f'Segmentation Results - {os.path.basename(img_path)}')
        
        axes[0].imshow(img_resized)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(img_resized, cmap='gray')
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        axes[2].imshow(img_resized, cmap='gray')
        axes[2].set_title('UNet')
        axes[2].axis('off')
        
        axes[3].imshow(img_resized, cmap='gray')
        axes[3].set_title('DeepLab')
        axes[3].axis('off')
        
        axes[4].imshow(img_resized, cmap='gray')
        axes[4].set_title('TPSN')
        axes[4].axis('off')
        
        axes[5].imshow(img_resized, cmap='gray')
        axes[5].set_title('HBS Segmentation')
        axes[5].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        continue


# In[ ]:


# Initialize arrays to store evaluation results
num_samples = len(test_dataloader)
print(f"Total samples in test dataloader: {num_samples}")

# Create arrays to store images, masks, and predictions for later analysis
images = []
true_masks = []
unet_preds = []
unet_hbsn_preds = []
deeplab_preds = []
deeplab_hbsn_preds = []

scores = []

for i, (img, mask) in enumerate(test_dataloader):
    img = img.to(unet.config.device, dtype=unet.config.dtype)
    mask = mask.to(unet.config.device, dtype=unet.config.dtype)

    mask1, _ = unet(img)
    mask1 = unet.get_hard_mask(mask1)
    _, iou1 = unet.get_metrics(mask1, mask)


    mask2, _ = unet_hbsn(img)
    mask2 = unet_hbsn.get_hard_mask(mask2)
    _, iou2 = unet_hbsn.get_metrics(mask2, mask)

    mask3, _ = deeplab(img)
    mask3 = deeplab.get_hard_mask(mask3)
    _, iou3 = deeplab.get_metrics(mask3, mask)

    mask4, _ = deeplab_hbsn(img)
    mask4 = deeplab_hbsn.get_hard_mask(mask4)
    _, iou4 = deeplab_hbsn.get_metrics(mask4, mask)
    
    scores.append([iou1.item(), iou2.item(), iou3.item(), iou4.item()])

    # Store images and predictions for later visualization
    images.append(img[0].detach().cpu().numpy())
    true_masks.append(mask[0, 0].detach().cpu().numpy())
    unet_preds.append(mask1[0, 0].detach().cpu().numpy())
    unet_hbsn_preds.append(mask2[0, 0].detach().cpu().numpy())
    deeplab_preds.append(mask3[0, 0].detach().cpu().numpy())
    deeplab_hbsn_preds.append(mask4[0, 0].detach().cpu().numpy())
    
    # Print progress
    if i % 50 == 0:
        print(f"Processed {i}/{num_samples} samples")


# In[ ]:


scores = np.array(scores)
images = np.array(images)
true_masks = np.array(true_masks)
unet_preds = np.array(unet_preds)
unet_hbsn_preds = np.array(unet_hbsn_preds)
deeplab_preds = np.array(deeplab_preds)
deeplab_hbsn_preds = np.array(deeplab_hbsn_preds)
print(f"Scores shape: {scores.shape}")
print(f"Images shape: {images.shape}")
print(f"True masks shape: {true_masks.shape}")
print(f"UNet predictions shape: {unet_preds.shape}")
print(f"UNet HBSN predictions shape: {unet_hbsn_preds.shape}")
print(f"DeepLab predictions shape: {deeplab_preds.shape}")
print(f"DeepLab HBSN predictions shape: {deeplab_hbsn_preds.shape}")


# In[ ]:


# Calculate the improvement metric for each sample
improvements = (scores[:, 1] - scores[:, 0]) + (scores[:, 3] - scores[:, 2])

# Sort scores by improvement in descending order
sorted_indices = np.argsort(-improvements)
sorted_scores = scores[sorted_indices]

# Create a plot showing the improvements
plt.figure(figsize=(12, 6))
plt.scatter(range(len(improvements)), improvements[sorted_indices], alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Sample Index (sorted by improvement)')
plt.ylabel('Total Improvement (UNet HBSN - UNet + DeepLab HBSN - DeepLab)')
plt.title('Model Improvement with HBSN across Samples')
plt.grid(True, alpha=0.3)

# Print top 10 most improved and least improved samples
print("Top 10 most improved samples (indices):", sorted_indices[:10])
print("Top 10 least improved samples (indices):", sorted_indices[-10:])

# Calculate overall statistics
mean_improvement = np.mean(improvements)
median_improvement = np.median(improvements)
positive_improvements = np.sum(improvements > 0)
print(f"Mean improvement: {mean_improvement:.4f}")
print(f"Median improvement: {median_improvement:.4f}")
print(f"Samples with positive improvement: {positive_improvements}/{len(improvements)} ({positive_improvements/len(improvements)*100:.2f}%)")


# In[ ]:


# Displaying top 10 most improved samples based on sorted_indices
# Get the top 10 samples with most improvement
# Display top 10 most improved samples from our saved arrays

count = 0

for i, idx in enumerate(sorted_indices):
    # Calculate IoU values from scores array
    iou1 = scores[idx, 0]  # UNet
    iou2 = scores[idx, 1]  # UNet+HBSN
    iou3 = scores[idx, 2]  # DeepLab
    iou4 = scores[idx, 3]  # DeepLab+HBSN

    if iou2 < 0.65 or iou4 < 0.65:
        continue
    if iou2 - iou1 < 0.01 or iou4 - iou3 < 0.01:
        continue
    
    # Get the data from our saved arrays
    img = images[idx]
    mask = true_masks[idx]
    unet_pred = unet_preds[idx]
    unet_hbsn_pred = unet_hbsn_preds[idx]
    deeplab_pred = deeplab_preds[idx]
    deeplab_hbsn_pred = deeplab_hbsn_preds[idx]
    
    
    
    # Create a figure to display the images and masks
    fig, axs = plt.subplots(1, 6, figsize=(20, 4))
    fig.suptitle(f'Sample #{idx} - Improvement: UNet {(iou2-iou1):.4f}, Deeplab {(iou4-iou3):.4f}')
    
    # Display original image - convert from CHW to HWC format for display
    img_display = np.transpose(img, (1, 2, 0))
    axs[0].imshow(img_display)
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    # Display ground truth
    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title('Ground Truth')
    axs[1].axis('off')
    
    # Display UNet result
    axs[2].imshow(unet_pred, cmap='gray')
    axs[2].set_title(f'UNet (IoU: {iou1:.4f})')
    axs[2].axis('off')
    
    # Display UNet+HBSN result
    axs[3].imshow(unet_hbsn_pred, cmap='gray')
    axs[3].set_title(f'UNet+HBSN (IoU: {iou2:.4f})')
    axs[3].axis('off')
    
    # Display DeepLab result
    axs[4].imshow(deeplab_pred, cmap='gray')
    axs[4].set_title(f'DeepLab (IoU: {iou3:.4f})')
    axs[4].axis('off')
    
    # Display DeepLab+HBSN result
    axs[5].imshow(deeplab_hbsn_pred, cmap='gray')
    axs[5].set_title(f'DeepLab+HBSN (IoU: {iou4:.4f})')
    axs[5].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
    plt.show()
    
    # Print IoU values
    print(f"Sample #{idx} IoU values:")
    print(f"UNet: {iou1:.4f}, UNet+HBSN: {iou2:.4f}")
    print(f"DeepLab: {iou3:.4f}, DeepLab+HBSN: {iou4:.4f}")
    print(f"Improvement (UNet): {iou2 - iou1:.4f}")
    print(f"Improvement (DeepLab): {iou4 - iou3:.4f}")
    print("-" * 50)
    
    count += 1

    # if count >= 50:
    #     break

