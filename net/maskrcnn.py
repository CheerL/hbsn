from typing import Dict, List, Optional, OrderedDict, Tuple

import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.detection import (MaskRCNN_ResNet50_FPN_Weights,
                                          maskrcnn_resnet50_fpn)
from torchvision.models.detection.roi_heads import (keypointrcnn_inference,
                                                    maskrcnn_inference)
from torchvision.models.detection.rpn import concat_box_prediction_layers
from torchvision.models.detection.transform import (paste_masks_in_image,
                                                    resize_boxes,
                                                    resize_keypoints)

from net.base_net import BaseNet
from net.hbsn import HBSNet
from net.seg_hbsn_net import SegHBSNNet

DTYPE = torch.float32


class MaskRCNN(SegHBSNNet):
    def __init__(
        self, height=256, width=256, input_channels=3, output_channels=1, 
        select_num=10, weight_hidden_size=20, 
        dice_rate=0.1, iou_rate=0, hbs_loss_rate=1, mask_scale=100,
        hbsn_checkpoint='',
        hbsn_channels=[64, 128, 256, 512], hbsn_radius=50,
        hbsn_stn_mode=0, hbsn_stn_rate=0.0,
        dtype=DTYPE, device="cpu"):
        super().__init__(
            height, width, input_channels, output_channels,
            dice_rate, iou_rate, hbs_loss_rate, mask_scale,
            hbsn_checkpoint, hbsn_channels, hbsn_radius, hbsn_stn_mode, hbsn_stn_rate,
            dtype, device
        )
        self.select_num = select_num
        self.weight_hidden_size = weight_hidden_size
        
        self.model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        
        self.weight_layer = nn.Sequential(
            nn.Linear(2, self.weight_hidden_size),
            nn.ReLU(),
            nn.Linear(self.weight_hidden_size, 1),
            nn.Softmax(dim=1)
        )
        self.mask_conv = nn.Sequential(
            nn.Conv2d(self.select_num, self.output_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.to(device)
        
    
    def model_forward(self, images):
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        images, _ = self.model.transform(images, None)

        features = self.model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        
        proposals = self.rpn(images, features)
        detections = self.roi_heads(features, proposals, images.image_sizes)
        detections = self.postprocess(detections, images.image_sizes)

        masks = [x["masks"][:self.select_num].squeeze(1) for x in detections]
        masks = [F.pad(x, (0, 0, 0, 0, 0, self.select_num - x.shape[0])) for x in masks]
        masks = torch.stack(masks)
        if str(masks.device) != str(self.device):
            masks = masks.to(self.device, dtype=self.dtype)

        weight = [torch.stack([x["labels"].float(), x["scores"]], dim=1)[:self.select_num] for x in detections]
        weight = [F.pad(x, (0, 0, 0, self.select_num - x.shape[0])) for x in weight]
        weight = torch.stack(weight)
        if str(weight.device) != str(self.device):
            weight = weight.to(self.device, dtype=self.dtype)
            
        weight = self.weight_layer(weight)
        masks = self.mask_conv(masks * weight.unsqueeze(3))
        # masks = get_mask(masks-0.5, eps=1/scale)
        return masks


    def rpn(
        self,
        images,
        features: Dict[str, torch.Tensor]
    ) -> List[torch.Tensor]:

        """
        Args:
            images (ImageList): images for which we want to compute the predictions
            features (Dict[str, Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (List[Dict[str, Tensor]]): ground-truth boxes present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes.

        Returns:
            boxes (List[Tensor]): the predicted boxes from the RPN, one Tensor per
                image.
            losses (Dict[str, Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        # RPN uses all feature maps that are available
        features = list(features.values())
        objectness, pred_bbox_deltas = self.model.rpn.head(features)
        anchors = self.model.rpn.anchor_generator(images, features)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.model.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, _ = self.model.rpn.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        return boxes
    
    def roi_heads(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[torch.Tensor],
        image_shapes: List[Tuple[int, int]]
    ):
        """
        Args:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        box_features = self.model.roi_heads.box_roi_pool(features, proposals, image_shapes)
        box_features = self.model.roi_heads.box_head(box_features)
        class_logits, box_regression = self.model.roi_heads.box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        
        boxes, scores, labels = self.model.roi_heads.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
        num_images = len(boxes)
        for i in range(num_images):
            result.append(
                {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                }
            )

        if self.model.roi_heads.has_mask():
            mask_proposals = [p["boxes"] for p in result]

            if self.model.roi_heads.mask_roi_pool is not None:
                mask_features = self.model.roi_heads.mask_roi_pool(features, mask_proposals, image_shapes)
                mask_features = self.model.roi_heads.mask_head(mask_features)
                mask_logits = self.model.roi_heads.mask_predictor(mask_features)
            else:
                raise Exception("Expected mask_roi_pool to be not None")

            labels = [r["labels"] for r in result]
            masks_probs = maskrcnn_inference(mask_logits, labels)
            for mask_prob, r in zip(masks_probs, result):
                r["masks"] = mask_prob

        # keep none checks in if conditional so torchscript will conditionally
        # compile each branch
        if (
            self.model.roi_heads.keypoint_roi_pool is not None
            and self.model.roi_heads.keypoint_head is not None
            and self.model.roi_heads.keypoint_predictor is not None
        ):
            keypoint_proposals = [p["boxes"] for p in result]

            keypoint_features = self.model.roi_heads.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
            keypoint_features = self.model.roi_heads.keypoint_head(keypoint_features)
            keypoint_logits = self.model.roi_heads.keypoint_predictor(keypoint_features)

            if keypoint_logits is None or keypoint_proposals is None:
                raise ValueError(
                    "both keypoint_logits and keypoint_proposals should not be None when not in training mode"
                )

            keypoints_probs, kp_scores = keypointrcnn_inference(keypoint_logits, keypoint_proposals)
            for keypoint_prob, kps, r in zip(keypoints_probs, kp_scores, result):
                r["keypoints"] = keypoint_prob
                r["keypoints_scores"] = kps

        return result

    def postprocess(
        self,
        result: List[Dict[str, torch.Tensor]],
        image_shapes: List[Tuple[int, int]]
    ) -> List[Dict[str, torch.Tensor]]:
        for i, (pred, im_s) in enumerate(zip(result, image_shapes)):
            o_im_s = (self.height, self.width)
            boxes = pred["boxes"]
            boxes = resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
            if "masks" in pred:
                masks = pred["masks"]
                masks = paste_masks_in_image(masks, boxes, o_im_s)
                result[i]["masks"] = masks
            if "keypoints" in pred:
                keypoints = pred["keypoints"]
                keypoints = resize_keypoints(keypoints, im_s, o_im_s)
                result[i]["keypoints"] = keypoints
        return result

    @property
    def fixable_layers(self):
        return nn.ModuleList([
            super().fixable_layers,
            self.model
        ])