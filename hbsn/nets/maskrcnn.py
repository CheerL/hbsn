"""MaskRCNN(ResNet50-FPN) + weight_layer/mask_conv。

手抄了 torchvision 0.28 的内部推理链（rpn/roi_heads/postprocess）——
升级 torchvision 前必须核对，state_dict 键：model.* / weight_layer.* / mask_conv.*。
"""

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn,
)
from torchvision.models.detection.roi_heads import (
    keypointrcnn_inference,
    maskrcnn_inference,
)
from torchvision.models.detection.rpn import concat_box_prediction_layers
from torchvision.models.detection.transform import (
    paste_masks_in_image,
    resize_boxes,
    resize_keypoints,
)

from hbsn.nets.base import torch_dtype
from hbsn.nets.hbsn import HBSNet
from hbsn.nets.segmentation import SegHBSNNet


class MaskRCNN(SegHBSNNet):
    def __init__(self, hbsn: HBSNet, config):
        super().__init__(hbsn, config)

    def build_model(self):
        self.model = maskrcnn_resnet50_fpn(
            weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        self.weight_layer = nn.Sequential(
            nn.Linear(2, self.config.weight_hidden_size),
            nn.ReLU(),
            nn.Linear(self.config.weight_hidden_size, 1),
            nn.Softmax(dim=1),
        )
        self.mask_conv = nn.Sequential(
            nn.Conv2d(
                self.config.select_num,
                self.config.output_channels,
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )

    def model_forward(self, images):
        images, _ = self.model.transform(images, None)

        features = self.model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        proposals = self.rpn(images, features)
        detections = self.roi_heads(features, proposals, images.image_sizes)
        detections = self.postprocess(detections, images.image_sizes)

        masks = [
            x["masks"][: self.config.select_num].squeeze(1) for x in detections
        ]
        masks = [
            F.pad(x, (0, 0, 0, 0, 0, self.config.select_num - x.shape[0]))
            for x in masks
        ]
        masks = torch.stack(masks)
        masks = masks.to(self.config.device, dtype=torch_dtype(self.config))

        weight = [
            torch.stack([x["labels"].float(), x["scores"]], dim=1)[
                : self.config.select_num
            ]
            for x in detections
        ]
        weight = [
            F.pad(x, (0, 0, 0, self.config.select_num - x.shape[0]))
            for x in weight
        ]
        weight = torch.stack(weight)
        weight = weight.to(self.config.device, dtype=torch_dtype(self.config))

        weight = self.weight_layer(weight)
        masks = self.mask_conv(masks * weight.unsqueeze(3))
        return masks

    def rpn(
        self, images, features: dict[str, torch.Tensor]
    ) -> list[torch.Tensor]:
        features = list(features.values())
        objectness, pred_bbox_deltas = self.model.rpn.head(features)
        anchors = self.model.rpn.anchor_generator(images, features)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [
            s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors
        ]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(
            objectness, pred_bbox_deltas
        )
        proposals = self.model.rpn.box_coder.decode(
            pred_bbox_deltas.detach(), anchors
        )
        proposals = proposals.view(num_images, -1, 4)
        boxes, _ = self.model.rpn.filter_proposals(
            proposals,
            objectness,
            images.image_sizes,
            num_anchors_per_level,
        )
        return boxes

    def roi_heads(
        self,
        features: dict[str, torch.Tensor],
        proposals: list[torch.Tensor],
        image_shapes: list[tuple[int, int]],
    ):
        box_features = self.model.roi_heads.box_roi_pool(
            features, proposals, image_shapes
        )
        box_features = self.model.roi_heads.box_head(box_features)
        class_logits, box_regression = self.model.roi_heads.box_predictor(
            box_features
        )

        result: list[dict[str, torch.Tensor]] = []
        boxes, scores, labels = self.model.roi_heads.postprocess_detections(
            class_logits, box_regression, proposals, image_shapes
        )
        for i in range(len(boxes)):
            result.append(
                {"boxes": boxes[i], "labels": labels[i], "scores": scores[i]}
            )

        if self.model.roi_heads.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            if self.model.roi_heads.mask_roi_pool is not None:
                mask_features = self.model.roi_heads.mask_roi_pool(
                    features, mask_proposals, image_shapes
                )
                mask_features = self.model.roi_heads.mask_head(mask_features)
                mask_logits = self.model.roi_heads.mask_predictor(mask_features)
            else:
                raise ValueError("Expected mask_roi_pool to be not None")

            labels = [r["labels"] for r in result]
            masks_probs = maskrcnn_inference(mask_logits, labels)
            for mask_prob, r in zip(masks_probs, result, strict=True):
                r["masks"] = mask_prob

        if (
            self.model.roi_heads.keypoint_roi_pool is not None
            and self.model.roi_heads.keypoint_head is not None
            and self.model.roi_heads.keypoint_predictor is not None
        ):
            keypoint_proposals = [p["boxes"] for p in result]
            keypoint_features = self.model.roi_heads.keypoint_roi_pool(
                features, keypoint_proposals, image_shapes
            )
            keypoint_features = self.model.roi_heads.keypoint_head(
                keypoint_features
            )
            keypoint_logits = self.model.roi_heads.keypoint_predictor(
                keypoint_features
            )

            if keypoint_logits is None or keypoint_proposals is None:
                raise ValueError(
                    "both keypoint_logits and keypoint_proposals should not be None when not in training mode"
                )

            keypoints_probs, kp_scores = keypointrcnn_inference(
                keypoint_logits, keypoint_proposals
            )
            for keypoint_prob, kps, r in zip(
                keypoints_probs, kp_scores, result, strict=True
            ):
                r["keypoints"] = keypoint_prob
                r["keypoints_scores"] = kps

        return result

    def postprocess(
        self,
        result: list[dict[str, torch.Tensor]],
        image_shapes: list[tuple[int, int]],
    ) -> list[dict[str, torch.Tensor]]:
        for i, (pred, im_s) in enumerate(
            zip(result, image_shapes, strict=True)
        ):
            o_im_s = (self.config.height, self.config.width)
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
        return nn.ModuleList([super().fixable_layers, self.model])
