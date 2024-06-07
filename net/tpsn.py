import segmentation_models_pytorch as smp
import torch.nn.functional as F

from net.seg_hbsn_net import SegHBSNNet


class TPSN(SegHBSNNet):
    def build_model(self):
        self.model = smp.UnetPlusPlus(encoder_name='resnet50', encoder_weights='imagenet', classes=1, activation='sigmoid')

    def model_forward(self, x):
        x = self.model(x)
        return x
    
    # @property
    # def fixable_layers(self):
    #     return nn.ModuleList([
    #         super().fixable_layers,
    #         self.model.encoder
    #     ])
        
    # @property
    # def uninitializable_layers(self):
    #     return nn.ModuleList([
    #         super().uninitializable_layers,
    #         self.model
    #     ])
        
    def loss(self, predict, ground_truth):
        predict_mask, predict_hbs = predict
        mse_loss = F.mse_loss(predict_mask, ground_truth)
        f1, iou = self.get_metrics(self.binarize_mask(predict_mask), ground_truth)
        f1 = f1.mean()
        iou = iou.mean()
        dice_loss = 1 - f1
        iou_loss = 1 - iou

        ground_truth_hbs = self.hbsn(ground_truth)
        hbs_loss_dict, (_, ground_truth_hbs) = self.hbsn.loss(
            predict_hbs, ground_truth_hbs
        )

        loss = (
            mse_loss
            + self.config.dice_rate * dice_loss
            + self.config.iou_rate * iou_loss
            + self.hbs_loss_rate * hbs_loss_dict["loss"]
        )

        loss_dict = {
            "loss": loss,
            "mse_loss": mse_loss,
            "dice": f1,
            "iou": iou,
            "hbs_loss": hbs_loss_dict["hbs_loss"],
        }
        return loss_dict, (predict_mask, predict_hbs, ground_truth_hbs)
