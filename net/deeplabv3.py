from torchvision.models.segmentation import (DeepLabV3_ResNet50_Weights,
                                             deeplabv3_resnet50)

model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
