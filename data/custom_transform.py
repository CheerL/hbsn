
import random
from typing import Any, Dict, List

import torch
from torch import Tensor
from torchvision.datapoints import Mask
# from torchvision.transforms import functional as F
from torchvision.transforms import v2 as transforms
from torchvision.transforms.v2 import functional as F
from torchvision.transforms.v2.utils import query_spatial_size


class BoundedRandomAffine(transforms.RandomAffine):
    def forward(self, img):
        fill = self.fill
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            else:
                fill = [float(f) for f in fill]

        x, y = torch.where(img[0]>0.5)
        dis = ((x-width/2).pow(2) + (y-height/2).pow(2)).sqrt()
        max_dis = dis.max()
        
        
        angle = float(torch.empty(1).uniform_(float(self.degrees[0]), float(self.degrees[1])).item())
        
        if self.scale is not None:
            max_scale = min(self.scale[1], min(width,height) / (max_dis * 2))
            min_scale = min(self.scale[0], max_scale)
            scale_ranges = [min_scale, max_scale]
            scale = float(torch.empty(1).uniform_(scale_ranges[0], scale_ranges[1]).item())
        else:
            scale = 1.0

        if self.translate is not None:
            scaled_max_dis = scale * max_dis
            translate = self.translate
            max_dx = min(float(translate[0] * width), width/2 - scaled_max_dis)
            max_dy = min(float(translate[1] * height), height/2 - scaled_max_dis)
            tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
            ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
            translations = (tx, ty)
        else:
            translations = (0, 0)

        shear_x = shear_y = 0.0
        if self.shear is not None:
            shears = self.shear
            shear_x = float(torch.empty(1).uniform_(shears[0], shears[1]).item())
            if len(shears) == 4:
                shear_y = float(torch.empty(1).uniform_(shears[2], shears[3]).item())
        shear = (shear_x, shear_y)
        
        ret = [angle, translations, scale, shear]
        return F.affine(img, *ret, interpolation=self.interpolation, fill=fill, center=self.center)
    
class SoftLabel(transforms.Transform):
    def __init__(self, kernel_size=3, sigma=1, noise_rate=0.05):
        super().__init__()
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.noise_rate = noise_rate

    def rescale(self, x, new_min, new_max):
        x = x - x.min()
        x = x / x.max()
        x = x * (new_max - new_min) + new_min
        return x
    
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        new_min = torch.randn(1) * 0.1
        if new_min < 0:
            return inpt
        new_max = 1+torch.randn(1) * 0.1
        mask = inpt > 0.5
        noise = torch.randn_like(inpt)
 
        x = inpt + noise * self.noise_rate
        x = F.gaussian_blur(x, kernel_size=self.kernel_size, sigma=self.sigma)
        x[mask] = self.rescale(x[mask], 0.5, new_max)
        x[~mask] = self.rescale(x[~mask], new_min, 0.5)
        return x

class BoundedRandomCrop(transforms.RandomCrop):
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        cropped_height, cropped_width = self.size
        
        assert len(flat_inputs) == 2, "Expected input to be a tuple of (image, mask)"
        image, mask = flat_inputs
        assert isinstance(image, torch.Tensor), f"Expected image to be a tensor, but got {type(image)}"
        assert isinstance(mask, Mask), f"Expected mask to be a Mask object, but got {type(mask)}"

        _, mask_y, mask_x = torch.where(mask > 0.5)
        mask_width_min = mask_x.min()
        mask_width_max = mask_x.max()
        mask_height_min = mask_y.min()
        mask_height_max = mask_y.max()

        
        mask_width = mask_width_max - mask_width_min
        mask_height = mask_height_max - mask_height_min
        if mask_height > cropped_height or mask_width > cropped_width:
            rate = min(
                cropped_width / mask_width * 0.9, 
                cropped_height / mask_height * 0.9
            )
            ori_height, ori_width = query_spatial_size(flat_inputs)
            resized_height = int(ori_height * rate)
            resized_width = int(ori_width * rate)
            needs_resize = True
            resize_size = (resized_height, resized_width)
            
            flat_inputs = [
                F.resize(x, size=(resized_height, resized_width), antialias=True) 
                for x in flat_inputs
            ]
            image, mask = flat_inputs
            
            _, mask_y, mask_x = torch.where(mask > 0.5)
            mask_width_min = mask_x.min()
            mask_width_max = mask_x.max()
            mask_height_min = mask_y.min()
            mask_height_max = mask_y.max()
        else:
            needs_resize = False
            resize_size = None
            
        padded_height, padded_width = query_spatial_size(flat_inputs)
        if self.padding is not None:
            pad_left, pad_right, pad_top, pad_bottom = self.padding
            padded_height += pad_top + pad_bottom
            padded_width += pad_left + pad_right
        else:
            pad_left = pad_right = pad_top = pad_bottom = 0

        if self.pad_if_needed:
            if padded_height < cropped_height:
                diff = cropped_height - padded_height

                pad_top += diff
                pad_bottom += diff
                padded_height += 2 * diff

            if padded_width < cropped_width:
                diff = cropped_width - padded_width

                pad_left += diff
                pad_right += diff
                padded_width += 2 * diff

        if padded_height < cropped_height or padded_width < cropped_width:
            raise ValueError(
                f"Required crop size {(cropped_height, cropped_width)} is larger than "
                f"{'padded ' if self.padding is not None else ''}input image size {(padded_height, padded_width)}."
            )

        # We need a different order here than we have in self.padding since this padding will be parsed again in `F.pad`
        padding = [pad_left, pad_top, pad_right, pad_bottom]
        needs_pad = any(padding)
        needs_vert_crop, top = (
            (True, int(torch.randint(
                max(0, mask_height_max - cropped_height + pad_top), 
                min(padded_height - cropped_height + 1, mask_height_min + pad_top + 1), 
                size=()
            )))
            if padded_height > cropped_height
            else (False, 0)
        )
        needs_horz_crop, left = (
            (True, int(torch.randint(
                max(0, mask_width_max - cropped_width + pad_left), 
                min(padded_width - cropped_width + 1, mask_width_min + pad_left + 1), 
                size=()
            )))
            if padded_width > cropped_width
            else (False, 0)
        )
        return dict(
            needs_crop=needs_vert_crop or needs_horz_crop,
            top=top,
            left=left,
            height=cropped_height,
            width=cropped_width,
            needs_pad=needs_pad,
            padding=padding,
            needs_resize=needs_resize,
            resize_size=resize_size
        )
        
    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if params["needs_resize"]:
            inpt = F.resize(inpt, size=params["resize_size"], antialias=True)

        if params["needs_pad"]:
            fill = self._fill[type(inpt)]
            inpt = F.pad(inpt, padding=params["padding"], fill=fill, padding_mode=self.padding_mode)

        if params["needs_crop"]:
            inpt = F.crop(inpt, top=params["top"], left=params["left"], height=params["height"], width=params["width"])

        return inpt

class ResizeMax(transforms.Transform):
    """
    A PyTorch Transform class that resizes an image such that the maximum dimension 
    is equal to a specified size while maintaining the aspect ratio.
    """
    
    def __init__(self, 
                 max_sz:int=256 # The maximum size for any dimension (height or width) of the image.
                ):
        """
        Initialize ResizeMax object with a specified max_sz. 
        """
        # Call to the parent class (Transform) constructor
        super().__init__()

        # Set the maximum size for any dimension of the image
        self.max_sz = max_sz
        
    def _transform(self, 
                   inpt: Any, # The input image tensor to be resized.
                   params: Dict[str, Any] # A dictionary of parameters. Not used in this method but is present for compatibility with the parent's method signature.
                  ) -> torch.Tensor: # The resized image tensor.
        """
        Apply the ResizeMax transformation on an input image tensor.
        """

        # Copy the input tensor to a new variable
        x = inpt

        # Get the width and height of the image tensor
        height, width = x.shape[-2:]

        # Calculate the size for the smaller dimension, such that the aspect ratio 
        # of the image is maintained when the larger dimension is resized to max_sz
        size = int(min(height, width) / (max(height, width) / self.max_sz))

        # Resize the image tensor with antialiasing for smoother output
        x = F.resize(x, size=size, antialias=True)

        # Return the transformed (resized) image tensor
        return x

class PadSquare(transforms.Transform):
    """
    PadSquare is a PyTorch Transform class used to pad images to make them square. 
    Depending on the configuration, padding can be applied equally on both sides, 
    or can be randomly split between the two sides.
    """

    def __init__(self, 
                 padding_mode:str='constant', # The method to use for padding. Default is 'constant'.
                 fill:tuple=(123, 117, 104), # The RGB values to use for padding if padding_mode is 'constant'.
                 shift:bool=True # If True, padding is randomly split between the two sides. If False, padding is equally applied.
                ):
        """
        The constructor for PadSquare class.
        """
        super().__init__()
        self.padding_mode = padding_mode
        self.fill = fill
        self.shift = shift
        self.pad_split = None

    def forward(self, 
                *inputs: Any # The inputs to the forward method.
               ) -> Any: # The result of the superclass forward method.
        """
        The forward method that sets up the padding split factor if 'shift' is True, 
        and then calls the superclass forward method.
        """
        self.pad_split = random.random() if self.shift else None
        return super().forward(*inputs)

    def _transform(self, 
                   inpt: Any, # The input to be transformed.
                   params: Dict[str, Any] # A dictionary of parameters for the transformation.
                  ) -> Any: # The transformed input.
        """
        The _transform method that applies padding to the input to make it square.
        """
        x = inpt
        
        # Get the width and height of the image tensor
        h, w = x.shape[-2:]
        
        # If shift is true, padding is randomly split between two sides
        if self.shift:
            offset = (max(w, h) - min(w, h))
            pad_1 = int(offset*self.pad_split)
            pad_2 = offset - pad_1
            
            # The padding is applied to the shorter dimension of the image
            self.padding = [0, pad_1, 0, pad_2] if h < w else [pad_1, 0, pad_2, 0]
            padding = self.padding
        else:
            # If shift is false, padding is equally split between two sides
            offset = (max(w, h) - min(w, h)) // 2
            padding = [0, offset] if h < w else [offset, 0]
        
        # Apply the padding to the image
        x = F.pad(x, padding=padding, padding_mode=self.padding_mode, fill=self.fill)
        
        return x

