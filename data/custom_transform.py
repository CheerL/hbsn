import random
from typing import Any, Dict

import torch
from torch import Tensor
from torch.nn import Module as Transform
from torchvision import transforms
from torchvision.transforms import functional as F


class BoundedRandomAffine(transforms.RandomAffine):
    def forward(self, img):
        fill = float(img[img < 0.5].mean())
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            else:
                fill = [float(f) for f in fill]

        x, y = torch.where(img[0] >= 0.5)
        dis = ((x - width / 2).pow(2) + (y - height / 2).pow(2)).sqrt()
        max_dis = dis.max()

        angle = float(
            torch.empty(1)
            .uniform_(float(self.degrees[0]), float(self.degrees[1]))
            .item()
        )

        if self.scale is not None:
            max_scale = min(self.scale[1], min(width, height) / (max_dis * 2))
            min_scale = min(self.scale[0], max_scale)
            scale_ranges = [min_scale, max_scale]
            scale = float(
                torch.empty(1).uniform_(scale_ranges[0], scale_ranges[1]).item()
            )
        else:
            scale = 1.0

        if self.translate is not None:
            scaled_max_dis = scale * max_dis
            translate = self.translate
            max_dx = min(
                float(translate[0] * width),
                width / 2 - scaled_max_dis,
            )
            max_dy = min(
                float(translate[1] * height),
                height / 2 - scaled_max_dis,
            )
            tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
            ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
            translations = (tx, ty)
        else:
            translations = (0, 0)

        shear_x = shear_y = 0.0
        if self.shear is not None:
            shears = self.shear
            shear_x = float(
                torch.empty(1).uniform_(shears[0], shears[1]).item()
            )
            if len(shears) == 4:
                shear_y = float(
                    torch.empty(1).uniform_(shears[2], shears[3]).item()
                )
        shear = (shear_x, shear_y)

        ret = [angle, translations, scale, shear]
        return F.affine(
            img,
            *ret,
            interpolation=self.interpolation,
            fill=fill,
            center=self.center,
        )


class SoftLabel(Transform):
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

    def forward(self, inpt: Any) -> Any:
        eps = 1e-5
        new_min = (torch.rand(1) - 0.5) * 0.4
        if new_min < 0:
            return inpt
        new_max = 1 - torch.rand(1) * 0.2
        mask = inpt >= 0.5
        noise = torch.randn_like(inpt)

        x = inpt + noise * self.noise_rate
        x = F.gaussian_blur(x, kernel_size=self.kernel_size, sigma=self.sigma)
        x[mask] = self.rescale(x[mask], 0.5, new_max)
        x[~mask] = self.rescale(x[~mask], new_min, 0.5 - eps)
        return x


class BoundedRandomCrop(transforms.RandomCrop):
    def get_params(self, mask) -> Dict[str, Any]:
        cropped_height, cropped_width = self.size
        # _, mask = flat_inputs
        # assert isinstance(image, torch.Tensor), f"Expected image to be a tensor, but got {type(image)}"
        # assert isinstance(mask, Mask), f"Expected mask to be a Mask object, but got {type(mask)}"

        _, mask_y, mask_x = torch.where(mask > 0.5)
        mask_width_min = mask_x.min().item()
        mask_width_max = mask_x.max().item()
        mask_height_min = mask_y.min().item()
        mask_height_max = mask_y.max().item()

        mask_width = mask_width_max - mask_width_min
        mask_height = mask_height_max - mask_height_min
        if mask_height > cropped_height or mask_width > cropped_width:
            rate = min(
                cropped_width / mask_width * 0.9,
                cropped_height / mask_height * 0.9,
            )
            ori_height, ori_width = mask.shape[-2:]
            resized_height = int(ori_height * rate)
            resized_width = int(ori_width * rate)
            needs_resize = True
            resize_size = (resized_height, resized_width)

            mask = F.resize(
                mask,
                size=[resized_height, resized_width],
                antialias=True,
            )

            _, mask_y, mask_x = torch.where(mask > 0.5)
            mask_width_min = mask_x.min().item()
            mask_width_max = mask_x.max().item()
            mask_height_min = mask_y.min().item()
            mask_height_max = mask_y.max().item()
        else:
            needs_resize = False
            resize_size = None

        padded_height, padded_width = mask.shape[-2:]
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
            (
                True,
                int(
                    torch.randint(
                        max(
                            0,
                            mask_height_max - cropped_height + pad_top,
                        ),
                        min(
                            padded_height - cropped_height + 1,
                            mask_height_min + pad_top + 1,
                        ),
                        size=(),
                    )
                ),
            )
            if padded_height > cropped_height
            else (False, 0)
        )
        needs_horz_crop, left = (
            (
                True,
                int(
                    torch.randint(
                        max(
                            0,
                            mask_width_max - cropped_width + pad_left,
                        ),
                        min(
                            padded_width - cropped_width + 1,
                            mask_width_min + pad_left + 1,
                        ),
                        size=(),
                    )
                ),
            )
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
            resize_size=resize_size,
        )

    def forward(self, inpt) -> Any:
        image, mask = inpt
        params = self.get_params(mask)
        if params["needs_resize"]:
            image = F.resize(image, size=params["resize_size"], antialias=True)
            mask = F.resize(mask, size=params["resize_size"], antialias=True)

        if params["needs_pad"]:
            image = F.pad(
                image,
                padding=params["padding"],
                padding_mode=self.padding_mode,
            )
            mask = F.pad(
                mask,
                padding=params["padding"],
                padding_mode=self.padding_mode,
            )

        if params["needs_crop"]:
            image = F.crop(
                image,
                top=params["top"],
                left=params["left"],
                height=params["height"],
                width=params["width"],
            )
            mask = F.crop(
                mask,
                top=params["top"],
                left=params["left"],
                height=params["height"],
                width=params["width"],
            )

        return image, mask


class ResizeMax(Transform):
    """
    A PyTorch Transform class that resizes an image such that the maximum dimension
    is equal to a specified size while maintaining the aspect ratio.
    """

    def __init__(self, max_sz: int = 256):
        """
        Initialize ResizeMax object with a specified max_sz.
        """
        # Call to the parent class (Transform) constructor
        super().__init__()

        # Set the maximum size for any dimension of the image
        self.max_sz = max_sz

    def get_params(self, x: torch.Tensor):
        height, width = x.shape[-2:]
        size = int(min(height, width) / (max(height, width) / self.max_sz))
        return {"size": size}

    def forward(self, inpt) -> torch.Tensor:  # The resized image tensor.
        """
        Apply the ResizeMax transformation on an input image tensor.
        """
        image, mask = inpt
        params = self.get_params(image)
        # Resize the image tensor with antialiasing for smoother output
        image = F.resize(image, size=params["size"], antialias=True)
        mask = F.resize(mask, size=params["size"], antialias=True)

        # Return the transformed (resized) image tensor
        return image, mask


class ToTensor(Transform):
    def forward(self, inpt):
        img, mask = inpt
        img = F.to_pil_image(img)
        img = F.to_tensor(img)
        return img, mask


class RandomFlip(Transform):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, inpt):
        img, mask = inpt
        if random.random() < self.p:
            img = F.hflip(img)
            mask = F.hflip(mask)

        if random.random() < self.p:
            img = F.vflip(img)
            mask = F.vflip(mask)
        return img, mask


class RandomRotation(transforms.RandomRotation):
    def get_fill(self, x):
        fill = self.fill
        channels, _, _ = F.get_dimensions(x)
        if isinstance(x, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            else:
                fill = [float(f) for f in fill]
        return fill

    def forward(self, inpt):
        img, mask = inpt
        angle = self.get_params(self.degrees)

        img = F.rotate(
            img,
            angle,
            self.interpolation,
            self.expand,
            self.center,
            self.get_fill(img),
        )
        mask = F.rotate(
            mask,
            angle,
            self.interpolation,
            self.expand,
            self.center,
            self.get_fill(mask),
        )
        return img, mask
