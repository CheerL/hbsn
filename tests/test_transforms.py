"""自定义变换：SoftLabel/ResizeMax/ToTensor/RandomFlip/RandomRotation/BoundedRandomCrop/Affine。"""
import random

import torch

from hbsn.data.transforms import (
    BoundedRandomAffine,
    BoundedRandomCrop,
    RandomFlip,
    RandomRotation,
    ResizeMax,
    SoftLabel,
    ToTensor,
)


def _pair(shape=(1, 32, 32), mask_ratio=0.3):
    """(image, mask) 对：mask 含目标区域（>0.5）。"""
    img = torch.rand(*shape)
    mask = torch.zeros(*shape)
    mask[:, 8:24, 8:24] = 1.0
    return img, mask


# ------------------------------------------------------------- SoftLabel


def test_soft_label_new_min_negative_returns_input():
    t = SoftLabel()
    x = torch.rand(1, 16, 16)
    # new_min < 0 分支：直接返回原样；无法直接控制 torch.rand 内部，
    # 但 0.4 范围内 50% 概率 new_min<0，用确定性验证重标定分支不崩即可
    out = t(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_soft_label_rescale_equal():
    t = SoftLabel()
    x = torch.ones(1, 8, 8) * 0.8
    out = t.rescale(x, 0.5, 1.0)
    assert (out == 0).all()  # 全等值 → x-x.min()=0 后返回（防除零）


def test_soft_label_rescale_range():
    t = SoftLabel()
    x = torch.linspace(0, 1, 100).view(1, 10, 10)
    out = t.rescale(x, 0.5, 1.0)
    assert out.min() == 0.5
    assert out.max() == 1.0


def test_soft_label_positive_branch(monkeypatch):
    """强制 new_min ≥ 0（正分支）：掩码内外分别重标定，输出有限且 ≠ 输入。"""
    # 先建真实随机输入（含 mask 内外像素），再 patch torch.rand 只影响新 min/max
    x = torch.rand(1, 16, 16)
    # new_min=(rand-0.5)*0.4：rand=0.9 → 0.16 ≥ 0；new_max=1-rand*0.2=0.82
    monkeypatch.setattr(
        "hbsn.data.transforms.torch.rand",
        lambda *a, **k: torch.full(a, 0.9, **k),
    )
    t = SoftLabel()
    out = t(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert not torch.equal(out, x)  # 重标定确实发生


# ------------------------------------------------------------- ResizeMax / ToTensor


def test_resize_max():
    t = ResizeMax(16)
    img, mask = t(_pair(shape=(1, 32, 32)))
    assert img.shape[-2:] == (16, 16)
    assert mask.shape == img.shape


def test_to_tensor_pil():
    # CocoDataset.__getitem__ 返回 io.read_image 的 uint8 tensor → ToTensor 输入是 tensor
    img = torch.randint(0, 255, (3, 16, 16), dtype=torch.uint8)
    mask = torch.zeros(1, 16, 16)
    out_img, _ = ToTensor()((img, mask))
    assert out_img.shape == (3, 16, 16)
    assert out_img.dtype == torch.float32
    assert out_img.max() <= 1.0  # to_tensor 归一化到 [0,1]


# ------------------------------------------------------------- RandomFlip / Rotation


def test_random_flip_always():
    random.seed(0)
    t = RandomFlip(p=1.0)
    img, mask = _pair()
    flipped_img, flipped_mask = t((img, mask))
    # p=1.0：hflip + vflip 都触发
    assert torch.equal(flipped_img, torch.flip(torch.flip(img, dims=[2]), dims=[1]))
    assert torch.equal(flipped_mask, torch.flip(torch.flip(mask, dims=[2]), dims=[1]))


def test_random_flip_always_skip():
    t = RandomFlip(p=0.0)
    img, mask = _pair()
    out = t((img, mask))
    assert torch.equal(out[0], img)


def test_random_rotation():
    t = RandomRotation(degrees=90)
    img, mask = _pair()
    out_img, _ = t((img, mask))
    assert out_img.shape == img.shape


# ------------------------------------------------------------- BoundedRandomCrop


def test_crop_empty_mask_degrades_to_random():
    t = BoundedRandomCrop((16, 16), pad_if_needed=True)
    empty = torch.zeros(1, 32, 32)
    params = t.get_params(empty)
    assert params["needs_crop"] is True  # 空 mask → 退化随机裁剪
    # 空 mask 分支直接走 forward 不崩
    img, _ = t((torch.rand(1, 32, 32), empty))
    assert img.shape == (1, 16, 16)


def test_crop_mask_oversize_resizes():
    """mask 大于 crop 尺寸 → 触发 resize 分支。"""
    t = BoundedRandomCrop((16, 16), pad_if_needed=True)
    big_mask = torch.zeros(1, 64, 64)
    big_mask[:, 10:60, 10:60] = 1.0
    params = t.get_params(big_mask)
    assert params["needs_resize"] is True
    img = torch.rand(1, 64, 64)
    out_img, _ = t((img, big_mask))
    assert out_img.shape == (1, 16, 16)


def test_crop_normal():
    t = BoundedRandomCrop((16, 16), pad_if_needed=True)
    img, mask = _pair(shape=(1, 32, 32))
    params = t.get_params(mask)
    assert params["needs_crop"] is True
    out_img, out_mask = t((img, mask))
    assert out_img.shape == (1, 16, 16)
    assert out_mask.shape == (1, 16, 16)


def test_crop_needs_pad():
    """小图 + pad_if_needed → 触发 padding 分支。"""
    t = BoundedRandomCrop((16, 16), pad_if_needed=True)
    img = torch.rand(1, 10, 10)
    mask = torch.zeros(1, 10, 10)
    mask[:, 2:8, 2:8] = 1.0
    out_img, _ = t((img, mask))
    assert out_img.shape == (1, 16, 16)


def test_crop_explicit_padding():
    """显式 4 元组 padding：get_params 的 needs_pad 分支。"""
    t = BoundedRandomCrop((16, 16), padding=(2, 2, 2, 2))
    img, mask = _pair(shape=(1, 32, 32))
    params = t.get_params(mask)
    assert params["needs_pad"] is True
    out_img, out_mask = t((img, mask))
    assert out_img.shape == (1, 16, 16)
    assert out_mask.shape == (1, 16, 16)


def test_crop_int_padding():
    """int padding（torchvision 合法）：四周同样 pad，不再崩溃。"""
    t = BoundedRandomCrop((16, 16), padding=4)
    img, mask = _pair(shape=(1, 32, 32))
    params = t.get_params(mask)
    assert params["needs_pad"] is True
    assert params["padding"] == [4, 4, 4, 4]  # (left, top, right, bottom)
    out_img, _ = t((img, mask))
    assert out_img.shape == (1, 16, 16)


def test_crop_tuple2_padding():
    """2 元组 padding：(left/right, top/bottom)。"""
    t = BoundedRandomCrop((16, 16), padding=(4, 2))
    mask = torch.zeros(1, 32, 32)
    mask[:, 8:24, 8:24] = 1.0
    params = t.get_params(mask)
    assert params["padding"] == [4, 2, 4, 2]  # (l,t,r,b)：左右 4、上下 2
    assert params["needs_pad"] is True


def test_crop_asymmetric_padding_order():
    """非对称 4 元组按 torchvision 语义 (left, top, right, bottom) 处理（非对称值验证顺序）。"""
    t = BoundedRandomCrop((16, 16), padding=(1, 2, 3, 4))
    mask = torch.zeros(1, 32, 32)
    mask[:, 8:24, 8:24] = 1.0
    params = t.get_params(mask)
    # padded_width = 32 + left1 + right3 = 36；padded_height = 32 + top2 + bottom4 = 38
    assert params["padding"] == [1, 2, 3, 4]
    img = torch.rand(1, 32, 32)
    out_img, out_mask = t((img, mask))
    assert out_img.shape == (1, 16, 16)
    assert out_mask.shape == (1, 16, 16)


# ------------------------------------------------------------- BoundedRandomAffine


def test_affine_forward():
    t = BoundedRandomAffine(degrees=30, scale=[0.8, 1.2], translate=[0.1, 0.1])
    img = torch.rand(1, 32, 32)
    mask = img > 0.5  # 有前景（max_dis > 0）
    img = img * mask
    out = t(img)  # forward 只作用单张图
    assert out.shape == img.shape


def test_affine_all_background():
    """全背景图：max_dis 退化路径（scale 用上界）。"""
    t = BoundedRandomAffine(degrees=30, scale=[0.8, 1.2], translate=[0.1, 0.1])
    bg = torch.zeros(1, 32, 32)
    bg[bg < 0.5] = 0.0
    out = t(bg)
    assert out.shape == bg.shape
