"""旧架构 HBSNet checkpoint 适配加载（bce 主干 → 当前模型结构）。

旧 checkpoint（2024-03, v0.7）特征：
- 主干键 `bce.*`：UNet 7 层，channels [8,16,32,64,128,256,512]，downs=ups=6 → 输出 256×256
- pre_stn/post_stn：`localization` + `fc_loc`（Sequential），无冗余 fc_loc1/fc_loc2 属性
- checkpoint 不含 config（旧版不存档）

适配方法（结构兼容，无需改权重）：
- 用旧 channels 构建当前 UNet/HBSNet（inc/downs/ups/outc + DoubleConv/BN 完全同构）
- `bce.` → `backbone.` 键改名
- `fc_loc.{0,2}.*` 权重复制到冗余 `fc_loc{1,2}.*`（STN forward 只用 fc_loc，复制仅为 strict 加载）
- stn_mode 由键集推断：有 post_stn→3，有 pre_stn→1，否则 0

推理：输出 256×256 → `resample_to_ghbs` 用 map_coordinates 重采样到 GHBS 128 网格
（radius=50 同源，旧网格中心 128 与 GHBS 中心 64 同物理坐标）。

用法：
    net, stn_mode = build_old("runs/remote_no_rn/no_rn_stn1_best993.pth")
    field = infer_to_ghbs(net, gray256)          # (128,128) 复数场
"""

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.ndimage import map_coordinates

from hbsn._legacy import pickle_shim  # noqa: F401  旧 Config 反序列化桩
from hbsn.registry import get_spec

# 旧架构 channels（7 层，deep 512；当前默认 6 层 deep 256）
OLD_CHANNELS = [8, 16, 32, 64, 128, 256, 512]
# GHBS 网格（128×128，radius=50）；旧模型 mask 亦 radius=50 → 同物理坐标
R, CX_OLD, CY_OLD = 50.0, 128.0, 128.0


def infer_stn_mode(state_dict: dict) -> int:
    """由 state_dict 键集推断旧 checkpoint 的 stn_mode。"""
    has_pre = any(k.startswith("pre_stn.") for k in state_dict)
    has_post = any(k.startswith("post_stn.") for k in state_dict)
    return 3 if has_post else (1 if has_pre else 0)


def build_old(checkpoint_path, device="cpu"):
    """构建旧架构 HBSNet 并 strict 加载旧权重。

    返回 (net, stn_mode)。net 为当前 HBSNet 类（输出 256×256），
    backbone 通道按旧架构（7 层）。推理后需 resample_to_ghbs 到 128 网格。
    """
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    stn_mode = infer_stn_mode(sd)

    spec = get_spec("hbsn")
    cfg = OmegaConf.merge(
        OmegaConf.structured(spec.net_schema),
        OmegaConf.create(
            {
                "channels_down": list(OLD_CHANNELS),
                "channels_up": list(OLD_CHANNELS),
                "stn_mode": stn_mode,
                "device": device,
            }
        ),
    )
    net = spec.net.factory(cfg)

    # bce.* → backbone.*；fc_loc.0/2 → 冗余 fc_loc1/2（同源权重）
    mapped = {
        (k.replace("bce.", "backbone.") if k.startswith("bce.") else k): v
        for k, v in sd.items()
    }
    for k, v in list(mapped.items()):
        if ".fc_loc.0." in k:
            mapped[k.replace(".fc_loc.0.", ".fc_loc1.")] = v
        if ".fc_loc.2." in k:
            mapped[k.replace(".fc_loc.2.", ".fc_loc2.")] = v
    net.load_state_dict(mapped, strict=True)
    net.eval()
    return net, stn_mode


def resample_to_ghbs(field256, z, mask, order=1):
    """旧模型 256×256 场 → GHBS 128 网格（map_coordinates，radius=50 同源）。

    旧网格中心 (128,128)，单位圆半径 50；GHBS 中心 (64,64) 半径 50，
    物理坐标一致，故直接按 z 映射到旧网格像素插值。
    """
    src_col = (z.real * R + CX_OLD).reshape(-1)
    src_row = (CY_OLD - z.imag * R).reshape(-1)
    out = np.empty((2, *z.shape))
    for _c in range(2):
        re = map_coordinates(
            field256[0],
            [src_row, src_col],
            order=order,
            mode="constant",
            cval=0,
        )
        im = map_coordinates(
            field256[1],
            [src_row, src_col],
            order=order,
            mode="constant",
            cval=0,
        )
        out[0] = re.reshape(z.shape)
        out[1] = im.reshape(z.shape)
    return (out[0] + 1j * out[1]) * mask


def infer_to_ghbs(net, gray_img, z, mask):
    """gray_img: (256,256) uint8/float [0,255] → GHBS 128 复数场（numpy）。

    net 可在 cuda：输入跟随 net 设备，输出转回 CPU numpy，重采样用 numpy z/mask。
    """
    dev = next(net.parameters()).device
    x = torch.as_tensor(gray_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    x = (x / 255.0).to(dev)
    with torch.no_grad():
        out = net(x)[0].cpu().numpy()  # (2,256,256)
    return resample_to_ghbs(out, z, mask)
