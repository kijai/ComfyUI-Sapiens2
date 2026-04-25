from typing import Tuple

import torch

import comfy.utils

IMAGENET_MEAN = (123.675, 116.28, 103.53)
IMAGENET_STD = (58.395, 57.12, 57.375)


def _resize_pad_to(x_bchw, target_h, target_w):
    """Resize keeping aspect ratio, then center-pad with zeros to (target_h, target_w).

    Returns the padded tensor and the (top, bottom, left, right) padding so
    callers can reverse it on the model output.
    """
    _, _, h, w = x_bchw.shape
    scale = min(target_h / h, target_w / w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    if (new_h, new_w) != (h, w):
        x_bchw = comfy.utils.common_upscale(x_bchw, new_w, new_h, "bilinear", "disabled")
    pad_top = (target_h - new_h) // 2
    pad_bottom = target_h - new_h - pad_top
    pad_left = (target_w - new_w) // 2
    pad_right = target_w - new_w - pad_left
    if pad_top or pad_bottom or pad_left or pad_right:
        x_bchw = torch.nn.functional.pad(
            x_bchw, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0,
        )
    return x_bchw, (pad_top, pad_bottom, pad_left, pad_right)


def to_model_input(image_bhwc, target_h=1024, target_w=768, device=None, dtype=None):
    """Convert a ComfyUI IMAGE batch into Sapiens2-ready (B, 3, H, W) input.

    Returns ``(x, original_hw, padding)`` where ``padding`` is
    (top, bottom, left, right) in target-resolution pixels, and ``original_hw``
    is the input (H, W) before any resize.
    """
    h, w = image_bhwc.shape[1], image_bhwc.shape[2]
    x = image_bhwc.movedim(-1, 1).contiguous()       # (B, 3, H, W), [0, 1]
    if device is not None:
        x = x.to(device=device, dtype=dtype, non_blocking=True)
    elif dtype is not None:
        x = x.to(dtype=dtype)
    x = x * 255.0
    x, padding = _resize_pad_to(x, target_h, target_w)

    mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std, (h, w), padding


def crop_pad_and_resize(y_bchw, padding: Tuple[int, int, int, int],
                        out_h: int, out_w: int, upscale_method: str = "bilinear"):
    """Reverse the resize+pad: crop padded borders, resize back to (out_h, out_w)."""
    pad_top, pad_bottom, pad_left, pad_right = padding
    _, _, H, W = y_bchw.shape
    y = y_bchw[:, :, pad_top : H - pad_bottom, pad_left : W - pad_right]
    if y.shape[-2:] != (out_h, out_w):
        y = comfy.utils.common_upscale(y, out_w, out_h, upscale_method, "disabled")
    return y
