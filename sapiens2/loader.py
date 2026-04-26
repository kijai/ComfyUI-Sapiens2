"""Auto-detect arch + task from a Sapiens2 safetensors checkpoint, build the
ported model, load the state dict, and wrap in a ComfyUI ModelPatcher."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

import comfy.model_management
import comfy.model_patcher
import comfy.ops
import comfy.utils
import folder_paths

from .backbone import ARCHS, Sapiens2
from .heads import NormalHead, PointmapHead, PoseHeatmapHead, SegHead
from .model import Sapiens2Estimator

# arch detection

_EMBED_DIM_TO_ARCH = {cfg["embed_dims"]: name for name, cfg in ARCHS.items()}


def _detect_arch(sd: Dict[str, torch.Tensor], prefix: str) -> str:
    key = f"{prefix}patch_embed.projection.weight"
    if key not in sd:
        raise ValueError(f"checkpoint missing {key}; not a Sapiens2 backbone state_dict")
    embed_dim = sd[key].shape[0]
    if embed_dim not in _EMBED_DIM_TO_ARCH:
        raise ValueError(
            f"unrecognised embed_dim {embed_dim}; supported: {list(_EMBED_DIM_TO_ARCH)}"
        )
    return _EMBED_DIM_TO_ARCH[embed_dim]


def _detect_prefix(sd: Dict[str, torch.Tensor]) -> str:
    """Return either 'backbone.' (combined task ckpt) or '' (pretrain-only ckpt)."""
    if "backbone.patch_embed.projection.weight" in sd:
        return "backbone."
    if "patch_embed.projection.weight" in sd:
        return ""
    raise ValueError("could not locate patch_embed.projection.weight in state_dict")


# task detection

_TASK_FINAL_KEYS = {
    "normal":   "decode_head.conv_normal.weight",
    "seg":      "decode_head.conv_seg.weight",
    "pointmap": "decode_head.conv_pointmap.weight",
    "pose":     "decode_head.conv_pose.weight",
}


def _detect_task(sd: Dict[str, torch.Tensor]) -> Optional[str]:
    for task, key in _TASK_FINAL_KEYS.items():
        if key in sd:
            return task
    return None


# head channel-width extraction

def _extract_upsample_channels(sd) -> List[int]:
    """Read out_ch from `decode_head.upsample_blocks.{i}.0.weight` shapes
    (Conv2d weight (out_ch*4, in_ch, 3, 3) feeding a PixelShuffle(2))."""
    chs: List[int] = []
    i = 0
    while (k := f"decode_head.upsample_blocks.{i}.0.weight") in sd:
        chs.append(sd[k].shape[0] // 4)
        i += 1
    return chs


def _extract_dense_conv_layers(sd) -> Tuple[List[int], List[int]]:
    """Read out_ch and kernel from `decode_head.conv_layers.{0,3,6,...}.weight`."""
    chs: List[int] = []
    ks: List[int] = []
    j = 0
    while f"decode_head.conv_layers.{j}.weight" in sd:
        w = sd[f"decode_head.conv_layers.{j}.weight"]
        chs.append(w.shape[0])
        ks.append(w.shape[2])
        j += 3
    return chs, ks


def _build_normal_head(sd, dtype, device, operations, embed_dim) -> NormalHead:
    upsample_channels = _extract_upsample_channels(sd)
    conv_out_channels, conv_kernel_sizes = _extract_dense_conv_layers(sd)
    return NormalHead(
        in_channels=embed_dim,
        upsample_channels=upsample_channels,
        conv_out_channels=conv_out_channels or None,
        conv_kernel_sizes=conv_kernel_sizes or None,
        dtype=dtype, device=device, operations=operations,
    )


def _build_pointmap_head(sd, dtype, device, operations, embed_dim) -> PointmapHead:
    upsample_channels = _extract_upsample_channels(sd)
    conv_out_channels, conv_kernel_sizes = _extract_dense_conv_layers(sd)

    # scale branch
    has_scale = "decode_head.scale_conv_layers.0.weight" in sd
    scale_kwargs: Dict[str, Any] = {}
    if has_scale:
        scale_chs: List[int] = []
        scale_ks: List[int] = []
        j = 0
        while f"decode_head.scale_conv_layers.{j}.weight" in sd:
            w = sd[f"decode_head.scale_conv_layers.{j}.weight"]
            scale_chs.append(w.shape[0])
            scale_ks.append(w.shape[2])
            j += 3
        scale_kwargs["scale_conv_out_channels"] = tuple(scale_chs)
        scale_kwargs["scale_conv_kernel_sizes"] = tuple(scale_ks)

        # final Linear stack: scale_final_layer.{1,3,5,...}.weight
        sfl: List[int] = [sd["decode_head.scale_final_layer.1.weight"].shape[1]]
        j = 1
        while (k := f"decode_head.scale_final_layer.{j}.weight") in sd:
            sfl.append(sd[k].shape[0])
            j += 2
        scale_kwargs["scale_final_layer"] = tuple(sfl)
    else:
        scale_kwargs["scale_conv_out_channels"] = None

    return PointmapHead(
        in_channels=embed_dim,
        upsample_channels=upsample_channels,
        conv_out_channels=conv_out_channels or None,
        conv_kernel_sizes=conv_kernel_sizes or None,
        dtype=dtype, device=device, operations=operations,
        **scale_kwargs,
    )


def _extract_deconv_channels(sd) -> Tuple[List[int], List[int]]:
    """Read out_ch and kernel from `decode_head.deconv_layers.{0,3,6,...}.weight`
    (ConvTranspose2d weight is (in_ch, out_ch, k, k) — out_ch is shape[1])."""
    chs: List[int] = []
    ks: List[int] = []
    j = 0
    while f"decode_head.deconv_layers.{j}.weight" in sd:
        w = sd[f"decode_head.deconv_layers.{j}.weight"]
        chs.append(w.shape[1])
        ks.append(w.shape[2])
        j += 3
    return chs, ks


def _build_seg_head(sd, dtype, device, operations, embed_dim) -> SegHead:
    deconv_chs, deconv_ks = _extract_deconv_channels(sd)
    conv_chs, conv_ks = _extract_dense_conv_layers(sd)
    num_classes = sd["decode_head.conv_seg.weight"].shape[0]
    return SegHead(
        in_channels=embed_dim,
        num_classes=num_classes,
        deconv_out_channels=deconv_chs,
        deconv_kernel_sizes=deconv_ks,
        conv_out_channels=conv_chs or None,
        conv_kernel_sizes=conv_ks or None,
        dtype=dtype, device=device, operations=operations,
    )


def _build_pose_head(sd, dtype, device, operations, embed_dim) -> PoseHeatmapHead:
    deconv_chs, deconv_ks = _extract_deconv_channels(sd)
    conv_chs, conv_ks = _extract_dense_conv_layers(sd)
    num_kp = sd["decode_head.conv_pose.weight"].shape[0]
    return PoseHeatmapHead(
        in_channels=embed_dim,
        out_channels=num_kp,
        deconv_out_channels=deconv_chs,
        deconv_kernel_sizes=deconv_ks,
        conv_out_channels=conv_chs or None,
        conv_kernel_sizes=conv_ks or None,
        dtype=dtype, device=device, operations=operations,
    )


_HEAD_BUILDERS = {
    "normal":   _build_normal_head,
    "seg":      _build_seg_head,
    "pointmap": _build_pointmap_head,
    "pose":     _build_pose_head,
}


# top-level entry

def load_sapiens2(filename: str):
    """Load a Sapiens2 checkpoint and return ``(patcher, task)``.

    ``task`` is one of {"normal", "seg", "pointmap", "pose", None}. None means
    the checkpoint is a backbone-only pretrain — the model has no decode_head.
    """
    full_path = folder_paths.get_full_path("sapiens2", filename)
    if full_path is None:
        raise FileNotFoundError(f"sapiens2 model not found: {filename}")

    sd = comfy.utils.load_torch_file(full_path, safe_load=True)

    prefix = _detect_prefix(sd)
    arch = _detect_arch(sd, prefix)

    # promote pretrain-only sd to backbone.* prefix so the estimator loads it
    if prefix == "":
        sd = {f"backbone.{k}": v for k, v in sd.items()}

    task = _detect_task(sd)

    load_device = comfy.model_management.text_encoder_device()
    offload_device = comfy.model_management.text_encoder_offload_device()
    dtype = comfy.model_management.text_encoder_dtype(load_device)
    operations = comfy.ops.manual_cast

    backbone = Sapiens2(
        arch=arch,
        dtype=dtype, device=offload_device, operations=operations,
    )

    if task is not None:
        head = _HEAD_BUILDERS[task](
            sd, dtype, offload_device, operations, embed_dim=backbone.embed_dims
        )
    else:
        head = None

    model = Sapiens2Estimator(backbone, head)
    model.eval()

    incompat = model.load_state_dict(sd, strict=False)
    logging.info(f"[Sapiens2] loaded {filename} arch={arch} task={task}")
    if incompat.missing_keys:
        logging.info(f"[Sapiens2] missing keys ({len(incompat.missing_keys)}): "
                     f"{incompat.missing_keys[:5]}{'...' if len(incompat.missing_keys) > 5 else ''}")
    if incompat.unexpected_keys:
        logging.info(f"[Sapiens2] unexpected keys ({len(incompat.unexpected_keys)}): "
                     f"{incompat.unexpected_keys[:5]}{'...' if len(incompat.unexpected_keys) > 5 else ''}")

    patcher = comfy.model_patcher.CoreModelPatcher(model, load_device=load_device, offload_device=offload_device)
    return patcher, task
