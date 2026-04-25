import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing_extensions import override

import comfy.model_management
import comfy.utils
import folder_paths
from comfy_api.latest import ComfyExtension, io

from .sapiens2.loader import load_sapiens2
from .sapiens2.preprocess import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    crop_pad_and_resize,
    to_model_input,
)

_sapiens2_dir = os.path.join(folder_paths.models_dir, "sapiens2")
os.makedirs(_sapiens2_dir, exist_ok=True)
folder_paths.add_model_folder_path("sapiens2", _sapiens2_dir)


# Constants
_SEG_PALETTE = [
    (0, 0, 0),       (128, 0, 0),     (0, 128, 0),     (128, 128, 0),
    (0, 0, 128),     (128, 0, 128),   (0, 128, 128),   (128, 128, 128),
    (64, 0, 0),      (192, 0, 0),     (64, 128, 0),    (192, 128, 0),
    (64, 0, 128),    (192, 0, 128),   (64, 128, 128),  (192, 128, 128),
    (0, 64, 0),      (128, 64, 0),    (0, 192, 0),     (128, 192, 0),
    (0, 64, 128),    (128, 64, 128),  (0, 192, 128),   (128, 192, 128),
    (64, 64, 0),     (192, 64, 0),    (64, 192, 0),    (192, 192, 0),
    (64, 64, 128),
]

# Goliath-308 → OpenPose subset mappings.
_GOLIATH_TO_OPENPOSE_BODY = [0, 69, 6, 8, 41, 5, 7, 62, 10, 12, 14, 9, 11, 13, 2, 1, 4, 3]
_GOLIATH_TO_OPENPOSE_FOOT = list(range(15, 21))

def _hand_indices(start: int, wrist: int):
    fingers = []
    for f in range(5):
        base = start + f * 4
        fingers.extend([base + 3, base + 2, base + 1, base])  # third(CMC)→tip
    return [wrist, *fingers]

_GOLIATH_TO_OPENPOSE_HAND_R = _hand_indices(start=21, wrist=41)
_GOLIATH_TO_OPENPOSE_HAND_L = _hand_indices(start=42, wrist=62)

_BODY_IDX_NP   = np.asarray(_GOLIATH_TO_OPENPOSE_BODY,   dtype=np.int64)
_FOOT_IDX_NP   = np.asarray(_GOLIATH_TO_OPENPOSE_FOOT,   dtype=np.int64)
_HAND_R_IDX_NP = np.asarray(_GOLIATH_TO_OPENPOSE_HAND_R, dtype=np.int64)
_HAND_L_IDX_NP = np.asarray(_GOLIATH_TO_OPENPOSE_HAND_L, dtype=np.int64)
_ZERO_FACE_70  = [0.0] * (70 * 3)

# Goliath face landmark groups (anatomical, by name from upstream config).
# Goliath has no jaw line — dlib 0..16 will be emitted as zeros.
_GOLIATH_FACE = dict(
    r_brow_up=[78, 80, 81, 83, 84],
    l_brow_up=[87, 89, 90, 92, 93],
    nose_bridge=[70, 71, 73, 74, 75, 178],
    nose_base=[180, 184, 179, 187, 181],
    r_eye_outer=121, r_eye_inner=120,
    r_eye_upper=[122, 123, 124, 125, 126, 127, 128],
    r_eye_lower=[163, 164, 165, 166, 167, 168, 169],
    l_eye_inner=96, l_eye_outer=97,
    l_eye_upper=[98, 99, 100, 101, 102, 103, 104],
    l_eye_lower=[146, 147, 148, 149, 150, 151, 152],
    lip_o_r_corner=188, lip_o_l_corner=189, cupid=190, lower_o_center=191,
    lip_o_upper=[192, 193, 196, 197, 198, 199],
    lip_o_lower=[194, 195, 200, 201, 202, 203],
    lip_i_r_corner=204, lip_i_l_corner=205, upper_i_center=206, lower_i_center=207,
    lip_i_upper=[208, 209, 212, 213, 214, 215],
    lip_i_lower=[210, 211, 216, 217, 218, 219],
    r_pupil=2, l_pupil=1,
)

# Skeleton edges from sapiens/pose/configs/_base_/keypoints308.py
# dataset_info.skeleton_info — body + feet + hands. Face is dots-only upstream.
_SAPIENS2_SKELETON = [
    (13, 11, (0, 255, 0)),    (11,  9, (0, 255, 0)),    (14, 12, (255, 128, 0)),
    (12, 10, (255, 128, 0)),  ( 9, 10, (51, 153, 255)), ( 5,  9, (51, 153, 255)),
    ( 6, 10, (51, 153, 255)), ( 5,  6, (51, 153, 255)), ( 5,  7, (0, 255, 0)),
    ( 6,  8, (255, 128, 0)),  ( 7, 62, (0, 255, 0)),    ( 8, 41, (255, 128, 0)),
    ( 1,  2, (51, 153, 255)), ( 0,  1, (51, 153, 255)), ( 0,  2, (51, 153, 255)),
    ( 1,  3, (51, 153, 255)), ( 2,  4, (51, 153, 255)), ( 3,  5, (51, 153, 255)),
    ( 4,  6, (51, 153, 255)), (13, 15, (0, 255, 0)),    (13, 16, (0, 255, 0)),
    (13, 17, (0, 255, 0)),    (14, 18, (255, 128, 0)),  (14, 19, (255, 128, 0)),
    (14, 20, (255, 128, 0)),
    (62, 45, (255, 128, 0)),  (45, 44, (255, 128, 0)),  (44, 43, (255, 128, 0)),
    (43, 42, (255, 128, 0)),  (62, 49, (255, 153, 255)),(49, 48, (255, 153, 255)),
    (48, 47, (255, 153, 255)),(47, 46, (255, 153, 255)),(62, 53, (102, 178, 255)),
    (53, 52, (102, 178, 255)),(52, 51, (102, 178, 255)),(51, 50, (102, 178, 255)),
    (62, 57, (255, 51, 51)),  (57, 56, (255, 51, 51)),  (56, 55, (255, 51, 51)),
    (55, 54, (255, 51, 51)),  (62, 61, (0, 255, 0)),    (61, 60, (0, 255, 0)),
    (60, 59, (0, 255, 0)),    (59, 58, (0, 255, 0)),
    (41, 24, (255, 128, 0)),  (24, 23, (255, 128, 0)),  (23, 22, (255, 128, 0)),
    (22, 21, (255, 128, 0)),  (41, 28, (255, 153, 255)),(28, 27, (255, 153, 255)),
    (27, 26, (255, 153, 255)),(26, 25, (255, 153, 255)),(41, 32, (102, 178, 255)),
    (32, 31, (102, 178, 255)),(31, 30, (102, 178, 255)),(30, 29, (102, 178, 255)),
    (41, 36, (255, 51, 51)),  (36, 35, (255, 51, 51)),  (35, 34, (255, 51, 51)),
    (34, 33, (255, 51, 51)),  (41, 40, (0, 255, 0)),    (40, 39, (0, 255, 0)),
    (39, 38, (0, 255, 0)),    (38, 37, (0, 255, 0)),
]

# Per-keypoint colors from upstream keypoint_info["color"].
_KP_COLORS = (
    [( 51, 153, 255)] * 70 +
    [(255, 255, 255)] * 26 +
    [(192,  64, 128)] * 24 +
    [( 64,  32, 192)] * 24 +
    [( 64, 192, 128)] * 17 +
    [( 64, 192,  32)] * 17 +
    [(  0, 192,   0)] * 10 +
    [(192,   0,   0)] * 16 +
    [(  0, 192, 192)] * 16 +
    [(200, 200,   0)] * 26 +
    [(  0, 200, 200)] * 26 +
    [(128, 192,  64)] *  9 +
    [(192,  32,  64)] *  9 +
    [(192, 128,  64)] *  9 +
    [( 32, 192, 192)] *  9
)

def _kp_color(idx: int):
    return _KP_COLORS[idx] if 0 <= idx < 308 else (200, 200, 200)

def _iter_chunks(patcher, x_bchw, frames_per_batch: int, desc: str = "Sapiens2"):
    """Yield ``(start_index, model_output)`` for each chunk of size
    ``frames_per_batch``. Reports progress to both the ComfyUI web UI and
    the terminal."""
    comfy.model_management.load_model_gpu(patcher)
    device = patcher.load_device
    B = x_bchw.shape[0]
    chunk = max(1, int(frames_per_batch))
    pbar = comfy.utils.ProgressBar(B)
    with tqdm(total=B, desc=desc) as tbar:
        for i in range(0, B, chunk):
            sub = x_bchw[i : i + chunk].to(device)
            with torch.no_grad():
                yield i, patcher.model(sub)
            n = sub.shape[0]
            pbar.update(n)
            tbar.update(n)
            del sub


# Pose: heatmap decoding (DARK + UDP, vectorised PyTorch port)
def _decode_heatmaps(hm_bkhw, input_h: int, input_w: int, blur_ks: int = 11):
    hm = hm_bkhw.to(torch.float32)
    B, K, H, W = hm.shape

    flat = hm.reshape(B, K, H * W)
    scores, idx = flat.max(dim=2)
    xs = (idx % W).to(torch.float32)
    ys = (idx // W).to(torch.float32)

    sigma = 0.3 * ((blur_ks - 1) * 0.5 - 1) + 0.8 if blur_ks > 1 else 1.0
    coords = torch.arange(blur_ks, dtype=hm.dtype, device=hm.device) - (blur_ks - 1) / 2.0
    g = torch.exp(-coords ** 2 / (2.0 * sigma ** 2))
    g = (g / g.sum()).to(hm.dtype)
    pad = blur_ks // 2
    hm_blur = hm.reshape(B * K, 1, H, W)
    hm_blur = F.conv2d(hm_blur, g.view(1, 1, 1, -1), padding=(0, pad))
    hm_blur = F.conv2d(hm_blur, g.view(1, 1, -1, 1), padding=(pad, 0))
    hm_blur = hm_blur.reshape(B, K, H, W)

    hm_log = hm_blur.clamp_(min=1e-3, max=50.0).log()
    hm_pad = F.pad(hm_log, (1, 1, 1, 1), mode="replicate")
    H2, W2 = H + 2, W + 2

    def neighbour(dx: int, dy: int):
        x_idx = (xs.long() + 1 + dx).clamp(0, W2 - 1)
        y_idx = (ys.long() + 1 + dy).clamp(0, H2 - 1)
        flat_idx = (y_idx * W2 + x_idx).unsqueeze(-1)
        return torch.gather(hm_pad.reshape(B, K, -1), 2, flat_idx).squeeze(-1)

    i_      = neighbour( 0,  0)
    ix1     = neighbour( 1,  0)
    ix1_    = neighbour(-1,  0)
    iy1     = neighbour( 0,  1)
    iy1_    = neighbour( 0, -1)
    ix1y1   = neighbour( 1,  1)
    ix1_y1_ = neighbour(-1, -1)

    dx = 0.5 * (ix1 - ix1_)
    dy = 0.5 * (iy1 - iy1_)
    dxx = ix1 - 2.0 * i_ + ix1_
    dyy = iy1 - 2.0 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + 2.0 * i_ - ix1_ - iy1_ + ix1_y1_)
    det = dxx * dyy - dxy * dxy + 1e-7
    nx = ( dyy * dx - dxy * dy) / det
    ny = (-dxy * dx + dxx * dy) / det
    xs = xs - nx
    ys = ys - ny

    xs = xs / max(W - 1, 1) * input_w
    ys = ys / max(H - 1, 1) * input_h
    return torch.stack([xs, ys], dim=-1), scores



# Pose: face dlib-68 mapping (vectorised)
def _goliath_to_dlib68_face_np(kps_np, sc_np):
    """Map (308, 2) Goliath coords + (308,) scores to a flat 70-keypoint
    face_keypoints_2d list (dlib-68 + 2 pupils). Goliath has no jawline, so
    dlib 0..16 are emitted as zeros."""
    g = _GOLIATH_FACE
    xs_all = kps_np[:, 0]
    ys_all = kps_np[:, 1]
    out: list = []

    def push(idx_seq):
        idx = np.asarray(idx_seq, dtype=np.int64)
        triples = np.empty((idx.shape[0], 3), dtype=np.float32)
        triples[:, 0] = xs_all[idx]
        triples[:, 1] = ys_all[idx]
        triples[:, 2] = sc_np[idx]
        out.extend(triples.flatten().tolist())

    def push_zeros(n):
        out.extend([0.0, 0.0, 0.0] * n)

    def sort_x(idx_list):
        idx = np.asarray(idx_list, dtype=np.int64)
        return idx[np.argsort(xs_all[idx])]

    push_zeros(17)
    push(sort_x(g["r_brow_up"]))
    push(sort_x(g["l_brow_up"]))
    nb = np.asarray(g["nose_bridge"], dtype=np.int64)
    nb = nb[np.argsort(ys_all[nb])]
    L = nb.shape[0]
    push([nb[0], nb[L // 3], nb[2 * L // 3], nb[-1]])
    push(g["nose_base"])

    upper_x = sort_x(g["r_eye_upper"])
    lower_x = sort_x(g["r_eye_lower"])
    push([g["r_eye_outer"], upper_x[len(upper_x) // 3], upper_x[2 * len(upper_x) // 3],
          g["r_eye_inner"], lower_x[2 * len(lower_x) // 3], lower_x[len(lower_x) // 3]])
    upper_x = sort_x(g["l_eye_upper"])
    lower_x = sort_x(g["l_eye_lower"])
    push([g["l_eye_inner"], upper_x[len(upper_x) // 3], upper_x[2 * len(upper_x) // 3],
          g["l_eye_outer"], lower_x[2 * len(lower_x) // 3], lower_x[len(lower_x) // 3]])

    cupid_x = float(xs_all[g["cupid"]])
    upper_lip = sort_x(g["lip_o_upper"])
    left  = upper_lip[xs_all[upper_lip] < cupid_x]
    right = upper_lip[xs_all[upper_lip] > cupid_x]
    push([g["lip_o_r_corner"]])
    push([int(left[0]),  int(left[-1])]  if len(left)  >= 2 else list(left)  + [g["lip_o_r_corner"]] * (2 - len(left)))
    push([g["cupid"]])
    push([int(right[0]), int(right[-1])] if len(right) >= 2 else list(right) + [g["lip_o_l_corner"]] * (2 - len(right)))
    push([g["lip_o_l_corner"]])

    lc_x = float(xs_all[g["lower_o_center"]])
    lower_lip = sort_x(g["lip_o_lower"])
    left_lo  = lower_lip[xs_all[lower_lip] < lc_x]
    right_lo = lower_lip[xs_all[lower_lip] > lc_x]
    push([int(right_lo[-1]), int(right_lo[0])] if len(right_lo) >= 2 else list(right_lo) + [g["lower_o_center"]] * (2 - len(right_lo)))
    push([g["lower_o_center"]])
    push([int(left_lo[-1]),  int(left_lo[0])]  if len(left_lo)  >= 2 else list(left_lo)  + [g["lower_o_center"]] * (2 - len(left_lo)))

    uic_x = float(xs_all[g["upper_i_center"]])
    upper_in = sort_x(g["lip_i_upper"])
    left_in  = upper_in[xs_all[upper_in] < uic_x]
    right_in = upper_in[xs_all[upper_in] > uic_x]
    top_r = int(left_in[0])  if len(left_in)  else g["upper_i_center"]
    top_l = int(right_in[-1]) if len(right_in) else g["upper_i_center"]
    lic_x = float(xs_all[g["lower_i_center"]])
    lower_in = sort_x(g["lip_i_lower"])
    bot_left  = lower_in[xs_all[lower_in] < lic_x]
    bot_right = lower_in[xs_all[lower_in] > lic_x]
    bot_r = int(bot_left[0])   if len(bot_left)  else g["lower_i_center"]
    bot_l = int(bot_right[-1]) if len(bot_right) else g["lower_i_center"]
    push([g["lip_i_r_corner"], top_r, g["upper_i_center"], top_l,
          g["lip_i_l_corner"], bot_l, g["lower_i_center"], bot_r])

    push([g["r_pupil"], g["l_pupil"]])
    return out



# Pose: format builders
def _flatten_subset_np(kps_np, sc_np, indices_np):
    out = np.empty((indices_np.shape[0], 3), dtype=np.float32)
    out[:, :2] = kps_np[indices_np]
    out[:, 2] = sc_np[indices_np]
    return out.flatten().tolist()


def _person_openpose(kps_np, sc_np, include_face: bool):
    face = _goliath_to_dlib68_face_np(kps_np, sc_np) if include_face else _ZERO_FACE_70
    return {
        "pose_keypoints_2d":       _flatten_subset_np(kps_np, sc_np, _BODY_IDX_NP),
        "foot_keypoints_2d":       _flatten_subset_np(kps_np, sc_np, _FOOT_IDX_NP),
        "face_keypoints_2d":       face,
        "hand_right_keypoints_2d": _flatten_subset_np(kps_np, sc_np, _HAND_R_IDX_NP),
        "hand_left_keypoints_2d":  _flatten_subset_np(kps_np, sc_np, _HAND_L_IDX_NP),
    }


def _person_raw_308(kps_np, sc_np):
    arr = np.empty((308, 3), dtype=np.float32)
    arr[:, :2] = kps_np
    arr[:, 2] = sc_np
    return {"keypoints_2d": arr.tolist()}


def _frame_openpose_np(kps_np, sc_np, cw, ch, include_face: bool):
    return {"canvas_width": cw, "canvas_height": ch,
            "people": [_person_openpose(kps_np, sc_np, include_face)]}


def _frame_raw_308_np(kps_np, sc_np, cw, ch):
    return {"canvas_width": cw, "canvas_height": ch,
            "people": [_person_raw_308(kps_np, sc_np)]}



# Pose: bbox-crop preprocessing
def _normalize_bboxes(bboxes, T: int):
    """Coerce a BOUNDING_BOX input to ``list[list[bbox]]`` of length T."""
    if bboxes is None:
        return None
    if not isinstance(bboxes, list):
        return [[bboxes]] * T
    if len(bboxes) == 0:
        return [None] * T
    first = bboxes[0]
    if isinstance(first, dict) or (isinstance(first, (list, tuple)) and len(first) == 4
                                   and not isinstance(first[0], (list, tuple, dict))):
        return [bboxes] * T
    if len(bboxes) < T:
        return list(bboxes) + [bboxes[-1]] * (T - len(bboxes))
    return list(bboxes[:T])


def _bbox_to_xyxy(bbox, img_w: int, img_h: int):
    if isinstance(bbox, dict):
        x1 = int(bbox.get("x", 0))
        y1 = int(bbox.get("y", 0))
        x2 = int(bbox.get("x", 0) + bbox.get("width", 0))
        y2 = int(bbox.get("y", 0) + bbox.get("height", 0))
    else:
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _make_bbox_crops(image_bhwc, bboxes_per_frame, target_h, target_w, device, dtype):
    """Build ``(N_crops, 3, target_h, target_w)`` of normalised crops + a
    list of inverse-mapping params so output keypoints can be remapped back
    to original-image coords."""
    img_h = image_bhwc.shape[1]
    img_w = image_bhwc.shape[2]
    mean = torch.tensor(IMAGENET_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=device, dtype=dtype).view(1, 3, 1, 1)

    crops = []
    mapping = []
    for f_idx, bb_list in enumerate(bboxes_per_frame):
        if not bb_list:
            continue
        for bbox in bb_list:
            xyxy = _bbox_to_xyxy(bbox, img_w, img_h)
            if xyxy is None:
                continue
            x1, y1, x2, y2 = xyxy
            crop = image_bhwc[f_idx, y1:y2, x1:x2, :]
            crop = crop.movedim(-1, 0).contiguous().unsqueeze(0)
            crop = crop.to(device=device, dtype=dtype, non_blocking=True) * 255.0

            h, w = y2 - y1, x2 - x1
            scale = min(target_h / h, target_w / w)
            new_h = int(round(h * scale))
            new_w = int(round(w * scale))
            if (new_h, new_w) != (h, w):
                crop = comfy.utils.common_upscale(crop, new_w, new_h, "bilinear", "disabled")
            pad_top = (target_h - new_h) // 2
            pad_left = (target_w - new_w) // 2
            pad_bottom = target_h - new_h - pad_top
            pad_right = target_w - new_w - pad_left
            if pad_top or pad_bottom or pad_left or pad_right:
                crop = torch.nn.functional.pad(
                    crop, (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant", value=0.0,
                )
            crop = (crop - mean) / std
            crops.append(crop)
            mapping.append((f_idx, x1, y1, scale, pad_top, pad_left))

    if not crops:
        return None, []
    return torch.cat(crops, dim=0), mapping


# Nodes (V3 schema-based)
class Sapiens2Loader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Sapiens2Loader",
            display_name="Sapiens2 Loader",
            category="Sapiens2",
            inputs=[
                io.Combo.Input("checkpoint", options=folder_paths.get_filename_list("sapiens2")),
            ],
            outputs=[io.Custom("SAPIENS2_MODEL").Output("sapiens2_model")],
        )

    @classmethod
    def execute(cls, checkpoint) -> io.NodeOutput:
        return io.NodeOutput(load_sapiens2(checkpoint))


_FRAMES_PER_BATCH_TOOLTIP = "Frames per forward pass. Lower if you OOM."


class Sapiens2Normal(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Sapiens2Normal",
            display_name="Sapiens2 Normals",
            category="Sapiens2",
            inputs=[
                io.Image.Input("image"),
                io.Custom("SAPIENS2_MODEL").Input("sapiens2_model"),
                io.Int.Input("frames_per_batch", default=1, min=1, max=256, tooltip=_FRAMES_PER_BATCH_TOOLTIP),
            ],
            outputs=[io.Image.Output(display_name="normal")],
        )

    @classmethod
    def execute(cls, image, sapiens2_model, frames_per_batch) -> io.NodeOutput:
        patcher, task = sapiens2_model
        if task != "normal":
            raise ValueError(f"loaded checkpoint is task={task!r}, expected 'normal'")
        comfy.model_management.load_model_gpu(patcher)
        x, orig_hw, padding = to_model_input(image, device=patcher.load_device, dtype=torch.float16)
        out_device = comfy.model_management.intermediate_device()
        out_dtype = comfy.model_management.intermediate_dtype()
        chunks = []
        for _, n in _iter_chunks(patcher, x, frames_per_batch, desc="Sapiens2 Normals"):
            n = n / n.norm(dim=1, keepdim=True).clamp_(min=1e-6)
            n = (n + 1.0) * 0.5
            n = crop_pad_and_resize(n, padding, orig_hw[0], orig_hw[1])
            chunks.append(n.to(device=out_device, dtype=out_dtype))
        out = torch.cat(chunks, dim=0).clamp_(0.0, 1.0).movedim(1, -1)
        return io.NodeOutput(out)


class Sapiens2Seg(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Sapiens2Seg",
            display_name="Sapiens2 Body-Part Segmentation",
            category="Sapiens2",
            inputs=[
                io.Image.Input("image"),
                io.Custom("SAPIENS2_MODEL").Input("sapiens2_model"),
                io.Int.Input("frames_per_batch", default=1, min=1, max=256, tooltip=_FRAMES_PER_BATCH_TOOLTIP),
            ],
            outputs=[
                io.Mask.Output(display_name="class_id_mask"),
                io.Image.Output(display_name="colored"),
            ],
        )

    @classmethod
    def execute(cls, image, sapiens2_model, frames_per_batch) -> io.NodeOutput:
        patcher, task = sapiens2_model
        if task != "seg":
            raise ValueError(f"loaded checkpoint is task={task!r}, expected 'seg'")
        comfy.model_management.load_model_gpu(patcher)
        x, orig_hw, padding = to_model_input(image, device=patcher.load_device, dtype=torch.float16)
        out_device = comfy.model_management.intermediate_device()
        out_dtype = comfy.model_management.intermediate_dtype()
        id_chunks = []
        num_classes = None
        for _, logits in _iter_chunks(patcher, x, frames_per_batch, desc="Sapiens2 Seg"):
            num_classes = logits.shape[1]
            ids = logits.argmax(dim=1)
            ids = F.interpolate(ids.unsqueeze(1).float(), size=(1024, 768), mode="nearest").squeeze(1).to(torch.long)
            ids = crop_pad_and_resize(
                ids.unsqueeze(1).float(), padding, orig_hw[0], orig_hw[1],
                upscale_method="nearest-exact",
            ).squeeze(1).to(torch.long)
            id_chunks.append(ids.to(out_device))
        class_ids = torch.cat(id_chunks, dim=0)

        palette = torch.tensor(_SEG_PALETTE[:num_classes], device=class_ids.device, dtype=out_dtype) / 255.0
        colored = palette[class_ids]
        mask = class_ids.to(out_dtype) / max((num_classes or 1) - 1, 1)
        return io.NodeOutput(mask, colored)


class Sapiens2Pointmap(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Sapiens2Pointmap",
            display_name="Sapiens2 Pointmap (XYZ)",
            category="Sapiens2",
            inputs=[
                io.Image.Input("image"),
                io.Custom("SAPIENS2_MODEL").Input("sapiens2_model"),
                io.Int.Input("frames_per_batch", default=1, min=1, max=256, tooltip=_FRAMES_PER_BATCH_TOOLTIP),
            ],
            outputs=[io.Image.Output(display_name="pointmap_xyz")],
        )

    @classmethod
    def execute(cls, image, sapiens2_model, frames_per_batch) -> io.NodeOutput:
        patcher, task = sapiens2_model
        if task != "pointmap":
            raise ValueError(f"loaded checkpoint is task={task!r}, expected 'pointmap'")
        comfy.model_management.load_model_gpu(patcher)
        x, orig_hw, padding = to_model_input(image, device=patcher.load_device, dtype=torch.float16)
        out_device = comfy.model_management.intermediate_device()
        out_dtype = comfy.model_management.intermediate_dtype()
        chunks = []
        for _, out in _iter_chunks(patcher, x, frames_per_batch, desc="Sapiens2 Pointmap"):
            pm = (out[0] if isinstance(out, tuple) else out).to(torch.float32)
            pm = crop_pad_and_resize(pm, padding, orig_hw[0], orig_hw[1])
            flat = pm.flatten(2)
            mn = flat.quantile(0.01, dim=2)[..., None, None]
            mx = flat.quantile(0.99, dim=2)[..., None, None]
            pm = ((pm - mn) / (mx - mn + 1e-6)).clamp_(0.0, 1.0)
            chunks.append(pm.to(device=out_device, dtype=out_dtype))
        out = torch.cat(chunks, dim=0).movedim(1, -1)
        return io.NodeOutput(out)


class Sapiens2Pose(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Sapiens2Pose",
            display_name="Sapiens2 Pose (308 KP)",
            category="Sapiens2",
            inputs=[
                io.Image.Input("image"),
                io.Custom("SAPIENS2_MODEL").Input("sapiens2_model"),
                io.Combo.Input("output_format", options=["openpose", "raw_308"], default="openpose"),
                io.Boolean.Input("include_face", default=False),
                io.Int.Input("frames_per_batch", default=1, min=1, max=256, tooltip=_FRAMES_PER_BATCH_TOOLTIP),
                io.BoundingBox.Input("bboxes", optional=True, force_input=True,
                    tooltip="Person bboxes (RT-DETR / SDPoseFaceBBoxes / SAM3). "
                            "When given, each person is cropped+warped individually "
                            "(matches upstream Sapiens2 inference). Without bboxes, "
                            "the whole image is resized — only correct when input is "
                            "already a single-person crop."),
            ],
            outputs=[io.Custom("POSE_KEYPOINT").Output(display_name="keypoints")],
        )

    @classmethod
    def execute(cls, image, sapiens2_model, output_format, include_face,
                frames_per_batch, bboxes=None) -> io.NodeOutput:
        patcher, task = sapiens2_model
        if task != "pose":
            raise ValueError(f"loaded checkpoint is task={task!r}, expected 'pose'")

        comfy.model_management.load_model_gpu(patcher)
        device = patcher.load_device
        T = int(image.shape[0])
        cw = int(image.shape[2])
        ch = int(image.shape[1])

        # Bbox-crop path: per-person TopdownAffine → DARK decode → remap.
        if bboxes is not None and (not isinstance(bboxes, list) or len(bboxes) > 0):
            bb_per_frame = _normalize_bboxes(bboxes, T)
            x, mapping = _make_bbox_crops(image, bb_per_frame, 1024, 768, device, torch.float16)
            if x is None:
                empty = [{"canvas_width": cw, "canvas_height": ch, "people": []} for _ in range(T)]
                return io.NodeOutput(empty)

            kps_chunks: list = []
            sc_chunks: list = []
            for _, hm in _iter_chunks(patcher, x, frames_per_batch, desc="Sapiens2 Pose"):
                kps, scores = _decode_heatmaps(hm, 1024, 768)
                kps_chunks.append(kps)
                sc_chunks.append(scores)
                del hm
            kps_np = torch.cat(kps_chunks, dim=0).detach().cpu().to(torch.float32).numpy()
            sc_np = torch.cat(sc_chunks, dim=0).detach().cpu().to(torch.float32).numpy()
            del kps_chunks, sc_chunks

            for c_idx, (_, x1, y1, scale, pad_top, pad_left) in enumerate(mapping):
                kps_np[c_idx, :, 0] = (kps_np[c_idx, :, 0] - pad_left) / scale + x1
                kps_np[c_idx, :, 1] = (kps_np[c_idx, :, 1] - pad_top)  / scale + y1

            frames = [{"canvas_width": cw, "canvas_height": ch, "people": []} for _ in range(T)]
            for c_idx, (f_idx, *_) in enumerate(mapping):
                if output_format == "openpose":
                    frames[f_idx]["people"].append(
                        _person_openpose(kps_np[c_idx], sc_np[c_idx], include_face)
                    )
                else:
                    frames[f_idx]["people"].append(
                        _person_raw_308(kps_np[c_idx], sc_np[c_idx])
                    )
            return io.NodeOutput(frames)

        # Full-image path (single-person crop assumed).
        x, orig_hw, padding = to_model_input(image, device=device, dtype=torch.float16)
        pad_top, pad_bottom, pad_left, pad_right = padding
        sx = orig_hw[1] / (768 - pad_left - pad_right)
        sy = orig_hw[0] / (1024 - pad_top - pad_bottom)

        kps_chunks = []
        sc_chunks = []
        for _, hm in _iter_chunks(patcher, x, frames_per_batch, desc="Sapiens2 Pose"):
            kps, scores = _decode_heatmaps(hm, 1024, 768)
            kps[..., 0] = (kps[..., 0] - pad_left) * sx
            kps[..., 1] = (kps[..., 1] - pad_top) * sy
            kps_chunks.append(kps)
            sc_chunks.append(scores)
            del hm

        kps_np = torch.cat(kps_chunks, dim=0).detach().cpu().to(torch.float32).numpy()
        sc_np = torch.cat(sc_chunks, dim=0).detach().cpu().to(torch.float32).numpy()
        del kps_chunks, sc_chunks

        if output_format == "openpose":
            frames = [_frame_openpose_np(kps_np[b], sc_np[b], cw, ch, include_face)
                      for b in range(kps_np.shape[0])]
        else:
            frames = [_frame_raw_308_np(kps_np[b], sc_np[b], cw, ch)
                      for b in range(kps_np.shape[0])]
        return io.NodeOutput(frames)


class Sapiens2DrawPose(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Sapiens2DrawPose",
            display_name="Sapiens2 Draw Pose (308 KP)",
            category="Sapiens2",
            inputs=[
                io.Custom("POSE_KEYPOINT").Input("keypoints"),
                io.Boolean.Input("draw_skeleton", default=True),
                io.Boolean.Input("draw_points", default=True),
                io.Boolean.Input("draw_face", default=True),
                io.Int.Input("point_radius", default=3, min=1, max=20),
                io.Int.Input("stick_width", default=3, min=1, max=20),
                io.Float.Input("score_threshold", default=0.3, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[io.Image.Output()],
        )

    @classmethod
    def execute(cls, keypoints, draw_skeleton, draw_points, draw_face,
                point_radius, stick_width, score_threshold) -> io.NodeOutput:
        try:
            import cv2
        except ImportError:
            cv2 = None

        out_device = comfy.model_management.intermediate_device()
        out_dtype = comfy.model_management.intermediate_dtype()

        if not keypoints:
            return io.NodeOutput(torch.zeros((1, 64, 64, 3), device=out_device, dtype=out_dtype))

        outputs = []
        for f_idx, frame in enumerate(keypoints):
            H = int(frame["canvas_height"])
            W = int(frame["canvas_width"])
            canvas = np.zeros((H, W, 3), dtype=np.uint8)

            for person in frame["people"]:
                # Need the full 308-keypoint payload (raw_308 format) to draw.
                if "keypoints_2d" not in person:
                    continue
                arr = person["keypoints_2d"]
                pts = [(float(p[0]), float(p[1]), float(p[2])) for p in arr]
                if len(pts) < 308:
                    continue

                if draw_skeleton:
                    for a, b, color in _SAPIENS2_SKELETON:
                        xa, ya, sa = pts[a]
                        xb, yb, sb = pts[b]
                        if sa < score_threshold or sb < score_threshold:
                            continue
                        p1 = (int(round(xa)), int(round(ya)))
                        p2 = (int(round(xb)), int(round(yb)))
                        if cv2 is not None:
                            cv2.line(canvas, p1, p2, color, stick_width, lineType=cv2.LINE_AA)
                        else:
                            from PIL import Image as _PILImage, ImageDraw as _PILDraw
                            tmp = _PILImage.fromarray(canvas)
                            _PILDraw.Draw(tmp).line([p1, p2], fill=color, width=stick_width)
                            canvas = np.asarray(tmp).copy()

                if draw_points:
                    for i, (x, y, s) in enumerate(pts):
                        if s < score_threshold:
                            continue
                        if not draw_face and i >= 70:
                            continue
                        cx, cy = int(round(x)), int(round(y))
                        if not (0 <= cx < W and 0 <= cy < H):
                            continue
                        color = _kp_color(i)
                        if cv2 is not None:
                            cv2.circle(canvas, (cx, cy), point_radius, color, -1, lineType=cv2.LINE_AA)
                        else:
                            r = point_radius
                            canvas[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1] = color

            outputs.append(canvas)

        out_np = np.stack(outputs).astype(np.float32) / 255.0
        return io.NodeOutput(torch.from_numpy(out_np).to(device=out_device, dtype=out_dtype))


class Sapiens2Extension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            Sapiens2Loader,
            Sapiens2Normal,
            Sapiens2Seg,
            Sapiens2Pointmap,
            Sapiens2Pose,
            Sapiens2DrawPose,
        ]


async def comfy_entrypoint() -> Sapiens2Extension:
    return Sapiens2Extension()
