"""Top-level wrapper that mirrors the upstream NormalEstimator / SegEstimator /
PointmapEstimator / PoseTopdownEstimator: a backbone plus a decode_head with
exactly those two attribute names, so released safetensors with
``backbone.*`` / ``decode_head.*`` keys load directly via load_state_dict.
"""

from typing import Optional

import torch.nn as nn
from torch import Tensor

from .backbone import Sapiens2


class Sapiens2Estimator(nn.Module):
    def __init__(self, backbone: Sapiens2, decode_head: Optional[nn.Module]):
        super().__init__()
        self.backbone = backbone
        if decode_head is not None:
            self.decode_head = decode_head

    def forward(self, x: Tensor):
        feat = self.backbone(x)
        if hasattr(self, "decode_head"):
            return self.decode_head(feat)
        return feat
