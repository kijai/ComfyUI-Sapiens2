"""Task heads for Sapiens2"""

from typing import List, Optional, Sequence, Tuple

import torch.nn as nn
from torch import Tensor


def _deconv_padding(kernel: int) -> Tuple[int, int]:
    if kernel == 4:
        return 1, 0
    if kernel == 3:
        return 1, 1
    if kernel == 2:
        return 0, 0
    raise ValueError(f"unsupported deconv kernel {kernel}")


class _DenseUpsampleHead(nn.Module):

    def __init__(
        self,
        in_channels: int,
        upsample_channels: List[int],
        conv_out_channels: Optional[Sequence[int]],
        conv_kernel_sizes: Optional[Sequence[int]],
        out_channels: int,
        out_attr: str,  # "conv_normal" or "conv_pointmap"
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()

        self.input_conv = nn.Sequential(
            operations.Conv2d(in_channels, in_channels, kernel_size=3, padding=1,
                              dtype=dtype, device=device),
            nn.InstanceNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )

        up_blocks: List[nn.Module] = []
        cur = in_channels
        for out_ch in upsample_channels:
            up_blocks.append(nn.Sequential(
                operations.Conv2d(cur, out_ch * 4, kernel_size=3, padding=1,
                                  dtype=dtype, device=device),
                nn.PixelShuffle(2),
                nn.InstanceNorm2d(out_ch),
                nn.SiLU(inplace=True),
            ))
            cur = out_ch
        self.upsample_blocks = nn.Sequential(*up_blocks)

        flat: List[nn.Module] = []
        if conv_out_channels and conv_kernel_sizes:
            for out_ch, k in zip(conv_out_channels, conv_kernel_sizes):
                flat.append(operations.Conv2d(cur, out_ch, kernel_size=k,
                                              padding=(k - 1) // 2,
                                              dtype=dtype, device=device))
                flat.append(nn.InstanceNorm2d(out_ch))
                flat.append(nn.SiLU(inplace=True))
                cur = out_ch
        self.conv_layers = nn.Sequential(*flat)

        setattr(self, out_attr,
                operations.Conv2d(cur, out_channels, kernel_size=1,
                                  dtype=dtype, device=device))
        self._out_attr = out_attr

    def _trunk(self, x: Tensor) -> Tensor:
        return self.conv_layers(self.upsample_blocks(self.input_conv(x)))

    def forward(self, x: Tensor) -> Tensor:
        return getattr(self, self._out_attr)(self._trunk(x))


class NormalHead(_DenseUpsampleHead):
    def __init__(self, in_channels, upsample_channels,
                 conv_out_channels=None, conv_kernel_sizes=None,
                 dtype=None, device=None, operations=None):
        super().__init__(
            in_channels=in_channels, upsample_channels=upsample_channels,
            conv_out_channels=conv_out_channels, conv_kernel_sizes=conv_kernel_sizes,
            out_channels=3, out_attr="conv_normal",
            dtype=dtype, device=device, operations=operations,
        )


class PointmapHead(_DenseUpsampleHead):
    def __init__(self, in_channels, upsample_channels,
                 conv_out_channels=None, conv_kernel_sizes=None,
                 scale_conv_out_channels=(1536, 512, 128),
                 scale_conv_kernel_sizes=(1, 1, 1),
                 scale_final_layer=(48 * 128, 512, 64, 1),
                 dtype=None, device=None, operations=None):
        super().__init__(
            in_channels=in_channels, upsample_channels=upsample_channels,
            conv_out_channels=conv_out_channels, conv_kernel_sizes=conv_kernel_sizes,
            out_channels=3, out_attr="conv_pointmap",
            dtype=dtype, device=device, operations=operations,
        )
        if scale_conv_out_channels is not None:
            scale: List[nn.Module] = []
            cur = in_channels
            for out_ch, k in zip(scale_conv_out_channels, scale_conv_kernel_sizes):
                scale.append(operations.Conv2d(cur, out_ch, kernel_size=k, stride=2,
                                               padding=(k - 1) // 2,
                                               dtype=dtype, device=device))
                scale.append(nn.InstanceNorm2d(out_ch))
                scale.append(nn.SiLU(inplace=True))
                cur = out_ch
            self.scale_conv_layers = nn.Sequential(*scale)

            final: List[nn.Module] = [nn.Flatten()]
            in_features = scale_final_layer[0]
            for i in range(1, len(scale_final_layer)):
                final.append(operations.Linear(in_features, scale_final_layer[i],
                                               dtype=dtype, device=device))
                if i < len(scale_final_layer) - 1:
                    final.append(nn.SiLU())
                in_features = scale_final_layer[i]
            self.scale_final_layer = nn.Sequential(*final)
        else:
            self.scale_conv_layers = None
            self.scale_final_layer = None

    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        pointmap = self.conv_pointmap(self._trunk(x))
        scale = None
        if self.scale_conv_layers is not None:
            scale = self.scale_final_layer(self.scale_conv_layers(x))
        return pointmap, scale


class _DeconvHead(nn.Module):
    """Shared scaffolding for SegHead and PoseHeatmapHead."""

    def __init__(
        self,
        in_channels: int,
        deconv_out_channels: Sequence[int],
        deconv_kernel_sizes: Sequence[int],
        conv_out_channels: Optional[Sequence[int]],
        conv_kernel_sizes: Optional[Sequence[int]],
        out_channels: int,
        out_attr: str,
        dtype=None,
        device=None,
        operations=None,
    ):
        super().__init__()

        deconv: List[nn.Module] = []
        cur = in_channels
        for out_ch, k in zip(deconv_out_channels, deconv_kernel_sizes):
            pad, opad = _deconv_padding(k)
            deconv.append(operations.ConvTranspose2d(
                cur, out_ch, kernel_size=k, stride=2,
                padding=pad, output_padding=opad, bias=False,
                dtype=dtype, device=device,
            ))
            deconv.append(nn.InstanceNorm2d(out_ch))
            deconv.append(nn.SiLU(inplace=True))
            cur = out_ch
        self.deconv_layers = nn.Sequential(*deconv)

        if conv_out_channels and conv_kernel_sizes:
            conv: List[nn.Module] = []
            for out_ch, k in zip(conv_out_channels, conv_kernel_sizes):
                conv.append(operations.Conv2d(cur, out_ch, kernel_size=k, stride=1,
                                              padding=(k - 1) // 2,
                                              dtype=dtype, device=device))
                conv.append(nn.InstanceNorm2d(out_ch))
                conv.append(nn.SiLU(inplace=True))
                cur = out_ch
            self.conv_layers = nn.Sequential(*conv)
        else:
            self.conv_layers = nn.Identity()

        setattr(self, out_attr,
                operations.Conv2d(cur, out_channels, kernel_size=1,
                                  dtype=dtype, device=device))
        self._out_attr = out_attr

    def forward(self, x: Tensor) -> Tensor:
        return getattr(self, self._out_attr)(self.conv_layers(self.deconv_layers(x)))


class SegHead(_DeconvHead):
    def __init__(self, in_channels, num_classes,
                 deconv_out_channels=(256, 256, 256), deconv_kernel_sizes=(4, 4, 4),
                 conv_out_channels=None, conv_kernel_sizes=None,
                 dtype=None, device=None, operations=None):
        super().__init__(
            in_channels=in_channels,
            deconv_out_channels=deconv_out_channels,
            deconv_kernel_sizes=deconv_kernel_sizes,
            conv_out_channels=conv_out_channels,
            conv_kernel_sizes=conv_kernel_sizes,
            out_channels=num_classes, out_attr="conv_seg",
            dtype=dtype, device=device, operations=operations,
        )


class PoseHeatmapHead(_DeconvHead):
    def __init__(self, in_channels, out_channels,
                 deconv_out_channels=(256, 256, 256), deconv_kernel_sizes=(4, 4, 4),
                 conv_out_channels=None, conv_kernel_sizes=None,
                 dtype=None, device=None, operations=None):
        super().__init__(
            in_channels=in_channels,
            deconv_out_channels=deconv_out_channels,
            deconv_kernel_sizes=deconv_kernel_sizes,
            conv_out_channels=conv_out_channels,
            conv_kernel_sizes=conv_kernel_sizes,
            out_channels=out_channels, out_attr="conv_pose",
            dtype=dtype, device=device, operations=operations,
        )
