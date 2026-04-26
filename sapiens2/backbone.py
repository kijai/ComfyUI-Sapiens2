import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import comfy.model_management
from comfy.ldm.modules.attention import optimized_attention

class RopePositionEmbedding(nn.Module):
    """2D image-coordinate rotary positional embedding."""

    def __init__(self, embed_dim: int, *, num_heads: int, device=None):
        super().__init__()
        D_head = embed_dim // num_heads
        self.D_head = D_head
        periods = 100.0 ** (
            2 * torch.arange(D_head // 4, device=device, dtype=torch.bfloat16) / (D_head // 2)
        )
        self.register_buffer("periods", periods, persistent=True)
        self._cache_hw: Optional[Tuple[int, int]] = None
        self._cache: Optional[Tuple[Tensor, Tensor]] = None

    def forward(self, *, H: int, W: int) -> Tuple[Tensor, Tensor]:
        if self._cache_hw == (H, W):
            return self._cache
        dd = {"device": self.periods.device, "dtype": torch.bfloat16}
        coords_h = torch.arange(0.5, H, **dd) / H
        coords_w = torch.arange(0.5, W, **dd) / W
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1
        ).flatten(0, 1)
        coords = 2.0 * coords - 1.0
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]
        angles = angles.flatten(1, 2).tile(2)
        self._cache = (torch.sin(angles), torch.cos(angles))
        self._cache_hw = (H, W)
        return self._cache


class LayerScale(nn.Module):
    """Per-channel learnable scaling. Trained parameter (NOT training-only)"""

    def __init__(self, dim: int, init_value: float = 1e-4, dtype=None, device=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(dim, dtype=dtype, device=device) * init_value
        )

    def forward(self, x: Tensor) -> Tensor:
        w = comfy.model_management.cast_to_device(self.weight, x.device, x.dtype)
        return x.mul_(w)


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dims, patch_size,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.projection = operations.Conv2d(
            in_channels, embed_dims,
            kernel_size=patch_size, stride=patch_size,
            padding=0, bias=True, dtype=dtype, device=device,
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        return x, out_size


class SwiGLUFFN(nn.Module):
    def __init__(self, embed_dims, feedforward_channels,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.w12 = operations.Linear(
            embed_dims, 2 * feedforward_channels, bias=True, dtype=dtype, device=device
        )
        self.w3 = operations.Linear(
            feedforward_channels, embed_dims, bias=True, dtype=dtype, device=device
        )

    def forward(self, x: Tensor, identity: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        # In-place silu+mul on x1 (a view of x12) avoids allocating a fresh
        # (B, N, ffn) tensor for ``F.silu(x1) * x2`` — saves ~60 MB on 5B B=1.
        x1 = F.silu(x1, inplace=True).mul_(x2)
        return identity.add_(self.w3(x1))


class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dims, num_heads, num_kv_heads=None,
                 layer_scale_init_value=0.0,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = embed_dims // num_heads

        self.wq = operations.Linear(embed_dims, embed_dims, bias=True, dtype=dtype, device=device)
        self.wk = operations.Linear(embed_dims, self.num_kv_heads * self.head_dim, bias=True, dtype=dtype, device=device)
        self.wv = operations.Linear(embed_dims, self.num_kv_heads * self.head_dim, bias=True, dtype=dtype, device=device)
        self.q_norm = operations.RMSNorm(self.head_dim, eps=1e-6, dtype=dtype, device=device)
        self.k_norm = operations.RMSNorm(self.head_dim, eps=1e-6, dtype=dtype, device=device)
        self.proj = operations.Linear(embed_dims, embed_dims, bias=True, dtype=dtype, device=device)

        if layer_scale_init_value > 0:
            self.gamma = LayerScale(embed_dims, layer_scale_init_value, dtype=dtype, device=device)
        else:
            self.gamma = nn.Identity()

    def _apply_rope(self, qk: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
        # rotate-half then x*cos + rotate(x)*sin
        x1, x2 = qk.chunk(2, dim=-1)
        rotated = torch.cat([-x2, x1], dim=-1)
        return torch.mul(qk, cos).addcmul_(rotated, sin)

    def forward(self, x: Tensor, rope: Tuple[Tensor, Tensor]) -> Tensor:
        B, N, _ = x.shape
        q = self.wq(x).view(B, N, self.num_heads,    self.head_dim).permute(0, 2, 1, 3)
        k = self.wk(x).view(B, N, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.wv(x).view(B, N, self.num_kv_heads, self.head_dim).permute(0, 2, 1, 3)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if self.num_kv_heads != self.num_heads:
            factor = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(factor, dim=1)
            v = v.repeat_interleave(factor, dim=1)

        sin, cos = rope
        n_extra = q.shape[-2] - sin.shape[-2]
        # apply rope only to patch tokens (skip the n_extra cls+storage prefix)
        q[:, :, n_extra:] = self._apply_rope(q[:, :, n_extra:], sin, cos)
        k[:, :, n_extra:] = self._apply_rope(k[:, :, n_extra:], sin, cos)

        out = optimized_attention(q, k, v, heads=self.num_heads, skip_reshape=True)
        return self.gamma(self.proj(out))


class TransformerEncoderLayer2(nn.Module):
    def __init__(self, embed_dims, num_heads, num_kv_heads,
                 feedforward_channels, layer_scale_init_value,
                 dtype=None, device=None, operations=None):
        super().__init__()
        self.ln1 = operations.RMSNorm(embed_dims, eps=1e-6, dtype=dtype, device=device)
        self.attn = GroupedQueryAttention(
            embed_dims=embed_dims, num_heads=num_heads, num_kv_heads=num_kv_heads,
            layer_scale_init_value=layer_scale_init_value,
            dtype=dtype, device=device, operations=operations,
        )
        self.ln2 = operations.RMSNorm(embed_dims, eps=1e-6, dtype=dtype, device=device)
        self.ffn = SwiGLUFFN(
            embed_dims=embed_dims, feedforward_channels=feedforward_channels,
            dtype=dtype, device=device, operations=operations,
        )

    def forward(self, x: Tensor, rope: Tuple[Tensor, Tensor]) -> Tensor:
        x.add_(self.attn(self.ln1(x), rope))
        return self.ffn(self.ln2(x), identity=x)

ARCHS = {
    "sapiens2_0.4b": dict(embed_dims=1024, num_layers=24, num_heads=16, feedforward_channels=4096),
    "sapiens2_0.8b": dict(embed_dims=1280, num_layers=32, num_heads=16, feedforward_channels=5120),
    "sapiens2_1b":   dict(embed_dims=1536, num_layers=40, num_heads=24, feedforward_channels=6144),
    "sapiens2_5b":   dict(embed_dims=2432, num_layers=56, num_heads=32, feedforward_channels=9728),
}

class Sapiens2(nn.Module):
    """Sapiens2 backbone (inference). Outputs a (B, C, H/16, W/16) feature map."""
    # First/last 8 layers use full multi-head attn; middle layers use GQA with
    # num_kv_heads = num_heads // 2.
    MHSA_EARLY = 8
    MHSA_LATE = 8

    def __init__(self, arch: str = "sapiens2_1b", patch_size: int = 16, in_channels: int = 3, n_storage_tokens: int = 8,
                 layer_scale_init_value: float = 1e-4, dtype=None, device=None, operations=None):
        super().__init__()
        if arch not in ARCHS:
            raise ValueError(f"unknown arch {arch}; pick from {list(ARCHS)}")
        cfg = ARCHS[arch]
        self.embed_dims = cfg["embed_dims"]
        num_layers = cfg["num_layers"]
        num_heads = cfg["num_heads"]
        ffn = cfg["feedforward_channels"]

        self.patch_embed = PatchEmbed(in_channels, self.embed_dims, patch_size, dtype=dtype, device=device, operations=operations)
        self.rope_embed = RopePositionEmbedding(embed_dim=self.embed_dims, num_heads=num_heads, device=device)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims, dtype=dtype, device=device))
        self.storage_tokens = nn.Parameter(
            torch.zeros(1, n_storage_tokens, self.embed_dims, dtype=dtype, device=device)
        ) if n_storage_tokens > 0 else None
        self.num_extra_tokens = 1 + n_storage_tokens

        self.blocks = nn.Sequential()
        for i in range(num_layers):
            in_band = i < self.MHSA_EARLY or i >= num_layers - self.MHSA_LATE
            self.blocks.append(TransformerEncoderLayer2(
                embed_dims=self.embed_dims, num_heads=num_heads,
                num_kv_heads=None if in_band else num_heads // 2,
                feedforward_channels=ffn,
                layer_scale_init_value=layer_scale_init_value,
                dtype=dtype, device=device, operations=operations,
            ))

        self.ln1 = operations.RMSNorm(self.embed_dims, eps=1e-6, dtype=dtype, device=device)

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        x, hw = self.patch_embed(x)

        cls = comfy.model_management.cast_to_device(self.cls_token, x.device, x.dtype).expand(B, -1, -1)
        prepend = [cls]
        if self.storage_tokens is not None:
            st = comfy.model_management.cast_to_device(self.storage_tokens, x.device, x.dtype).expand(B, -1, -1)
            prepend.append(st)
        x = torch.cat(prepend + [x], dim=1)

        sin, cos = self.rope_embed(H=hw[0], W=hw[1])
        rope = (sin.to(x.dtype), cos.to(x.dtype))
        for layer in self.blocks:
            x = layer(x, rope)
        x = self.ln1(x)

        # featmap: drop extra tokens, reshape (B, N, C) -> (B, C, H, W)
        patch = x[:, self.num_extra_tokens :]
        return patch.reshape(B, hw[0], hw[1], self.embed_dims).permute(0, 3, 1, 2)
