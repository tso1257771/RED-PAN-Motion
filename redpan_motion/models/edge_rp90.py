"""
EdgeRP90 — the edge-deployable 90 s multi-task seismic model (``edge_rp90_v1``).

A partial/depthwise-separable 1-D encoder → dilated-TCN neck → decoder with
**additive** skips, **neck-only** squeeze-excitation, and a cheap **sign-split**
raw-Z polarity branch. Same task and output contract as the production
``MTAN_R2UNet_RP90_Motion`` (v49):

  - Picker:   (B, 3, T) — P / S / Noise           (softmax)
  - Polarity: (B, 3, T) — first-motion [N, U, D]   (softmax)
  - Detector: (B, 2, T) — event / no-event mask    (softmax)

The shipped checkpoint was trained with the same supervised recipe as
``redpan_motion``; only the architecture differs. Its job is to be small and
INT8-friendly without costing accuracy or latency.

Key choices:
  - **PConv** (partial convolution, FasterNet) blocks by default — 1-D tiny-channel
    *depthwise* conv is memory-bound on ARM CPUs, so a small dense conv on a channel
    slice is faster there and quantizes with fewer foot-guns. ``block="dw"`` builds
    the identical macro-architecture with depthwise blocks, purely for the on-device
    latency A/B.
  - **Additive** decoder skips (concat copies thrash a Pi's cache).
  - Sign-split raw-Z polarity: ``[ReLU(z), ReLU(-z)]`` preserves first-motion sign
    explicitly while costing ~35–50× less than redpan_motion's full-resolution
    polarity head.
  - An early-exit **gate head** off the neck (129 params). It is trained with the
    rest of the model, and nothing uses it at inference.

Config: ``model_type = "edge_rp90_v1"``.
"""
from __future__ import annotations

import logging
import pickle
from math import prod
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────
def _partial_channels(ch: int, div: int = 4) -> int:
    """PConv partial-channel count: ``ch/div`` rounded to nearest 8, clamped to
    ``[8, ch]``.  (24→8, 40→8, 64→16, 96→24, 128→32, 12→8.)"""
    cp = int(round(ch / div / 8.0)) * 8
    return max(8, min(cp, ch))


class _SE(nn.Module):
    """Squeeze-excitation with a hard-sigmoid gate (piecewise-linear → quantizes
    cleanly, unlike a plain sigmoid)."""

    def __init__(self, ch: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, ch // reduction)
        self.fc1 = nn.Conv1d(ch, hidden, 1)
        self.fc2 = nn.Conv1d(hidden, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=2, keepdim=True)
        s = F.relu(self.fc1(s), inplace=True)
        s = self.fc2(s)
        s = torch.clamp(0.2 * s + 0.5, 0.0, 1.0)  # hard-sigmoid
        return x * s


class _ResBlock(nn.Module):
    """Residual separable block.

    ``block="pconv"``: partial conv on the first ``Cp`` channels (FasterNet PConv),
    the rest passed through untouched. ``block="dw"``: depthwise conv over all
    channels. Order in both cases::

        (P/DW)Conv → BN → ReLU → PW(1×1) → BN → [SE] → [Dropout] → +identity → ReLU
    """

    def __init__(self, ch: int, kernel: int = 7, dilation: int = 1,
                 block: str = "pconv", dropout: float = 0.0, use_se: bool = False):
        super().__init__()
        pad = (kernel - 1) // 2 * dilation
        self.block = block
        if block == "pconv":
            self.cp = _partial_channels(ch)
            self.spatial = nn.Conv1d(self.cp, self.cp, kernel, padding=pad,
                                     dilation=dilation, bias=False)
        elif block == "dw":
            self.cp = ch
            self.spatial = nn.Conv1d(ch, ch, kernel, padding=pad,
                                     dilation=dilation, groups=ch, bias=False)
        else:
            raise ValueError(f"block must be 'pconv'/'dw'; got {block!r}")
        self.bn1 = nn.BatchNorm1d(ch)
        self.pw = nn.Conv1d(ch, ch, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(ch)
        self.se = _SE(ch) if use_se else None
        self.drop = nn.Dropout(dropout) if dropout > 0 else None
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        if self.block == "pconv":
            out = torch.cat([self.spatial(x[:, :self.cp]), x[:, self.cp:]], dim=1)
        else:
            out = self.spatial(x)
        out = self.act(self.bn1(out))
        out = self.bn2(self.pw(out))
        if self.se is not None:
            out = self.se(out)
        if self.drop is not None:
            out = self.drop(out)
        return F.relu(out + identity)


class _DownProj(nn.Module):
    """Depthwise strided conv → pointwise projection (kernel wider than stride for
    anti-aliasing)."""

    def __init__(self, cin: int, cout: int, kernel: int, stride: int):
        super().__init__()
        pad = (kernel - 1) // 2
        self.dw = nn.Conv1d(cin, cin, kernel, stride=stride, padding=pad,
                            groups=cin, bias=False)
        self.bn1 = nn.BatchNorm1d(cin)
        self.pw = nn.Conv1d(cin, cout, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.dw(x)), inplace=True)
        x = F.relu(self.bn2(self.pw(x)), inplace=True)
        return x


def _match_len(x: torch.Tensor, tgt: int) -> torch.Tensor:
    """Crop or zero-pad ``x`` along the time axis to length ``tgt`` (guards the
    additive skips against any off-by-one from strided-conv/upsample rounding)."""
    t = x.shape[-1]
    if t > tgt:
        return x[..., :tgt]
    if t < tgt:
        return F.pad(x, (0, tgt - t))
    return x


class _UpStage(nn.Module):
    """Decoder stage: nearest-upsample → 1×1 project → **add** encoder skip → fuse."""

    def __init__(self, cin: int, cout: int, scale: int, kernel: int,
                 block: str, dropout: float = 0.0):
        super().__init__()
        self.scale = scale
        self.proj = nn.Sequential(
            nn.Conv1d(cin, cout, 1, bias=False),
            nn.BatchNorm1d(cout),
            nn.ReLU(inplace=True),
        )
        self.fuse = _ResBlock(cout, kernel, 1, block, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Project channels down at the LOWER resolution, then upsample — the same
        # result as project-after-upsample but far cheaper (the 1×1 runs at 1/scale
        # the samples). Keeps the additive skip and per-stage fuse block intact.
        x = self.proj(x)
        x = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        x = _match_len(x, skip.shape[-1])
        return self.fuse(x + skip)


# ──────────────────────────────────────────────────────────────────────────
# Polarity heads
# ──────────────────────────────────────────────────────────────────────────
class _SignSplitPolarityHead(nn.Module):
    """Cheap polarity head. Parameter-free sign split ``[ReLU(z), ReLU(-z)]`` keeps
    first-motion sign explicit; conditioned on the picker feature (where P is)."""

    def __init__(self, ps_ch: int, mid: int = 12, out_ch: int = 3,
                 block: str = "pconv", dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, mid, 9, padding=4, bias=False),
            nn.BatchNorm1d(mid), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList(
            [_ResBlock(mid, 9, d, block) for d in (1, 2, 4)])
        self.ps_proj = nn.Conv1d(ps_ch, mid, 1)
        self.post = _ResBlock(mid, 7, 1, block, dropout=dropout)
        self.head = nn.Conv1d(mid, out_ch, 1)

    def forward(self, z_raw: torch.Tensor, ps_feat: torch.Tensor) -> torch.Tensor:
        h = self.stem(torch.cat([F.relu(z_raw), F.relu(-z_raw)], dim=1))
        for b in self.blocks:
            h = b(h)
        h = self.post(h + self.ps_proj(ps_feat))
        return self.head(h)


class _FullResSepPolarityHead(nn.Module):
    """Fallback (Design-A style): depthwise-separable head reading raw Z at full
    resolution. Heavier but keeps the physical sign; enabled via
    ``polarity_head="fullres_sep"``."""

    def __init__(self, ps_ch: int, mid: int = 24, out_ch: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, mid, 7, padding=3, bias=False),
            nn.BatchNorm1d(mid), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList(
            [_ResBlock(mid, k, 1, "dw") for k in (7, 5, 3)])
        self.ps_proj = nn.Conv1d(ps_ch, mid, 1)
        self.post = _ResBlock(mid, 3, 1, "dw", dropout=dropout)
        self.head = nn.Conv1d(mid, out_ch, 1)

    def forward(self, z_raw: torch.Tensor, ps_feat: torch.Tensor) -> torch.Tensor:
        h = self.stem(z_raw)
        for b in self.blocks:
            h = b(h)
        h = self.post(h + self.ps_proj(ps_feat))
        return self.head(h)


# ──────────────────────────────────────────────────────────────────────────
# The model
# ──────────────────────────────────────────────────────────────────────────
class EdgeRP90(nn.Module):
    """Edge-deployable 90 s multi-task seismic model. See module docstring.

    Args:
        input_size: (T, C) — only C is used to size the stem (default (9000, 3)).
        block: ``"pconv"`` (default) or ``"dw"`` residual-block primitive.
        polarity_head: ``"sign_split"`` (default) or ``"fullres_sep"``.
        polarity_output_channels: 3 = softmax [N,U,D] (default); 1 = signed tanh.
        dropout_rate: dropout in the full-resolution blocks (stem / D0 / heads).
        pad_mode: length-alignment padding mode (reflect/replicate/constant).
    """

    # Encoder downsampling strides (product = alignment factor).
    _STRIDES: Tuple[int, ...] = (2, 3, 5, 2)
    _CHANNELS: Tuple[int, ...] = (24, 40, 64, 96, 128)  # e0,e1,e2,e3,neck
    CH_Z = 2

    def __init__(
        self,
        input_size: Tuple[int, int] = (9000, 3),
        block: str = "pconv",
        polarity_head: str = "sign_split",
        polarity_output_channels: int = 3,
        dropout_rate: float = 0.1,
        pad_mode: str = "reflect",
    ):
        super().__init__()
        if block not in ("pconv", "dw"):
            raise ValueError(f"block must be 'pconv'/'dw'; got {block!r}")
        if polarity_head not in ("sign_split", "fullres_sep"):
            raise ValueError(
                f"polarity_head must be 'sign_split'/'fullres_sep'; got {polarity_head!r}")
        if polarity_output_channels not in (1, 2, 3):
            raise ValueError(
                f"polarity_output_channels must be 1/2/3; got {polarity_output_channels}")
        if pad_mode not in ("reflect", "replicate", "constant"):
            raise ValueError(f"pad_mode must be reflect/replicate/constant; got {pad_mode!r}")

        self.block = block
        self.polarity_output_channels = polarity_output_channels
        self.pad_mode = pad_mode
        self.align_factor = int(prod(self._STRIDES))  # 60
        c0, c1, c2, c3, cn = self._CHANNELS
        dp = dropout_rate

        # ===== Encoder =====
        self.stem = nn.Sequential(
            nn.Conv1d(input_size[1], c0, 11, padding=5, bias=False),
            nn.BatchNorm1d(c0), nn.ReLU(inplace=True))
        self.stem_blocks = nn.Sequential(  # e0, full-res → dropout
            _ResBlock(c0, 7, 1, block, dropout=dp),
            _ResBlock(c0, 7, 1, block, dropout=dp))

        self.down1 = _DownProj(c0, c1, 9, 2)
        self.enc1_blocks = nn.Sequential(
            _ResBlock(c1, 7, 1, block), _ResBlock(c1, 7, 2, block))
        self.down2 = _DownProj(c1, c2, 9, 3)
        self.enc2_blocks = nn.Sequential(
            _ResBlock(c2, 7, 1, block), _ResBlock(c2, 7, 2, block))
        self.down3 = _DownProj(c2, c3, 7, 5)
        self.enc3_blocks = nn.Sequential(
            _ResBlock(c3, 5, 1, block), _ResBlock(c3, 5, 2, block))
        self.down4 = _DownProj(c3, cn, 5, 2)

        # ===== Neck (dilated TCN @ T/60), SE after the wide-dilation blocks =====
        neck_dils = (1, 2, 4, 8, 16, 32, 1, 2)
        self.neck = nn.Sequential(*[
            _ResBlock(cn, 5, d, block, use_se=(d in (4, 16, 32)))
            for d in neck_dils])
        # NOTE: the early-exit "gate head" (window-eventness) is intentionally NOT
        # built here. In the plain (non-distillation) training path it would receive
        # no gradient and trip DDP's unused-parameter check. It returns with the
        # distillation phase (redpan_motion/training/distill.py), which supervises it.

        # ===== Decoder (additive skips), upsizes = reversed strides =====
        self.up0 = _UpStage(cn, c3, 2, 5, block)            # →T/30, +e3
        self.up1 = _UpStage(c3, c2, 5, 7, block)            # →T/6,  +e2
        self.up2 = _UpStage(c2, c1, 3, 7, block)            # →T/2,  +e1
        self.up3 = _UpStage(c1, c0, 2, 7, block, dropout=dp)  # →T,   +e0

        # ===== Heads =====
        self.picker_pre = _ResBlock(c0, 7, 1, block, dropout=dp)
        self.picker_head = nn.Conv1d(c0, 3, 1)
        self.detector_pre = _ResBlock(c0, 15, 1, block, dropout=dp)
        self.detector_head = nn.Conv1d(c0, 2, 1)
        if polarity_head == "sign_split":
            self.polarity = _SignSplitPolarityHead(
                c0, out_ch=polarity_output_channels, block=block, dropout=dp)
        else:
            self.polarity = _FullResSepPolarityHead(
                c0, out_ch=polarity_output_channels, dropout=dp)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    # ── length alignment (pad up to a multiple of align_factor, crop back) ──
    def _pad_to_aligned(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        t_orig = x.shape[-1]
        pad = (self.align_factor - t_orig % self.align_factor) % self.align_factor
        if pad == 0:
            return x, t_orig
        if self.pad_mode == "constant":
            return F.pad(x, (0, pad), mode="constant", value=0.0), t_orig
        max_native = x.shape[-1] - 1 if self.pad_mode == "reflect" else x.shape[-1]
        if pad <= max_native:
            return F.pad(x, (0, pad), mode=self.pad_mode), t_orig
        first = max(max_native, 0)
        if first > 0:
            x = F.pad(x, (0, first), mode=self.pad_mode)
        return F.pad(x, (0, pad - first), mode="constant", value=0.0), t_orig

    def forward(self, x: torch.Tensor, z_raw: Optional[torch.Tensor] = None):
        """Forward pass.

        Args:
            x: (B, 3, T) normalized ENZ input.
            z_raw: (B, 1, T) optional UNPROCESSED Z for polarity (falls back to
                the normalized Z channel of ``x``).

        Returns:
            ``(picker, polarity, detector)`` — all maps (B, C, T) cropped to the input
            length; ``polarity`` softmax over [N, U, D] (or signed tanh if C==1).
        """
        if x.dim() != 3:
            raise ValueError(f"expected (B, C, T); got {tuple(x.shape)}")
        x_p, t_orig = self._pad_to_aligned(x)
        z_p = None
        if z_raw is not None:
            if z_raw.shape[-1] != x.shape[-1]:
                raise ValueError(
                    f"z_raw length ({z_raw.shape[-1]}) must match x ({x.shape[-1]})")
            z_p, _ = self._pad_to_aligned(z_raw)
        out = self._forward_core(x_p, z_p)

        def _crop(t):
            if not isinstance(t, torch.Tensor) or t.dim() < 1 or t.shape[-1] < t_orig:
                return t
            return t[..., :t_orig].contiguous()

        return tuple(_crop(o) for o in out)

    def _forward_core(self, x: torch.Tensor, z_raw: Optional[torch.Tensor]):
        e0 = self.stem_blocks(self.stem(x))       # c0 @ T
        e1 = self.enc1_blocks(self.down1(e0))      # c1 @ T/2
        e2 = self.enc2_blocks(self.down2(e1))      # c2 @ T/6
        e3 = self.enc3_blocks(self.down3(e2))      # c3 @ T/30
        neck = self.neck(self.down4(e3))           # cn @ T/60

        d = self.up0(neck, e3)                     # c3 @ T/30
        d = self.up1(d, e2)                        # c2 @ T/6
        d = self.up2(d, e1)                        # c1 @ T/2
        d = self.up3(d, e0)                        # c0 @ T

        pf = self.picker_pre(d)
        picker = F.softmax(self.picker_head(pf).float(), dim=1)     # (B, 3, T)
        detector = F.softmax(
            self.detector_head(self.detector_pre(d)).float(), dim=1)  # (B, 2, T)

        z_ch = z_raw if z_raw is not None else x[:, self.CH_Z:self.CH_Z + 1, :]
        pol_logits = self.polarity(z_ch, pf).float()
        if self.polarity_output_channels == 1:
            polarity = torch.tanh(pol_logits)
        else:
            polarity = F.softmax(pol_logits, dim=1)                 # (B, 3, T)

        return picker, polarity, detector

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_edge_rp90(
    input_size: Tuple[int, int] = (9000, 3),
    block: str = "pconv",
    polarity_head: str = "sign_split",
    polarity_output_channels: int = 3,
    dropout_rate: float = 0.1,
    pad_mode: str = "reflect",
    pretrained_weights: Optional[str] = None,
    **_ignored,
) -> EdgeRP90:
    """Build the EdgeRP90 model. Extra kwargs are ignored so the same config dict
    that builds the v49 model (with its polarity-stream keys) can be reused."""
    model = EdgeRP90(
        input_size=input_size,
        block=block,
        polarity_head=polarity_head,
        polarity_output_channels=polarity_output_channels,
        dropout_rate=dropout_rate,
        pad_mode=pad_mode,
    )
    if pretrained_weights is not None:
        try:
            raw = torch.load(pretrained_weights, map_location="cpu", weights_only=True)
        except (pickle.UnpicklingError, RuntimeError) as exc:
            logger.warning(
                "weights_only=True load failed for %s (%s); retrying with "
                "weights_only=False (only load checkpoints you trust)",
                pretrained_weights, exc)
            raw = torch.load(pretrained_weights, map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and "model_state_dict" in raw:
            sd = raw["model_state_dict"]
        elif isinstance(raw, dict) and "state_dict" in raw:
            sd = raw["state_dict"]
        else:
            sd = raw
        md = model.state_dict()
        matched = {k: v for k, v in sd.items() if k in md and v.shape == md[k].shape}
        md.update(matched)
        model.load_state_dict(md)
        logger.info("Loaded %d/%d layers from pretrained", len(matched), len(md))
    return model
