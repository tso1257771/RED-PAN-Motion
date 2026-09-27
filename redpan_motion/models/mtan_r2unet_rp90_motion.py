"""
MTAN R2U-Net RP90-Motion — the 90-second multi-task seismic model.

A variable-depth MTAN (Multi-Task Attention Network) over a Recurrent-Residual
U-Net (R2U-Net) backbone, with built-in arbitrary-length input handling.

Three output heads:
  - Picker:   (B, 3, T) — P / S / Noise (softmax)
  - Detector: (B, 2, T) — event / no-event mask (softmax)
  - Polarity: (B, C, T) — first-motion at the P arrival; C = polarity_output_channels
              (3 = softmax over [N, U, D], the default; 1 = signed tanh)

The polarity stream seeds from the (optionally raw) Z component, runs a dedicated
residual head, and concatenates the picker's P/S attention features so the head
has direct access to P-location knowledge.

Config: model_type = "mtan_r2unet_rp90_motion"
"""
from __future__ import annotations

import logging
import pickle
from math import prod
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from redpan_motion.models.mtan_r2unet import (
    RRConvUnit, UpConvUnit, MTANBlock,
)

logger = logging.getLogger(__name__)


class _PolarityResBlock(nn.Module):
    """Residual block for the polarity head. Keeps temporal resolution (no
    pooling); the skip connection lets it degrade gracefully to identity."""

    def __init__(self, ch: int, kernel: int = 7, dropout: float = 0.1):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Sequential(
            nn.Conv1d(ch, ch, kernel, padding=pad, bias=False),
            nn.BatchNorm1d(ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(ch, ch, kernel, padding=pad, bias=False),
            nn.BatchNorm1d(ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.conv(x))


class MTAN_R2UNet_RP90_Motion(nn.Module):
    """MTAN R2U-Net with mixed strides, a polarity head, and built-in
    length-flexible inference.

    Any input length ``T`` is accepted: internally padded up to the next multiple
    of the encoder cumulative stride before the encoder, then cropped back to ``T``
    after the decoder.

    Args:
        input_size: (T, C), e.g. (9000, 3).
        nb_filters: per-level channel counts; ``len == depth`` (e.g. [6,12,18,24,32]).
        strides: per-level downsampling stride; ``len == depth`` (e.g. [5,5,3,3,2]).
        kernel_size: conv kernel (default 7).
        dropout_rate: dropout probability.
        rrconv_iters: recurrent iterations per RRConv.
        use_polarity: build the polarity stream + head.
        polarity_output_channels: 3 = softmax [N,U,D] (default); 1 = signed tanh.
        pol_stream_width_mult: widen the Z-only polarity input stream by this
            factor (``pol_stream_ch = nb_filters[0] * mult``).
        pol_init_rrconv_iters: RRConv depth of the polarity input conv
            (defaults to ``rrconv_iters``).
        pol_head_ps_att_width: width the picker-side attention is projected to
            before being concatenated into the polarity head (default 16; set
            ``None`` to concat at ``nb_filters[0]`` with no projection).
        pad_mode: length-alignment padding mode (reflect/replicate/constant).
    """

    CH_Z = 2

    def __init__(
        self,
        input_size: Tuple[int, int] = (9000, 3),
        nb_filters: Optional[List[int]] = None,
        strides: Optional[List[int]] = None,
        kernel_size: int = 7,
        dropout_rate: float = 0.1,
        rrconv_iters: int = 2,
        use_polarity: bool = True,
        polarity_output_channels: int = 3,
        pol_stream_width_mult: int = 3,
        pol_init_rrconv_iters: Optional[int] = 4,
        pol_head_ps_att_width: Optional[int] = 16,
        picker_classes: int = 3,
        detector_classes: int = 2,
        pad_mode: str = "reflect",
    ):
        super().__init__()
        if nb_filters is None:
            nb_filters = [6, 12, 18, 24, 32]
        if strides is None:
            strides = [5, 5, 3, 3, 2]
        assert len(nb_filters) == len(strides), \
            f"nb_filters ({len(nb_filters)}) must match strides ({len(strides)})"
        assert polarity_output_channels in (1, 2, 3), \
            f"polarity_output_channels must be 1/2/3, got {polarity_output_channels}"
        if pad_mode not in ("reflect", "replicate", "constant"):
            raise ValueError(
                f"pad_mode must be reflect/replicate/constant; got {pad_mode!r}")

        self.nb_filters = nb_filters
        self.strides = strides
        self.depth = len(strides)
        self.use_polarity = use_polarity
        self.polarity_output_channels = polarity_output_channels
        self.upsizes = list(reversed(strides))
        self.pad_mode = pad_mode
        # Encoder uses strides[:-1]; the last stride is consumed by the first
        # decoder upconv. Aligning input to this product guarantees clean integer
        # T at every encoder level.
        self.align_factor = (int(prod(strides[:-1])) if len(strides) > 1
                             else int(strides[0]))
        in_ch = input_size[1]

        # ===== Initial (no downsampling) =====
        self.init_rrconv = RRConvUnit(
            in_ch, nb_filters[0], kernel_size, 1, dropout_rate, rrconv_iters)
        self.init_ps_mtan = MTANBlock(
            nb_filters[0], nb_filters[0], kernel_size, 1, strides[0], 'down', dropout_rate)
        self.init_mask_mtan = MTANBlock(
            nb_filters[0], nb_filters[0], kernel_size, 1, strides[0], 'down', dropout_rate)

        if use_polarity:
            assert pol_stream_width_mult >= 1, \
                f"pol_stream_width_mult must be >= 1, got {pol_stream_width_mult}"
            self.pol_stream_width_mult = int(pol_stream_width_mult)
            self.pol_stream_ch = nb_filters[0] * self.pol_stream_width_mult
            self.pol_init_rrconv_iters = (int(pol_init_rrconv_iters)
                                          if pol_init_rrconv_iters is not None
                                          else rrconv_iters)
            # PhaseNet+ style "encoder_polarity": conv on raw Z at input resolution.
            self.pol_init_rrconv = RRConvUnit(
                1, self.pol_stream_ch, kernel_size, 1, dropout_rate,
                self.pol_init_rrconv_iters)

        # ===== Encoder =====
        self.enc_exp_convs = nn.ModuleList()
        self.enc_down_convs = nn.ModuleList()
        self.enc_ps_mtans = nn.ModuleList()
        self.enc_mask_mtans = nn.ModuleList()
        for i in range(self.depth - 1):
            self.enc_exp_convs.append(
                RRConvUnit(nb_filters[i], nb_filters[i], kernel_size, 1,
                           dropout_rate, rrconv_iters))
            self.enc_down_convs.append(
                RRConvUnit(nb_filters[i], nb_filters[i + 1], kernel_size,
                           strides[i], dropout_rate, rrconv_iters))
            ref_ch = nb_filters[i - 1] if i > 0 else nb_filters[0]
            self.enc_ps_mtans.append(
                MTANBlock(nb_filters[i], ref_ch, kernel_size, strides[i],
                          strides[i], 'down', dropout_rate))
            self.enc_mask_mtans.append(
                MTANBlock(nb_filters[i], ref_ch, kernel_size, strides[i],
                          strides[i], 'down', dropout_rate))

        # ===== Bottleneck =====
        self.bottleneck = RRConvUnit(
            nb_filters[-1], nb_filters[-1], kernel_size, 1, dropout_rate, rrconv_iters)

        # ===== Decoder =====
        self.dec_upconvs = nn.ModuleList()
        self.dec_fuse_convs = nn.ModuleList()
        self.dec_ps_mtans = nn.ModuleList()
        self.dec_mask_mtans = nn.ModuleList()
        for i in range(self.depth):
            in_dec = nb_filters[-1] if i == 0 else nb_filters[self.depth - i]
            out_dec = nb_filters[self.depth - 1 - i]
            up_i = self.upsizes[i]
            self.dec_upconvs.append(
                UpConvUnit(in_dec, out_dec, kernel_size, up_i, dropout_rate))
            self.dec_fuse_convs.append(
                RRConvUnit(out_dec * 2, out_dec, kernel_size, 1, dropout_rate, rrconv_iters))
            ref_ch = nb_filters[-2] if i == 0 else nb_filters[self.depth - i]
            self.dec_ps_mtans.append(
                MTANBlock(out_dec, ref_ch, kernel_size, 1, up_i, 'up', dropout_rate))
            self.dec_mask_mtans.append(
                MTANBlock(out_dec, ref_ch, kernel_size, 1, up_i, 'up', dropout_rate))

        # ===== Output heads =====
        self.picker_head = nn.Conv1d(nb_filters[0], picker_classes, 1)
        nn.init.kaiming_uniform_(self.picker_head.weight, nonlinearity='relu')
        nn.init.zeros_(self.picker_head.bias)
        self.detector_head = nn.Conv1d(nb_filters[0], detector_classes, 1)
        nn.init.kaiming_uniform_(self.detector_head.weight, nonlinearity='relu')
        nn.init.zeros_(self.detector_head.bias)

        if use_polarity:
            base_pol_ch = self.pol_stream_ch
            # Project the picker-side attention to pol_head_ps_att_width before
            # concatenating it into the polarity head.
            if (pol_head_ps_att_width is not None
                    and int(pol_head_ps_att_width) != nb_filters[0]):
                self.ps_att_proj = nn.Conv1d(nb_filters[0], int(pol_head_ps_att_width), 1)
                nn.init.kaiming_uniform_(self.ps_att_proj.weight, nonlinearity='relu')
                nn.init.zeros_(self.ps_att_proj.bias)
                ps_extra = int(pol_head_ps_att_width)
            else:
                self.ps_att_proj = None
                ps_extra = nb_filters[0]
            pol_head_in = base_pol_ch + ps_extra
            # Residual polarity head: 3 ResBlocks (kernels 7→5→3) before the 1×1.
            self.polarity_pre_head = nn.Sequential(
                nn.Conv1d(pol_head_in, pol_head_in, 7, padding=3, bias=False),
                nn.BatchNorm1d(pol_head_in),
                nn.ReLU(inplace=True),
                _PolarityResBlock(pol_head_in, kernel=7, dropout=dropout_rate),
                _PolarityResBlock(pol_head_in, kernel=5, dropout=dropout_rate),
                _PolarityResBlock(pol_head_in, kernel=3, dropout=dropout_rate),
            )
            self.polarity_head = nn.Conv1d(pol_head_in, polarity_output_channels, 1)
            nn.init.xavier_uniform_(self.polarity_head.weight)
            nn.init.zeros_(self.polarity_head.bias)

    # ──────────────────────────────────────────────────────────────────
    # Length alignment
    # ──────────────────────────────────────────────────────────────────
    def _pad_to_aligned(self, x: torch.Tensor) -> tuple:
        t_orig = x.shape[-1]
        rem = t_orig % self.align_factor
        pad = (self.align_factor - rem) % self.align_factor
        if pad == 0:
            return x, t_orig
        if self.pad_mode == "constant":
            x = F.pad(x, (0, pad), mode="constant", value=0.0)
        else:
            max_native = x.shape[-1] - 1 if self.pad_mode == "reflect" else x.shape[-1]
            if pad <= max_native:
                x = F.pad(x, (0, pad), mode=self.pad_mode)
            else:
                first = max(max_native, 0)
                rest = pad - first
                if first > 0:
                    x = F.pad(x, (0, first), mode=self.pad_mode)
                x = F.pad(x, (0, rest), mode="constant", value=0.0)
        return x, t_orig

    def forward(self, x: torch.Tensor, z_raw: Optional[torch.Tensor] = None):
        """Forward pass.

        Args:
            x: (B, 3, T) normalized ENZ input for backbone / picker / detector.
            z_raw: (B, 1, T) optional UNPROCESSED Z for the polarity stream
                   (preserves first-motion sign). Falls back to ``x[:, CH_Z]``.

        Returns:
            (picker, polarity, detector). ``polarity`` is ``None`` when
            ``use_polarity=False``.
        """
        if x.dim() != 3:
            raise ValueError(f"expected (B, C, T); got {tuple(x.shape)}")
        x_padded, t_orig = self._pad_to_aligned(x)
        z_padded = None
        if z_raw is not None:
            if z_raw.shape[-1] != x.shape[-1]:
                raise ValueError(
                    f"z_raw length ({z_raw.shape[-1]}) must match x ({x.shape[-1]})")
            z_padded, _ = self._pad_to_aligned(z_raw)
        out = self._forward_core(x_padded, z_raw=z_padded)

        def _crop(t):
            if not isinstance(t, torch.Tensor) or t.dim() < 1:
                return t
            return t[..., :t_orig].contiguous() if t.shape[-1] >= t_orig else t

        return tuple(_crop(o) for o in out)

    def _forward_core(self, x: torch.Tensor, z_raw: Optional[torch.Tensor] = None):
        input_len = x.shape[2]

        # ===== Initial =====
        e0 = self.init_rrconv(x)
        ps_att = self.init_ps_mtan(e0, e0)
        mask_att = self.init_mask_mtan(e0, e0)
        if self.use_polarity:
            # Raw Z preserves the physical first-motion sign that per-channel
            # normalization may alter; fall back to the normalized Z in x.
            z_ch = z_raw if z_raw is not None else x[:, self.CH_Z:self.CH_Z + 1, :]
            pol_att = self.pol_init_rrconv(z_ch)  # stays at input resolution

        enc_feats = [e0]
        ps_atts = [ps_att]
        mask_atts = [mask_att]

        # ===== Encoder =====
        feat = e0
        for i in range(self.depth - 1):
            exp = self.enc_exp_convs[i](feat)
            feat = self.enc_down_convs[i](feat)
            enc_feats.append(feat)
            ps_att = self.enc_ps_mtans[i](exp, ps_att)
            ps_atts.append(ps_att)
            mask_att = self.enc_mask_mtans[i](exp, mask_att)
            mask_atts.append(mask_att)

        # ===== Bottleneck =====
        feat = self.bottleneck(feat)

        # ===== Decoder (chained) =====
        ps_att = ps_atts[-1]
        mask_att = mask_atts[-1]
        for i in range(self.depth):
            skip = enc_feats[-(i + 1)]
            up = self.dec_upconvs[i](feat, skip)
            feat = self.dec_fuse_convs[i](up)
            ps_att = self.dec_ps_mtans[i](feat, ps_att)
            mask_att = self.dec_mask_mtans[i](feat, mask_att)

        # ===== Match length =====
        def _match(t, tgt):
            if t.shape[2] > tgt:
                return t[:, :, :tgt]
            if t.shape[2] < tgt:
                return F.pad(t, (0, tgt - t.shape[2]))
            return t

        ps_att = _match(ps_att, input_len)
        mask_att = _match(mask_att, input_len)

        # ===== Heads =====
        picker = F.softmax(self.picker_head(ps_att).float(), dim=1)        # (B, 3, T)
        detector = F.softmax(self.detector_head(mask_att).float(), dim=1)  # (B, 2, T)

        if not self.use_polarity:
            return picker, None, detector

        pol_att = _match(pol_att, input_len)
        # Concat picker's attention into the head input, giving polarity direct
        # access to P-location knowledge.
        ps_cond = ps_att if self.ps_att_proj is None else self.ps_att_proj(ps_att)
        head_in = torch.cat([pol_att, ps_cond], dim=1)
        head_in = self.polarity_pre_head(head_in)
        pol_logits = self.polarity_head(head_in).float()
        if self.polarity_output_channels == 1:
            polarity = torch.tanh(pol_logits)            # (B, 1, T), signed
        else:
            polarity = F.softmax(pol_logits, dim=1)      # (B, 2/3, T)
        return picker, polarity, detector

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ──────────────────────────────────────────────────────────────────
    # Receptive-field analysis (optional helpers, specific to this backbone)
    # ──────────────────────────────────────────────────────────────────
    def backbone_bottleneck_rf(self) -> int:
        """Theoretical RF along the backbone path at the bottleneck (samples)."""
        k = self.init_rrconv.init_conv.conv.kernel_size[0]
        rf, s = 1, 1
        n_init = 1 + self.init_rrconv.rrconv_iters
        rf += n_init * (k - 1) * s
        for i, stride in enumerate(self.strides[:-1]):
            iters = self.enc_down_convs[i].rrconv_iters
            rf += (k - 1) * s
            s *= stride
            rf += iters * (k - 1) * s
        rf += (1 + self.bottleneck.rrconv_iters) * (k - 1) * s
        return rf

    def output_rf_estimate(self) -> int:
        """Approximate final-output RF including decoder temporal convolutions."""
        k = self.init_rrconv.init_conv.conv.kernel_size[0]
        rf = self.backbone_bottleneck_rf()
        s = 1
        for stride in self.strides[:-1]:
            s *= stride
        for i, upsize in enumerate(self.upsizes):
            new_s = max(s // upsize, 1)
            iters = self.dec_fuse_convs[i].rrconv_iters
            rf += (k - 1) * new_s
            rf += (1 + iters) * (k - 1) * new_s
            s = new_s
        return rf

    def receptive_field_estimate(self) -> int:
        return self.backbone_bottleneck_rf()


def build_mtan_r2unet_rp90_motion(
    input_size: Tuple[int, int] = (9000, 3),
    nb_filters: Optional[List[int]] = None,
    strides: Optional[List[int]] = None,
    kernel_size: int = 7,
    dropout_rate: float = 0.1,
    rrconv_iters: int = 2,
    use_polarity: bool = True,
    polarity_output_channels: int = 3,
    pol_stream_width_mult: int = 3,
    pol_init_rrconv_iters: Optional[int] = 4,
    pol_head_ps_att_width: Optional[int] = 16,
    pad_mode: str = "reflect",
    pretrained_weights: Optional[str] = None,
) -> MTAN_R2UNet_RP90_Motion:
    """Build the 90 s MTAN R2U-Net motion model (picker + polarity + detector)."""
    if nb_filters is None:
        nb_filters = [6, 12, 18, 24, 32]
    if strides is None:
        strides = [5, 5, 3, 3, 2]
    model = MTAN_R2UNet_RP90_Motion(
        input_size=input_size,
        nb_filters=nb_filters,
        strides=strides,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate,
        rrconv_iters=rrconv_iters,
        use_polarity=use_polarity,
        polarity_output_channels=polarity_output_channels,
        pol_stream_width_mult=pol_stream_width_mult,
        pol_init_rrconv_iters=pol_init_rrconv_iters,
        pol_head_ps_att_width=pol_head_ps_att_width,
        pad_mode=pad_mode,
    )
    if pretrained_weights is not None:
        try:
            raw = torch.load(
                pretrained_weights, map_location='cpu', weights_only=True)
        except (pickle.UnpicklingError, RuntimeError) as exc:
            logger.warning(
                "weights_only=True load failed for %s (%s); retrying with "
                "weights_only=False (only load checkpoints you trust)",
                pretrained_weights, exc)
            raw = torch.load(
                pretrained_weights, map_location='cpu', weights_only=False)
        if isinstance(raw, dict) and 'model_state_dict' in raw:
            sd = raw['model_state_dict']
        elif isinstance(raw, dict) and 'state_dict' in raw:
            sd = raw['state_dict']
        else:
            sd = raw
        md = model.state_dict()
        matched = {k: v for k, v in sd.items() if k in md and v.shape == md[k].shape}
        missing = set(md) - set(matched)
        md.update(matched)
        model.load_state_dict(md)
        logger.info("Loaded %d/%d layers from pretrained", len(matched), len(md))
        if missing:
            logger.info("  Random init (%d): %s",
                        len(missing), ", ".join(sorted(missing)[:10]))
    return model
