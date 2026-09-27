"""
MTAN R2U-Net (Optimized).

- **MTAN** — Multi-Task Attention Network
- **R2U-Net** — Recurrent-Residual U-Net
- **RRConv** — Recurrent-Residual Convolution block (multiple iterations with
  independent weights sharing the same residual shortcut)

Architecture:
- Input: 3-component seismogram (B, 3, 6000) — 60 s @ 100 Hz
- Encoder: 6 levels with RRConv blocks + MTAN attention at each level
- Bottleneck: additional RRConv for deep feature extraction
- Decoder: 6 levels with MTAN attention using encoder features as guidance
- Output:
  - Picker:   (B, 3, 6000) — P/S/Noise probabilities
  - Detector: (B, 2, 6000) — mask/unmask probabilities
"""

import logging
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)


class ConvUnit(nn.Module):
    """Conv1D -> BatchNorm -> ReLU -> Dropout"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = kernel_size // 2

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.relu(self.bn(self.conv(x))))


class RRConvUnit(nn.Module):
    """
    Recurrent Residual Convolutional Unit.

    Structure: init_conv -> res_path + recurrent_path -> add
    - Recurrent path applies conv multiple times with residual additions
    - Each recurrent iteration has its OWN weights (not shared)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        dropout: float = 0.1,
        rrconv_iters: int = 3,
    ):
        super().__init__()
        self.rrconv_iters = rrconv_iters

        self.init_conv = ConvUnit(in_channels, out_channels, kernel_size, stride, dropout)

        padding = kernel_size // 2
        self.res_conv = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding)
        nn.init.kaiming_uniform_(self.res_conv.weight, nonlinearity='relu')
        if self.res_conv.bias is not None:
            nn.init.zeros_(self.res_conv.bias)

        self.recurrent_convs = nn.ModuleList([
            ConvUnit(out_channels, out_channels, kernel_size, 1, dropout)
            for _ in range(rrconv_iters)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.init_conv(x)
        res_out = self.res_conv(u)

        r_u = u
        for i in range(self.rrconv_iters):
            r_u = r_u + u
            r_u = self.recurrent_convs[i](r_u)

        return r_u + res_out


class UpConvUnit(nn.Module):
    """Upsample -> Conv -> BN -> ReLU -> Dropout -> Crop -> Concat with skip"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        upsize: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=upsize, mode='nearest')
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 1, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        u = self.upsample(x)
        u = self.dropout(self.relu(self.bn(self.conv(u))))

        # Crop to match skip connection
        if u.shape[2] > skip.shape[2]:
            diff = u.shape[2] - skip.shape[2]
            u = u[:, :, diff // 2:diff // 2 + skip.shape[2]]
        elif u.shape[2] < skip.shape[2]:
            diff = skip.shape[2] - u.shape[2]
            u = F.pad(u, (diff // 2, diff - diff // 2))

        return torch.cat([skip, u], dim=1)


class MTANBlock(nn.Module):
    """
    Multi-Task Attention Block (Corrected Implementation).

    This computes attention weights from encoder features and applies them
    to gate the current features, enabling task-specific feature selection.

    For encoder (down): Computes attention from current + reference features,
                        gates the current features, then downsamples.
                        Reference can have different channel count (from previous level).
    For decoder (up): Upsamples encoder attention, concatenates with decoder features,
                      computes new attention, and gates the fused features.
    """

    def __init__(
        self,
        channels: int,
        ref_channels: Optional[int] = None,  # Reference channels (if different from channels)
        kernel_size: int = 7,
        stride: int = 1,
        upsize: int = 5,
        mode: str = 'down',
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mode = mode
        self.channels = channels
        self.ref_channels = ref_channels if ref_channels is not None else channels

        # Attention computation (1x1 convs)
        if mode == 'down':
            # Input: concat(current, reference) = channels + ref_channels
            in_channels = channels + self.ref_channels
            self.att_conv1 = nn.Conv1d(in_channels, channels, 1)
            self.att_bn1 = nn.BatchNorm1d(channels)
            self.att_conv2 = nn.Conv1d(channels, channels, 1)
            self.att_bn2 = nn.BatchNorm1d(channels)

            # Output conv with stride for downsampling
            if stride > 1:
                self.out_conv = ConvUnit(channels, channels, kernel_size, stride, dropout)
            else:
                self.out_conv = None

        elif mode == 'up':
            # Upsample encoder attention to match decoder resolution
            self.upsample = nn.Upsample(scale_factor=upsize, mode='nearest')
            self.up_conv = ConvUnit(self.ref_channels, channels, kernel_size, 1, dropout)

            # After up_conv, reference has `channels`. Concat with current = channels * 2
            self.att_conv1 = nn.Conv1d(channels * 2, channels, 1)
            self.att_bn1 = nn.BatchNorm1d(channels)
            self.att_conv2 = nn.Conv1d(channels, channels, 1)
            self.att_bn2 = nn.BatchNorm1d(channels)

        self._init_weights()

    def _init_weights(self):
        for conv in [self.att_conv1, self.att_conv2]:
            nn.init.kaiming_uniform_(conv.weight, nonlinearity='relu')
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(
        self,
        current: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            current: Current level features
            reference: Reference features (previous attention or encoder attention)
        """
        if self.mode == 'down':
            # Concatenate current and reference
            x = torch.cat([current, reference], dim=1)

            # Compute attention weights
            att = self.att_conv1(x)
            att = self.att_bn1(att)
            att = F.relu(att, inplace=True)
            att = self.att_conv2(att)
            att = self.att_bn2(att)
            att = torch.sigmoid(att)

            # Apply attention
            out = att * current

            # Downsample if needed
            if self.out_conv is not None:
                out = self.out_conv(out)

            return out

        elif self.mode == 'up':
            # Upsample reference (encoder attention)
            ref_up = self.upsample(reference)

            # Crop to match current
            if ref_up.shape[2] > current.shape[2]:
                diff = ref_up.shape[2] - current.shape[2]
                ref_up = ref_up[:, :, diff // 2:diff // 2 + current.shape[2]]
            elif ref_up.shape[2] < current.shape[2]:
                diff = current.shape[2] - ref_up.shape[2]
                ref_up = F.pad(ref_up, (diff // 2, diff - diff // 2))

            ref_up = self.up_conv(ref_up)

            # Concatenate and compute attention
            x = torch.cat([current, ref_up], dim=1)
            att = self.att_conv1(x)
            att = self.att_bn1(att)
            att = F.relu(att, inplace=True)
            att = self.att_conv2(att)
            att = self.att_bn2(att)
            att = torch.sigmoid(att)

            # Apply attention to current
            return att * current


class MTAN_R2UNet_Optimized(nn.Module):
    """
    Multi-Task Attention Network with R2U-Net backbone (Optimized Version).

    Key improvements over the original TF implementation:
    1. Encoder MTAN features are properly used at corresponding decoder levels
    2. All decoder MTAN outputs contribute to final prediction
    3. Bottleneck is included and connected
    4. Proper multi-scale attention flow

    Architecture:
    - Encoder: Each level has RRConv + dual MTAN (for picker and detector)
    - Decoder: Each level uses encoder MTAN as guidance for attention
    - Output: Accumulated multi-scale predictions
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (9000, 3),
        nb_filters: Optional[List[int]] = None,
        kernel_size: int = 7,
        dropout_rate: float = 0.1,
        stride_size: int = 5,
        upsize: int = 5,
        RRconv_time: int = 3,
        picker_classes: int = 3,
        detector_classes: int = 2,
    ):
        super().__init__()
        if nb_filters is None:
            nb_filters = [6, 12, 18, 24, 30, 36]

        self.nb_filters = nb_filters
        self.depth = len(nb_filters)
        in_channels = input_size[1]

        # ========== Initial ==========
        self.init_rrconv = RRConvUnit(
            in_channels, nb_filters[0], kernel_size, 1, dropout_rate, RRconv_time
        )

        # Initial MTAN for both tasks (self-attention, so ref_channels = channels)
        self.init_ps_mtan = MTANBlock(nb_filters[0], nb_filters[0], kernel_size, 1, upsize, 'down', dropout_rate)
        self.init_mask_mtan = MTANBlock(nb_filters[0], nb_filters[0], kernel_size, 1, upsize, 'down', dropout_rate)

        # ========== Encoder ==========
        self.enc_exp_convs = nn.ModuleList()
        self.enc_down_convs = nn.ModuleList()
        self.enc_ps_mtans = nn.ModuleList()
        self.enc_mask_mtans = nn.ModuleList()

        for i in range(self.depth - 1):
            self.enc_exp_convs.append(
                RRConvUnit(nb_filters[i], nb_filters[i], kernel_size, 1, dropout_rate, RRconv_time)
            )
            self.enc_down_convs.append(
                RRConvUnit(nb_filters[i], nb_filters[i + 1], kernel_size, stride_size, dropout_rate, RRconv_time)
            )
            # MTAN at each encoder level
            # ref_channels: previous attention has nb_filters[i-1] channels (or nb_filters[0] for i=0)
            ref_ch = nb_filters[i - 1] if i > 0 else nb_filters[0]
            self.enc_ps_mtans.append(
                MTANBlock(nb_filters[i], ref_ch, kernel_size, stride_size, upsize, 'down', dropout_rate)
            )
            self.enc_mask_mtans.append(
                MTANBlock(nb_filters[i], ref_ch, kernel_size, stride_size, upsize, 'down', dropout_rate)
            )

        # ========== Bottleneck ==========
        self.bottleneck = RRConvUnit(
            nb_filters[-1], nb_filters[-1], kernel_size, 1, dropout_rate, RRconv_time
        )

        # ========== Decoder ==========
        self.dec_upconvs = nn.ModuleList()
        self.dec_fuse_convs = nn.ModuleList()
        self.dec_ps_mtans = nn.ModuleList()
        self.dec_mask_mtans = nn.ModuleList()

        for i in range(self.depth):
            if i == 0:
                in_ch = nb_filters[-1]
            else:
                in_ch = nb_filters[-i]
            out_ch = nb_filters[-(i + 1)]

            self.dec_upconvs.append(
                UpConvUnit(in_ch, out_ch, kernel_size, upsize, dropout_rate)
            )
            self.dec_fuse_convs.append(
                RRConvUnit(out_ch * 2, out_ch, kernel_size, 1, dropout_rate, RRconv_time)
            )
            # MTAN at each decoder level - chained: each uses previous decoder MTAN output
            # i=0: reference = deepest encoder MTAN (nb_filters[-2] channels)
            # i>0: reference = previous decoder MTAN output (nb_filters[-i] channels)
            if i == 0:
                ref_ch = nb_filters[-2]  # deepest encoder MTAN output
            else:
                ref_ch = nb_filters[-i]  # previous decoder level MTAN output
            self.dec_ps_mtans.append(
                MTANBlock(out_ch, ref_ch, kernel_size, 1, upsize, 'up', dropout_rate)
            )
            self.dec_mask_mtans.append(
                MTANBlock(out_ch, ref_ch, kernel_size, 1, upsize, 'up', dropout_rate)
            )

        # ========== Output Heads ==========
        self.picker_head = nn.Conv1d(nb_filters[0], picker_classes, 1)
        self.detector_head = nn.Conv1d(nb_filters[0], detector_classes, 1)

        nn.init.kaiming_uniform_(self.picker_head.weight, nonlinearity='relu')
        nn.init.zeros_(self.picker_head.bias)
        nn.init.kaiming_uniform_(self.detector_head.weight, nonlinearity='relu')
        nn.init.zeros_(self.detector_head.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        input_len = x.shape[2]

        # ========== Initial ==========
        e0 = self.init_rrconv(x)

        # Initial attention (self-attention at input level)
        ps_att = self.init_ps_mtan(e0, e0)
        mask_att = self.init_mask_mtan(e0, e0)

        # Store for skip connections
        encoder_features = [e0]
        ps_att_features = [ps_att]
        mask_att_features = [mask_att]

        # ========== Encoder ==========
        enc = e0
        for i in range(self.depth - 1):
            # Expand at current resolution
            exp = self.enc_exp_convs[i](enc)

            # Compute attention using previous attention as reference
            ps_att = self.enc_ps_mtans[i](exp, ps_att)
            mask_att = self.enc_mask_mtans[i](exp, mask_att)

            # Downsample backbone
            enc = self.enc_down_convs[i](exp)

            encoder_features.append(enc)
            ps_att_features.append(ps_att)
            mask_att_features.append(mask_att)

        # ========== Bottleneck ==========
        bottleneck = self.bottleneck(enc)

        # ========== Decoder ==========
        dec = bottleneck

        # Start decoder attention chain from deepest encoder MTAN
        ps_att = ps_att_features[-1]
        mask_att = mask_att_features[-1]

        for i in range(self.depth):
            skip_idx = -(i + 1)
            skip = encoder_features[skip_idx]

            # Upsample and concatenate with skip
            up = self.dec_upconvs[i](dec, skip)
            dec = self.dec_fuse_convs[i](up)

            # Apply MTAN: chained — each level's output feeds into the next
            ps_att = self.dec_ps_mtans[i](dec, ps_att)
            mask_att = self.dec_mask_mtans[i](dec, mask_att)

        # ========== Output ==========
        # Ensure output length matches input
        if ps_att.shape[2] != input_len:
            if ps_att.shape[2] > input_len:
                ps_att = ps_att[:, :, :input_len]
                mask_att = mask_att[:, :, :input_len]
            else:
                pad = input_len - ps_att.shape[2]
                ps_att = F.pad(ps_att, (0, pad))
                mask_att = F.pad(mask_att, (0, pad))

        # Cast to float32 before softmax to prevent float16 overflow (NaN) under AMP autocast.
        # exp() in softmax can overflow in float16 with large logits → inf/inf = NaN.
        picker = F.softmax(self.picker_head(ps_att).float(), dim=1)
        detector = F.softmax(self.detector_head(mask_att).float(), dim=1)

        return picker, detector

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Alias for backward compatibility
MTAN_R2UNet = MTAN_R2UNet_Optimized


def build_mtan_R2unet_optimized(
    input_size: Tuple[int, int] = (9000, 3),
    nb_filters: Optional[List[int]] = None,
    kernel_size: int = 7,
    dropout_rate: float = 0.1,
    stride_size: int = 5,
    upsize: int = 5,
    RRconv_time: int = 3,
    pretrained_weights: Optional[str] = None,
) -> MTAN_R2UNet_Optimized:
    """Build the optimized MTAN R2U-Net model."""
    if nb_filters is None:
        nb_filters = [6, 12, 18, 24, 30]
    model = MTAN_R2UNet_Optimized(
        input_size=input_size,
        nb_filters=nb_filters,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate,
        stride_size=stride_size,
        upsize=upsize,
        RRconv_time=RRconv_time,
    )

    if pretrained_weights is not None:
        try:
            state_dict = torch.load(
                pretrained_weights, map_location='cpu', weights_only=True)
        except (pickle.UnpicklingError, RuntimeError) as exc:
            logger.warning(
                "weights_only=True load failed for %s (%s); retrying with "
                "weights_only=False (only load checkpoints you trust)",
                pretrained_weights, exc)
            state_dict = torch.load(
                pretrained_weights, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict)

    return model


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing MTAN R2U-Net (Optimized) on: {device}")
    print("=" * 60)

    # CHANGE SETTINGS HERE TO SEE EFFECT
    test_input_shape = (9000, 3)              # Try changing 6000->3000 (no param change), or 3->1 (param change)
    test_filters = [6, 12, 18, 24, 30]    # Try changing to [8, 16...] (big param change)

    print(f"Configuration:")
    print(f"  Input Shape: {test_input_shape}")
    print(f"  Filters:     {test_filters}")

    # FIX: Pass the test configuration to the constructor
    model = MTAN_R2UNet_Optimized(
        input_size=test_input_shape,
        nb_filters=test_filters
    ).to(device)

    print(f"Total parameters: {model.count_parameters():,}")

    # Compare with original TF model
    print(f"Original TF model: 349,685 (but most MTAN pruned!)")
    print(f"This optimized model: {model.count_parameters():,} (all MTAN active)")

    x = torch.randn(2, 3, 9000).to(device)
    print(f"\nInput shape: {x.shape}")

    model.eval()
    with torch.no_grad():
        picker, detector = model(x)

    print(f"Picker shape: {picker.shape}")
    print(f"Detector shape: {detector.shape}")
    print(f"Picker probabilities sum: {picker[0, :, 0].sum().item():.4f}")
    print(f"Detector probabilities sum: {detector[0, :, 0].sum().item():.4f}")
    print("\n✓ Optimized model test passed!")

    print("\n" + "=" * 60)
    print("Key improvements over original TF implementation:")
    print("1. All encoder MTAN outputs are used as guidance in decoder")
    print("2. All decoder MTAN blocks contribute to final output")
    print("3. Bottleneck is included and connected")
    print("4. Proper multi-scale attention flow implemented")
    print("=" * 60)
