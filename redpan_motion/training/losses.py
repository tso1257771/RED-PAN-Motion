"""
Loss functions for RED-PAN PyTorch training.

Implements Dynamic Weight Averaging (DWA) and other multi-task loss strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import numpy as np


class CategoricalCrossEntropy(nn.Module):
    """
    Categorical cross-entropy loss matching TensorFlow behavior.
    
    Works with probability distributions (after softmax).
    """
    
    def __init__(self, reduction: str = 'mean', eps: float = 1e-7):
        super().__init__()
        self.reduction = reduction
        self.eps = eps
    
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        sample_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            y_pred: Predicted probabilities (B, C, T) (Model Output)
            y_true: Ground truth labels (B, T, C) or (B, C, T)
            sample_weight: Optional per-sample weights (B, T)
        
        Returns:
            Loss value
        """
        # Align y_true to y_pred (B, C, T)
        # If y_true is (B, T, C) where T >> C, transpose it
        if y_true.shape[1] > y_true.shape[2]:
            y_true = y_true.transpose(1, 2)
            
        # Ensure y_pred matches (B, C, T) layout
        if y_pred.shape != y_true.shape:
            # Fallback if y_pred was somehow (B, T, C)
             if y_pred.shape[1] > y_pred.shape[2]:
                  y_pred = y_pred.transpose(1, 2)

        # Sanitize and clamp predictions/targets for numerical stability
        y_true = torch.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
        y_pred = torch.nan_to_num(y_pred, nan=self.eps, posinf=1.0 - self.eps, neginf=self.eps)
        y_pred = torch.clamp(y_pred, self.eps, 1.0 - self.eps)
        
        # Cross-entropy over channels (dim 1): -sum(y_true * log(y_pred))
        loss = -torch.sum(y_true * torch.log(y_pred), dim=1)  # (B, T)
        
        if sample_weight is not None:
            loss = loss * sample_weight
        
        if self.reduction == 'mean':
            # Match TF: Sum over time, Mean over batch
            # This gives larger gradients on peak timesteps (important for learning)
            return loss.sum(dim=-1).mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class DynamicWeightAveraging:
    """
    Dynamic Weight Averaging (DWA) for multi-task learning.
    
    Automatically balances task losses based on their rate of change.
    
    Reference:
        "End-to-End Multi-Task Learning with Attention" (Liu et al., 2019)
    
    The weight for task k at epoch t is:
        w_k(t) = K * exp(r_k(t-1) / T) / sum(exp(r_j(t-1) / T))
    
    where r_k(t-1) = L_k(t-1) / L_k(t-2) is the loss ratio.
    """
    
    def __init__(
        self,
        n_tasks: int = 2,
        temperature: float = 2.0,
        initial_weights: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            n_tasks: Number of tasks (default 2: picker + detector)
            temperature: Temperature parameter T (higher = more uniform)
            initial_weights: Optional initial task weights
        """
        self.n_tasks = n_tasks
        self.temperature = temperature
        
        # History of average losses per epoch
        self.loss_history = []
        
        # Current weights
        if initial_weights is not None:
            self.weights = initial_weights
        else:
            self.weights = torch.ones(n_tasks)
    
    def update(self, epoch_losses: torch.Tensor) -> torch.Tensor:
        """
        Update weights based on loss history.
        
        Args:
            epoch_losses: Average losses for each task from current epoch (n_tasks,)
        
        Returns:
            Updated task weights (n_tasks,)
        """
        self.loss_history.append(epoch_losses.detach().cpu())
        
        if len(self.loss_history) < 2:
            # Not enough history, return uniform weights
            return self.weights
        
        # Compute loss ratios
        prev_losses = self.loss_history[-2]
        curr_losses = self.loss_history[-1]
        
        # Avoid division by zero
        ratios = curr_losses / (prev_losses + 1e-8)
        
        # Compute softmax weights
        exp_ratios = torch.exp(ratios / self.temperature)
        self.weights = self.n_tasks * exp_ratios / exp_ratios.sum()
        
        return self.weights
    
    def get_weights(self) -> torch.Tensor:
        """Get current task weights."""
        return self.weights
    
    def reset(self):
        """Reset loss history and weights."""
        self.loss_history = []
        self.weights = torch.ones(self.n_tasks)


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for RED-PAN (picker + detector).
    
    Combines picker loss and detector loss with configurable weighting.
    """
    
    def __init__(
        self,
        picker_weight: float = 1.0,
        detector_weight: float = 1.0,
        use_dwa: bool = True,
        dwa_temperature: float = 2.0,
        weight_mode: str = 'balanced',
        weight_alpha: float = 1.0,
        weight_max: float = 50.0,
        label_order: str = 'PSN',
    ):
        """
        Args:
            picker_weight: Initial weight for picker loss
            detector_weight: Initial weight for detector loss
            use_dwa: Whether to use Dynamic Weight Averaging
            dwa_temperature: Temperature for DWA
            weight_mode: 'balanced' or 'soft'
            weight_alpha: Alpha param for weighting
            weight_max: Max weight cap
        """
        super().__init__()
        
        # Picker uses Balanced Cross Entropy (weighted)
        self.picker_loss_fn = BalancedCrossEntropy(
            mode=weight_mode,
            alpha=weight_alpha,
            max_weight=weight_max,
            label_order=label_order,
        )
        
        # Detector uses standard Cross Entropy (unweighted per timestep)
        self.detector_loss_fn = CategoricalCrossEntropy(reduction='mean')
        
        self.use_dwa = use_dwa
        
        if use_dwa:
            self.dwa = DynamicWeightAveraging(
                n_tasks=2,
                temperature=dwa_temperature,
                initial_weights=torch.tensor([picker_weight, detector_weight]),
            )
        else:
            self.register_buffer('weights', torch.tensor([picker_weight, detector_weight]))
    
    def forward(
        self,
        picker_pred: torch.Tensor,
        detector_pred: torch.Tensor,
        picker_target: torch.Tensor,
        detector_target: torch.Tensor,
        return_individual: bool = False,
    ) -> torch.Tensor:
        """
        Compute combined multi-task loss.
        """
        picker_loss = self.picker_loss_fn(picker_pred, picker_target)
        detector_loss = self.detector_loss_fn(detector_pred, detector_target)
        
        weights = self.dwa.get_weights() if self.use_dwa else self.weights
        weights = weights.to(picker_loss.device)
        
        combined_loss = weights[0] * picker_loss + weights[1] * detector_loss
        
        if return_individual:
            return combined_loss, picker_loss, detector_loss
        return combined_loss
    
    def update_weights(self, epoch_losses: torch.Tensor):
        """Update DWA weights after each epoch."""
        if self.use_dwa:
            self.dwa.update(epoch_losses)
    
    def get_weights(self) -> torch.Tensor:
        """Get current task weights."""
        if self.use_dwa:
            return self.dwa.get_weights()
        return self.weights


class BalancedCrossEntropy(nn.Module):
    """
    Cross-entropy loss with per-timestep balancing.
    
    Up-weights timesteps with peaks (P/S arrivals) vs flat background regions.
    Helps with class imbalance since most timesteps are noise.
    """
    
    def __init__(
        self,
        mode: str = 'balanced',
        alpha: float = 1.0,  # P12 uses 1.0 for balanced
        max_weight: float = 50.0,
        eps: float = 1e-6,
        min_flat_weight: float = 0.1,
        label_order: str = 'PSN',
    ):
        """
        Args:
            mode: 'balanced' (inverse frequency) or 'soft' (alpha mixing)
            alpha: 
                - For 'soft': mixing factor (0-1)
                - For 'balanced': unused (or base scalar)
            max_weight: Maximum weight cap
            eps: Numerical stability
            min_flat_weight: Minimum weight for flat regions (balanced mode)
        """
        super().__init__()
        self.mode = mode
        self.alpha = alpha
        self.max_weight = max_weight
        self.eps = eps
        self.min_flat_weight = min_flat_weight
        self.label_order = label_order.upper()
    
    def compute_weights(self, labels: torch.Tensor) -> torch.Tensor:
        """Dispatcher for weight computation."""
        # labels: (B, C, T) expected after align
        # Find peak intensity (max of P and S channels - index 0, 1)
        # Note: If channels=3, P=0, S=1, Noise=2? Or P=0(P), S=1(S), Noise=2(N)?
        # Usually P/S are first channels.
        
        # Labels are PSN: P=ch0, S=ch1, Noise=ch2. We want max(P, S).
        if labels.shape[1] >= 3:
            peak_intensity = labels[:, 0:2, :].max(dim=1).values
        elif labels.shape[1] == 2:
            # If 2 channels, assume P, S? Or Noise, Signal? 
            # Safer to just take max of whatever is passed if structure assumes signal-first.
            # But here we know it's Noise-first. 
            # If only 2 channels provided and one is noise.. ambiguous. 
            # Assuming standard 3-channel input for this project.
            peak_intensity = labels.max(dim=1).values
        else:
            peak_intensity = labels.max(dim=1).values
            
        if self.mode == 'balanced':
            return self._compute_balanced_weights(peak_intensity)
        else:
            return self._compute_soft_weights(peak_intensity)

    def _compute_balanced_weights(self, peak_intensity: torch.Tensor) -> torch.Tensor:
        """
        Inverse-frequency weighting (matches P12 compute_balanced_weights).
        """
        # peak_intensity: (B, T)
        seq_len = peak_intensity.shape[1]
        
        # Calculate ratio of peaks in the sequence
        # (B, 1)
        peak_sum = peak_intensity.sum(dim=-1, keepdim=True)
        peak_ratio = peak_sum / seq_len
        peak_ratio = torch.clamp(peak_ratio, 0.0, 1.0)
        
        # Peak weight = 1.0 / frequency
        peak_weight = torch.clamp(1.0 / (peak_ratio + self.eps), max=self.max_weight)
        
        # Flat weight = 1.0 / (1 - frequency)
        flat_weight = torch.clamp(
            1.0 / (1.0 - peak_ratio + self.eps),
            min=self.min_flat_weight,
            max=self.max_weight,  # cap to prevent fp16 overflow on dense-peak samples
        )
        
        # Combine: weight depends on whether it's a peak or flat
        weights = peak_intensity * peak_weight + (1.0 - peak_intensity) * flat_weight
        
        # Normalize to mean=1 over time
        weights = weights / (weights.mean(dim=-1, keepdim=True) + self.eps)
        weights = torch.nan_to_num(weights, nan=1.0, posinf=self.max_weight, neginf=0.0)
        
        return weights

    def _compute_soft_weights(self, peak_intensity: torch.Tensor) -> torch.Tensor:
        """
        Soft alpha-mixing weighting.
        """
        weights = self.alpha + (1.0 - self.alpha) * peak_intensity
        weights = torch.clamp(weights, min=self.eps, max=self.max_weight)
        weights = weights / (weights.mean(dim=-1, keepdim=True) + self.eps)
        return weights
    
    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            y_pred: Predicted probabilities (B, C, T)
            y_true: Ground truth labels (B, T, C) or (B, C, T)
        
        Returns:
            Weighted loss (averaged over time and batch)
        """
        # Align y_true to y_pred (B, C, T)
        if y_true.shape[1] > y_true.shape[2]:
            y_true = y_true.transpose(1, 2)
        y_true = torch.nan_to_num(y_true, nan=0.0, posinf=0.0, neginf=0.0)
            
        # Compute weights based on y_true (B, C, T) -> (B, T)
        weights = self.compute_weights(y_true)  # (B, T)
        
        # Clamp predictions
        y_pred = torch.nan_to_num(y_pred, nan=self.eps, posinf=1.0 - self.eps, neginf=self.eps)
        y_pred = torch.clamp(y_pred, self.eps, 1.0 - self.eps)
        
        # Cross-entropy over channels (dim 1)
        ce_loss = -torch.sum(y_true * torch.log(y_pred), dim=1)  # (B, T)
        
        # Apply weights and reduce: Sum over time, Mean over batch
        # This matches TF behavior and gives larger gradients on peak timesteps
        weighted_loss = (ce_loss * weights).sum(dim=-1).mean()
        
        return weighted_loss


if __name__ == '__main__':
    # Test losses
    torch.manual_seed(42)
    
    batch_size = 4
    seq_len = 6000
    
    # Create dummy data
    picker_pred = torch.softmax(torch.randn(batch_size, 3, seq_len), dim=1)
    detector_pred = torch.softmax(torch.randn(batch_size, 2, seq_len), dim=1)
    
    picker_target = torch.zeros(batch_size, seq_len, 3)
    picker_target[..., 2] = 1.0  # All noise
    
    detector_target = torch.zeros(batch_size, seq_len, 2)
    detector_target[..., 1] = 1.0  # All unmask
    
    # Test CategoricalCrossEntropy
    loss_fn = CategoricalCrossEntropy()
    loss = loss_fn(picker_pred, picker_target)
    print(f"CategoricalCE loss: {loss.item():.4f}")
    
    # Test MultiTaskLoss
    mtl = MultiTaskLoss(use_dwa=True)
    combined, picker_loss, detector_loss = mtl(
        picker_pred, detector_pred,
        picker_target, detector_target,
        return_individual=True
    )
    print(f"MultiTask loss: {combined.item():.4f} (picker: {picker_loss.item():.4f}, detector: {detector_loss.item():.4f})")
    print(f"DWA weights: {mtl.get_weights()}")
    
    # Simulate epoch update
    mtl.update_weights(torch.tensor([picker_loss.item(), detector_loss.item()]))
    mtl.update_weights(torch.tensor([picker_loss.item() * 0.9, detector_loss.item() * 1.1]))
    print(f"DWA weights after update: {mtl.get_weights()}")
    
    # Test BalancedCrossEntropy
    balanced_ce = BalancedCrossEntropy()
    balanced_loss = balanced_ce(picker_pred, picker_target)
    print(f"BalancedCE loss: {balanced_loss.item():.4f}")
    
    print("\n✓ All losses working correctly!")


# ─────────────────────────────────────────────────────────────────────
# Polarity loss + 3-task loss (for 90s model with polarity head)
# ─────────────────────────────────────────────────────────────────────
class PolarityBCELoss(nn.Module):
    """BCE loss for signed polarity in [-1, +1], weighted by |target|.

    Maps predictions and targets from [-1,+1] → [0,1], then applies BCE.
    Timesteps with target=0 (no polarity label) contribute zero loss
    because weight = |target| = 0.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true, y_weight=None):
        """BCE loss for signed polarity, weighted by y_weight.

        If y_weight is None, falls back to |y_true| (backward-compat behavior
        when the dataset didn't provide explicit weights). When the dataset
        supplies weights, y_weight should be nonzero at labeled P locations
        for U/D (matching |target|) AND for N (target=0 but supervised-zero),
        and zero elsewhere.
        """
        p = torch.clamp((y_pred + 1.0) / 2.0, self.eps, 1.0 - self.eps)
        t = torch.clamp((y_true + 1.0) / 2.0, self.eps, 1.0 - self.eps)
        bce = -(t * torch.log(p) + (1.0 - t) * torch.log(1.0 - p))
        weight = torch.abs(y_true) if y_weight is None else y_weight
        weighted = bce * weight
        weight_sum = weight.sum(dim=-1).clamp(min=1.0)
        per_sample = weighted.sum(dim=-1) / weight_sum
        T = y_pred.shape[-1]
        # Average over labeled samples only — avoids gradient dilution when
        # the batch contains many unlabeled samples (weight=0 everywhere).
        labeled = (weight.sum(dim=-1) > 0).to(per_sample.dtype)
        n_labeled = labeled.sum().clamp(min=1.0)
        return (per_sample * labeled * T).sum() / n_labeled


class PolarityMSELoss(nn.Module):
    """MSE loss for signed polarity in [-1, +1], weighted by |target|.

    Regression loss on the tanh output. Unlike PolarityBCELoss, the minimum
    is exactly 0 when y_pred == y_true (no H(t) floor from BCE on graded
    Gaussian targets). Timesteps with target=0 contribute zero (weight=|t|=0).
    """
    def forward(self, y_pred, y_true, y_weight=None):
        weight = torch.abs(y_true) if y_weight is None else y_weight
        sq_err = (y_pred - y_true).pow(2) * weight
        weight_sum = weight.sum(dim=-1).clamp(min=1.0)
        per_sample = sq_err.sum(dim=-1) / weight_sum
        T = y_pred.shape[-1]
        # Average over labeled samples only (matches PolarityBCELoss /
        # PolaritySoftmaxCELoss) to avoid dilution by unlabeled samples.
        labeled = (weight.sum(dim=-1) > 0).to(per_sample.dtype)
        n_labeled = labeled.sum().clamp(min=1.0)
        return (per_sample * labeled * T).sum() / n_labeled


class PolaritySoftmaxCELoss(nn.Module):
    """Masked categorical cross-entropy on 3-channel polarity (PhaseNet+ style).

    Predictions are softmax probabilities over [N, U, D] channels at each
    timestep; targets are a 3-channel distribution summing to 1 with U/D
    Gaussian peaks at the labeled pick. Loss is restricted to a per-sample
    mask (binary window around the pick) — unlabeled samples have an all-zero
    mask and contribute zero loss.

    Reduction: weighted-mean CE over masked timesteps per sample, then ×T,
    averaged over labeled samples only — keeps loss magnitude comparable
    regardless of the unlabeled fraction in the batch, and matches the
    picker/detector `sum_over_batch` scale on the labeled subset.
    """
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true, mask):
        """
        Args:
            y_pred: (B, 3, T) softmax probabilities over [N, U, D].
            y_true: (B, 3, T) target distribution (sums to 1).
            mask:   (B, 1, T) binary mask (1 inside loss window, 0 elsewhere).
                    All-zero along T for samples without a polarity label.
        """
        p = torch.clamp(y_pred, self.eps, 1.0 - self.eps)
        # Per-timestep CE summed over channels: (B, 1, T)
        ce = -(y_true * torch.log(p)).sum(dim=1, keepdim=True)
        weighted = ce * mask
        # Per-sample labeled flag (B, 1): 1 if mask has any 1s, else 0.
        labeled = (mask.sum(dim=-1) > 0).to(weighted.dtype)
        weight_sum = mask.sum(dim=-1).clamp(min=1.0)            # (B, 1)
        per_sample = weighted.sum(dim=-1) / weight_sum          # (B, 1)
        T = y_pred.shape[-1]
        # Average over labeled samples only; if batch has zero labeled samples,
        # return a differentiable zero (keeps grad graph intact).
        n_labeled = labeled.sum().clamp(min=1.0)
        return (per_sample * labeled * T).sum() / n_labeled


class PolaritySoftmaxUDLoss(nn.Module):
    """Masked categorical CE on 2-channel [U, D] polarity (Case B).

    Direct cross-entropy on the binary direction class — softmax across only
    [U, D] channels, supervised only at picks where the analyst labeled U or D.
    Abstention (N picks) is handled by the separate impulsive head, so the
    polarity head focuses solely on direction discrimination.

    Mask is nonzero only at U/D picks (zero at N picks and unlabeled samples)
    → labeled-sample averaging matches the picker/detector loss scale.
    """
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true, mask):
        """
        Args:
            y_pred: (B, 2, T) softmax probabilities over [U, D].
            y_true: (B, 2, T) target distribution (sum to 1 inside mask, else 0).
            mask:   (B, 1, T) binary mask — 1 inside loss window at U/D picks.
        """
        p = torch.clamp(y_pred, self.eps, 1.0 - self.eps)
        ce = -(y_true * torch.log(p)).sum(dim=1, keepdim=True)
        weighted = ce * mask
        labeled = (mask.sum(dim=-1) > 0).to(weighted.dtype)
        weight_sum = mask.sum(dim=-1).clamp(min=1.0)
        per_sample = weighted.sum(dim=-1) / weight_sum
        T = y_pred.shape[-1]
        n_labeled = labeled.sum().clamp(min=1.0)
        return (per_sample * labeled * T).sum() / n_labeled


class EQTPickerBCELoss(nn.Module):
    """Sigmoid BCE on a 1-channel Gaussian peak target (v8 EQT-style picker).

    Used for both P and S heads (called once per head). Sparsity of the Gaussian
    peak target → backbone learns "predict 0 by default", suppressing baseline
    on noise the way EQTransformer does. No mask/weight required.

    Reduction: sum-over-T, mean-over-B (matches picker/detector scale).
    """
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: (B, 1, T) sigmoid output ∈ [0, 1]
            y_true: (B, 1, T) Gaussian peak target ∈ [0, 1]
        """
        p = torch.clamp(y_pred, self.eps, 1.0 - self.eps)
        bce = -(y_true * torch.log(p) + (1.0 - y_true) * torch.log(1.0 - p))
        return bce.sum(dim=-1).mean()


class EQTDetectorBCELoss(nn.Module):
    """Sigmoid BCE on a 1-channel detector block target (v8 EQT-style detector).

    Target is the Event channel of the existing 2-channel detector target — a
    block (Event=1 between P and S, smoothly tapered). Sigmoid BCE drives output
    to 0 outside the block, addressing the noise-baseline issue of softmax CE.
    """
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: (B, 1, T) sigmoid output ∈ [0, 1]
            y_true: (B, 1, T) block target (1 between P and S, 0 elsewhere)
        """
        p = torch.clamp(y_pred, self.eps, 1.0 - self.eps)
        bce = -(y_true * torch.log(p) + (1.0 - y_true) * torch.log(1.0 - p))
        return bce.sum(dim=-1).mean()


class EventCenterBCELoss(nn.Module):
    """Sigmoid BCE on sparse Gaussian event_center target (v7 hybrid).

    Auxiliary detector head — fires only at event midpoint (P+S)/2. Sparsity of
    target naturally drives output to 0 outside event centers (PhaseNet+ event
    style). No mask required — BCE handles class imbalance through sigmoid
    saturation behavior.

    Reduction: sum-over-T, mean-over-B (matches picker/detector scale).
    """
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: (B, 1, T) sigmoid output ∈ [0, 1]
            y_true: (B, 1, T) sparse Gaussian target ∈ [0, 1]
        """
        p = torch.clamp(y_pred, self.eps, 1.0 - self.eps)
        bce = -(y_true * torch.log(p) + (1.0 - y_true) * torch.log(1.0 - p))
        # Sum over T, mean over B — matches picker/detector
        return bce.sum(dim=-1).mean()


class ImpulsiveBCELoss(nn.Module):
    """BCE for impulsive probability ∈ [0, 1], weighted by explicit y_weight.

    Same labeled-sample averaging as PolarityBCELoss (divides by n_labeled,
    not batch size) so loss magnitude and gradient don't dilute with unlabeled
    samples.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true, y_weight):
        # y_pred ∈ [0, 1] (sigmoid), y_true ∈ [0, 1] already (dataset-produced
        # Gaussian), y_weight ≥ 0. Clamp only y_pred — y_true's exact zeros
        # at emergent samples are the correct minimum-loss target.
        p = torch.clamp(y_pred, self.eps, 1.0 - self.eps)
        bce = -(y_true * torch.log(p) + (1.0 - y_true) * torch.log(1.0 - p))
        weighted = bce * y_weight
        weight_sum = y_weight.sum(dim=-1).clamp(min=1.0)
        per_sample = weighted.sum(dim=-1) / weight_sum
        T = y_pred.shape[-1]
        labeled = (y_weight.sum(dim=-1) > 0).to(per_sample.dtype)
        n_labeled = labeled.sum().clamp(min=1.0)
        return (per_sample * labeled * T).sum() / n_labeled


class MultiTaskLossWithPolarity(nn.Module):
    """3- or 4-task loss: picker + polarity [+ impulsive] + detector with DWA.

    Polarity loss is auto-masked for unlabeled samples (target=0).

    Set use_impulsive=True to enable the 4-task variant (v3):
        total = w[0]·picker + w[1]·polarity + w[2]·impulsive + w[3]·detector

    loss_reduction controls how losses are reduced:
      'mean':           mean over B and T (default, PyTorch standard)
      'sum_over_batch': sum over T, mean over B (TF standard — stronger gradients,
                        detector loss proportional to picker loss)
    """
    def __init__(
        self,
        picker_weight=1.0, polarity_weight=1.0, detector_weight=1.0,
        impulsive_weight=1.0,
        event_center_weight=1.0,
        use_dwa=True, dwa_temperature=2.0,
        weight_mode='balanced', weight_alpha=1.0, weight_max=50.0,
        loss_reduction='sum_over_batch',
        polarity_loss_type='softmax_ce',
        use_impulsive=False,
        use_event_center=False,
        use_eqt_heads=False,
        use_eqt_picker_heads=False,
        use_eqt_detector_head=False,
    ):
        super().__init__()
        self.loss_reduction = loss_reduction
        self.polarity_loss_type = polarity_loss_type
        self.use_impulsive = use_impulsive
        self.use_event_center = use_event_center
        self.use_eqt_heads = use_eqt_heads
        # Granular flags — `use_eqt_heads` is a shorthand for both
        self.use_eqt_picker_heads = use_eqt_picker_heads or use_eqt_heads
        self.use_eqt_detector_head = use_eqt_detector_head or use_eqt_heads

        # BalancedCrossEntropy already does sum(T)/mean(B) internally.
        # In EQT mode (use_eqt_heads), picker is replaced by separate sigmoid
        # P and S heads (BCE) — the picker_loss_fn here is reused as a fallback
        # but the forward path in EQT mode bypasses it via dedicated BCE losses.
        self.picker_loss_fn = BalancedCrossEntropy(
            mode=weight_mode, alpha=weight_alpha, max_weight=weight_max)
        if self.use_eqt_picker_heads:
            self.eqt_p_loss_fn = EQTPickerBCELoss()
            self.eqt_s_loss_fn = EQTPickerBCELoss()
        if self.use_eqt_detector_head:
            self.eqt_det_loss_fn = EQTDetectorBCELoss()
        # softmax_ce: PhaseNet+ style 3-channel [N,U,D] masked CE
        # softmax_ud: Case B 2-channel [U,D] masked CE (abstention via impulsive head)
        # mse: 1-channel signed regression (transitional)
        # bce: legacy 1-channel BCE on tanh (has H(t) floor — kept for compat)
        if polarity_loss_type == 'softmax_ce':
            self.polarity_loss_fn = PolaritySoftmaxCELoss()
        elif polarity_loss_type == 'softmax_ud':
            self.polarity_loss_fn = PolaritySoftmaxUDLoss()
        elif polarity_loss_type == 'bce':
            self.polarity_loss_fn = PolarityBCELoss()
        else:
            self.polarity_loss_fn = PolarityMSELoss()

        if use_impulsive:
            self.impulsive_loss_fn = ImpulsiveBCELoss()
        if use_event_center:
            self.event_center_loss_fn = EventCenterBCELoss()

        # Detector: match reduction to picker so losses are on same scale
        det_reduction = 'mean' if loss_reduction == 'mean' else 'none'
        self.detector_loss_fn = CategoricalCrossEntropy(reduction=det_reduction)
        self.use_dwa = use_dwa
        # DWA task ordering (when all heads enabled):
        #   [picker, polarity, impulsive, detector, event_center]
        weight_list = [picker_weight, polarity_weight]
        if use_impulsive:
            weight_list.append(impulsive_weight)
        weight_list.append(detector_weight)
        if use_event_center:
            weight_list.append(event_center_weight)
        init_w = torch.tensor(weight_list)
        n_tasks = len(weight_list)
        # Keep the static "scale prior" so DWA's adaptive weights are applied
        # ON TOP OF user-specified per-head magnitudes (e.g. polarity_loss_weight=0.2
        # so that polarity contribution stays at 1/5 the budget regardless of DWA).
        # Without this, DWA replaces init_w with ratio-based weights after epoch 2,
        # silently discarding polarity_loss_weight.
        self.register_buffer('static_scales', init_w.clone())
        if use_dwa:
            self.dwa = DynamicWeightAveraging(
                n_tasks=n_tasks, temperature=dwa_temperature,
                initial_weights=torch.ones(n_tasks),  # DWA starts uniform; static_scales applied separately
            )
        else:
            self.register_buffer('weights', init_w)

    def get_weights(self):
        # When DWA is on, multiply the adaptive weight by the static scale prior
        # so that user-specified polarity_loss_weight is honored throughout
        # training (otherwise DWA's ratio-normalized weights dominate after E2).
        if self.use_dwa:
            return self.dwa.get_weights() * self.static_scales
        return self.weights

    def update_weights(self, epoch_losses):
        if self.use_dwa:
            self.dwa.update(epoch_losses)

    def forward(
        self,
        picker_pred, polarity_pred, detector_pred,
        picker_target, polarity_target, detector_target,
        polarity_mask=None, polarity_raw=None, ungated_polarity_weight=0.0,
        impulsive_pred=None, impulsive_target=None, impulsive_weight=None,
        event_center_pred=None, event_center_target=None,
        return_individual=False,
    ):
        if self.use_eqt_picker_heads:
            # v8 picker: (B, 2, T) [P, S] sigmoid → 2× BCE on existing softmax target's
            # P (channel 0) and S (channel 1) Gaussian peaks. N channel ignored.
            p_loss = self.eqt_p_loss_fn(
                picker_pred[:, 0:1, :], picker_target[:, 0:1, :])
            s_loss = self.eqt_s_loss_fn(
                picker_pred[:, 1:2, :], picker_target[:, 1:2, :])
            picker_loss = p_loss + s_loss
        else:
            picker_loss = self.picker_loss_fn(picker_pred, picker_target)
        if polarity_pred is None:
            # Backbone-only mode: skip polarity loss entirely.
            polarity_loss = torch.zeros((), device=picker_loss.device,
                                        dtype=picker_loss.dtype)
        elif self.polarity_loss_type in ('softmax_ce', 'softmax_ud'):
            assert polarity_mask is not None, \
                f"{self.polarity_loss_type} polarity loss requires polarity_mask"
            polarity_loss = self.polarity_loss_fn(
                polarity_pred, polarity_target, polarity_mask)
        else:
            # For BCE/MSE: the dataset-supplied tensor (passed via
            # polarity_mask kwarg for API continuity) is now the explicit
            # |weight| tensor — nonzero at P for U/D/N labels so the head
            # is supervised toward ±1 / 0 respectively, and zero elsewhere.
            polarity_loss = self.polarity_loss_fn(
                polarity_pred, polarity_target, y_weight=polarity_mask)

        # Impulsive head (v3 optional 4th task)
        if self.use_impulsive:
            if impulsive_pred is None or impulsive_target is None \
                    or impulsive_weight is None:
                impulsive_loss = torch.zeros((), device=picker_loss.device,
                                             dtype=picker_loss.dtype)
            else:
                impulsive_loss = self.impulsive_loss_fn(
                    impulsive_pred, impulsive_target, impulsive_weight)

        if self.use_eqt_detector_head:
            # v8 detector: (B, 1, T) sigmoid vs detector_target[:, 0:1] (Event channel —
            # block target between P and S, 0 elsewhere). BCE drives output to 0 on noise.
            detector_loss = self.eqt_det_loss_fn(
                detector_pred, detector_target[:, 0:1, :])
        else:
            det_raw = self.detector_loss_fn(detector_pred, detector_target)
            # If sum_over_batch: det_raw is (B, T), reduce to sum(T)/mean(B)
            if self.loss_reduction == 'sum_over_batch' and det_raw.dim() >= 2:
                detector_loss = det_raw.sum(dim=-1).mean()
            else:
                detector_loss = det_raw if det_raw.dim() == 0 else det_raw.mean()

        if (polarity_raw is not None and ungated_polarity_weight > 0
                and self.polarity_loss_type not in ('softmax_ce', 'softmax_ud')):
            ungated = self.polarity_loss_fn(
                polarity_raw, polarity_target, y_weight=polarity_mask)
            polarity_loss = ((1.0 - ungated_polarity_weight) * polarity_loss
                             + ungated_polarity_weight * ungated)

        # Event center auxiliary head (v7 hybrid)
        if self.use_event_center:
            if event_center_pred is None or event_center_target is None:
                event_center_loss = torch.zeros((), device=picker_loss.device,
                                                dtype=picker_loss.dtype)
            else:
                event_center_loss = self.event_center_loss_fn(
                    event_center_pred, event_center_target)

        w = self.get_weights().to(picker_loss.device)
        # DWA task ordering (must match __init__ weight_list construction):
        #   [picker, polarity, (impulsive,) detector, (event_center)]
        idx = 0
        total = w[idx] * picker_loss; idx += 1
        total = total + w[idx] * polarity_loss; idx += 1
        if self.use_impulsive:
            total = total + w[idx] * impulsive_loss; idx += 1
        total = total + w[idx] * detector_loss; idx += 1
        if self.use_event_center:
            total = total + w[idx] * event_center_loss; idx += 1
        if return_individual:
            outputs = [total, picker_loss, polarity_loss]
            if self.use_impulsive:
                outputs.append(impulsive_loss)
            outputs.append(detector_loss)
            if self.use_event_center:
                outputs.append(event_center_loss)
            return tuple(outputs)
        return total
