"""Training loop for RED-PAN PyTorch.

Supports mixed precision (AMP), gradient accumulation, learning-rate scheduling,
checkpointing, and multi-GPU (DistributedDataParallel).  Multi-task losses are
balanced with DWA (Dynamic Weight Averaging).
"""

import os
import time
import logging
import pickle
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

from redpan_motion.training.losses import MultiTaskLoss

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Basic training
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Optimization
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    use_amp: bool = True
    
    # DWA
    use_dwa: bool = True
    dwa_temperature: float = 2.0
    
    # Loss Weighting (P12 Alignment)
    weight_mode: str = 'balanced'
    weight_alpha: float = 1.0
    weight_max: float = 50.0
    
    # Learning rate schedule
    lr_scheduler: str = 'cosine'  # 'cosine', 'step', 'none'
    warmup_epochs: int = 5
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    
    # Checkpointing
    checkpoint_dir: str = './checkpoints'
    save_every: int = 1
    
    # Logging
    log_every: int = 10
    use_progress_bar: bool = True
    debug_batch_stats: bool = False
    debug_batch_every: int = 200
    debug_loss_stats: bool = False
    debug_loss_every: int = 200
    skip_nan_batches: bool = False
    label_order: str = 'PSN'
    picker_output_permutation: Optional[Tuple[int, int, int]] = None
    detector_output_permutation: Optional[Tuple[int, int]] = None
    
    # Data loading
    num_workers: int = 4


class REDPANTrainer:
    """
    Trainer for RED-PAN model.
    
    Handles the complete training loop with best practices for
    memory efficiency and training stability.
    
    Example:
        model = MTAN_R2UNet()
        trainer = REDPANTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=TrainingConfig(epochs=100, batch_size=32)
        )
        trainer.train()
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            model: The MTAN_R2UNet model (can be DataParallel wrapped)
            train_loader: Training data loader
            val_loader: Optional validation data loader
            config: Training configuration
            optimizer: Optional custom optimizer
            scheduler: Optional custom LR scheduler
            device: Device to train on
        """
        self.config = config or TrainingConfig()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model (Move to device first)
        self.model = model.to(self.device)
        
        # Check if already DataParallel or needs wrapping
        # (Handling is usually done outside, but we can verify here)
        if isinstance(self.model, nn.DataParallel):
            logger.info(f"Trainer initializing with DataParallel model on {len(self.model.device_ids)} GPUs")
        
        # Data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Optimizer
        # Optimization
        self.optimizer = optimizer or (self.model.optimizer if hasattr(self.model, 'optimizer') else None)
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
            
        self.scheduler = scheduler
        # Check if scheduler should step per batch
        self.scheduler_step_per_batch = getattr(scheduler, 'step_per_batch', False)
        
        # Loss function
        from redpan_motion.training.losses import MultiTaskLoss
        self.loss_fn = MultiTaskLoss(
            picker_weight=1.0,
            detector_weight=1.0,
            use_dwa=self.config.use_dwa,
            dwa_temperature=self.config.dwa_temperature,
            weight_mode=self.config.weight_mode,
            weight_alpha=self.config.weight_alpha,
            weight_max=self.config.weight_max,
            label_order=self.config.label_order,
        )
        
        # Mixed precision — prefer bfloat16 (float32 dynamic range, no overflow/NaN)
        # over float16 (overflows at ~65504, causes NaN in deep MTAN blocks).
        # GradScaler is only needed for float16 (loss scaling prevents underflow).
        if self.config.use_amp and torch.cuda.is_available():
            # V100 (CC 7.0) reports bf16 as "supported" but only in software emulation (slow).
            # Require Ampere+ (CC >= 8.0) for hardware-accelerated bf16.
            bf16_ok = torch.cuda.is_bf16_supported() and torch.cuda.get_device_capability()[0] >= 8
            self.amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
            self.scaler = None if bf16_ok else GradScaler('cuda')
            logger.info(
                "AMP enabled: dtype=%s%s",
                self.amp_dtype,
                "" if bf16_ok else " (GradScaler active)"
            )
        else:
            self.amp_dtype = None
            self.scaler = None
        self.label_order = self.config.label_order.upper()
        if self.label_order != 'PSN':
            logger.warning(
                "label_order='%s' is deprecated; all datasets use PSN [P, S, Noise]. "
                "Forcing label_order='PSN'.", self.label_order)
            self.label_order = 'PSN'
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'picker_loss': [],
            'detector_loss': [],
            'learning_rate': [],
            'dwa_weights': [],
        }
        
        # Create checkpoint directory
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)

    def _apply_output_permutation(
        self,
        picker_pred: torch.Tensor,
        detector_pred: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reorder model outputs to align with target label/mask channel order."""
        if self.config.picker_output_permutation:
            perm = self.config.picker_output_permutation
            if picker_pred.dim() == 3 and picker_pred.shape[1] == 3:
                picker_pred = picker_pred[:, perm, :]
        if self.config.detector_output_permutation:
            perm = self.config.detector_output_permutation
            if detector_pred.dim() == 3 and detector_pred.shape[1] == 2:
                detector_pred = detector_pred[:, perm, :]
        return picker_pred, detector_pred
    
    def _create_scheduler(self) -> Optional[_LRScheduler]:
        """Create learning rate scheduler."""
        if self.config.lr_scheduler == 'none':
            return None
        
        total_steps = len(self.train_loader) * self.config.epochs
        warmup_steps = len(self.train_loader) * self.config.warmup_epochs
        
        if self.config.lr_scheduler == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(self.optimizer, T_max=total_steps - warmup_steps)
        elif self.config.lr_scheduler == 'step':
            from torch.optim.lr_scheduler import StepLR
            return StepLR(self.optimizer, step_size=30, gamma=0.5)
        
        return None
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_picker_loss = 0.0
        total_detector_loss = 0.0
        total_seq_len = 0
        n_batches = 0
        
        batch_iterator = self.train_loader
        if self.config.use_progress_bar:
            batch_iterator = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.config.epochs}", unit="batch")
            
        for batch_idx, batch in enumerate(batch_iterator):
            # Move to device
            waveform = batch['waveform'].to(self.device, non_blocking=True)
            label = batch['label'].to(self.device, non_blocking=True)
            mask = batch['mask'].to(self.device, non_blocking=True)

            # Optional debug stats (label/mask peaks)
            if self.config.debug_batch_stats and (batch_idx % self.config.debug_batch_every == 0):
                with torch.no_grad():
                    p_max = s_max = sig_max = noise_max = None
                    # Labels are PSN: P=ch0, S=ch1, Noise=ch2
                    if label.dim() == 3:
                        if label.shape[1] == 3 and label.shape[2] != 3:
                            p_max = label[:, 0, :].max().item()
                            s_max = label[:, 1, :].max().item()
                        elif label.shape[-1] == 3:
                            p_max = label[..., 0].max().item()
                            s_max = label[..., 1].max().item()
                    if mask.dim() == 3:
                        if mask.shape[1] == 2 and mask.shape[2] != 2:
                            sig_max = mask[:, 0, :].max().item()
                            noise_max = mask[:, 1, :].max().item()
                        elif mask.shape[-1] == 2:
                            sig_max = mask[..., 0].max().item()
                            noise_max = mask[..., 1].max().item()

                    logger.info(
                        "Batch %d debug stats | label P max: %.6f | label S max: %.6f | "
                        "mask signal max: %.6f | mask noise max: %.6f",
                        batch_idx, p_max or 0.0, s_max or 0.0, sig_max or 0.0, noise_max or 0.0
                    )
            
            # Forward pass with mixed precision
            with autocast('cuda', dtype=self.amp_dtype, enabled=self.config.use_amp):
                picker_pred, detector_pred = self.model(waveform)
                picker_pred, detector_pred = self._apply_output_permutation(
                    picker_pred, detector_pred
                )
                loss, picker_loss, detector_loss = self.loss_fn(
                    picker_pred, detector_pred, label, mask,
                    return_individual=True
                )
                
                # Scale for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps

            if not torch.isfinite(loss):
                logger.error("Non-finite loss at batch %d (loss=%s)", batch_idx, loss.item())
                if self.config.skip_nan_batches:
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                raise ValueError("Non-finite loss encountered; try lowering LR or disabling AMP.")

            # Optional debug loss scaling stats
            if self.config.debug_loss_stats and (batch_idx % self.config.debug_loss_every == 0):
                seq_len = waveform.shape[-1]
                loss_unscaled = loss.item() * self.config.gradient_accumulation_steps
                picker_loss_val = picker_loss.item()
                detector_loss_val = detector_loss.item()
                logger.info(
                    "Batch %d loss stats | loss: %.4f (per_ts: %.6f) | "
                    "picker: %.4f (per_ts: %.6f) | detector: %.4f (per_ts: %.6f) | T=%d",
                    batch_idx,
                    loss_unscaled, loss_unscaled / max(seq_len, 1),
                    picker_loss_val, picker_loss_val / max(seq_len, 1),
                    detector_loss_val, detector_loss_val / max(seq_len, 1),
                    seq_len,
                )
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step (with gradient accumulation)
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad(set_to_none=True)
                
                # Step scheduler if per-batch
                if self.scheduler is not None and self.scheduler_step_per_batch:
                    self.scheduler.step()
                
                self.global_step += 1
            
            # Accumulate losses
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            total_picker_loss += picker_loss.item()
            total_detector_loss += detector_loss.item()
            total_seq_len += waveform.shape[-1]

            n_batches += 1
            
            current_avg_loss = total_loss / n_batches
            
            # Update progress bar
            if self.config.use_progress_bar:
                # Type check to satisfy linter if batch_iterator is tqdm
                if hasattr(batch_iterator, 'set_postfix'):
                    batch_iterator.set_postfix(loss=f"{current_avg_loss:.4f}", refresh=False)
            
            # Logging (only if progress bar is disabled)
            if not self.config.use_progress_bar and (batch_idx + 1) % self.config.log_every == 0:
                lr = self.optimizer.param_groups[0]['lr']
                logger.info(
                    f"Epoch {self.current_epoch + 1} | "
                    f"Batch {batch_idx + 1}/{len(self.train_loader)} | "
                    f"Loss: {current_avg_loss:.4f}"
                )
        
        # Update DWA weights using balanced CE (tracks actual optimization progress)
        avg_picker_loss = total_picker_loss / n_batches if n_batches > 0 else 0.0
        avg_detector_loss = total_detector_loss / n_batches if n_batches > 0 else 0.0
        if self.config.use_dwa:
            self.loss_fn.update_weights(torch.tensor([avg_picker_loss, avg_detector_loss]))
        
        return {
            'loss': total_loss / n_batches if n_batches > 0 else 0.0,
            'picker_loss': avg_picker_loss,
            'detector_loss': avg_detector_loss,
            'seq_len': (total_seq_len / n_batches) if n_batches > 0 else 0.0,
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_picker_loss = 0.0
        total_detector_loss = 0.0
        total_seq_len = 0
        n_batches = 0
        
        batch_iterator = self.val_loader
        if self.config.use_progress_bar:
            batch_iterator = tqdm(self.val_loader, desc="Validation", unit="batch", leave=False)
        
        for batch in batch_iterator:
            waveform = batch['waveform'].to(self.device, non_blocking=True)
            label = batch['label'].to(self.device, non_blocking=True)
            mask = batch['mask'].to(self.device, non_blocking=True)
            
            with autocast('cuda', dtype=self.amp_dtype, enabled=self.config.use_amp):
                picker_pred, detector_pred = self.model(waveform)
                picker_pred, detector_pred = self._apply_output_permutation(
                    picker_pred, detector_pred
                )
                loss, picker_loss, detector_loss = self.loss_fn(
                    picker_pred, detector_pred, label, mask,
                    return_individual=True
                )
            
            total_loss += loss.item()
            total_picker_loss += picker_loss.item()
            total_detector_loss += detector_loss.item()
            total_seq_len += waveform.shape[-1]
            n_batches += 1
        
        return {
            'loss': total_loss / n_batches if n_batches > 0 else 0.0,
            'picker_loss': total_picker_loss / n_batches if n_batches > 0 else 0.0,
            'detector_loss': total_detector_loss / n_batches if n_batches > 0 else 0.0,
            'seq_len': (total_seq_len / n_batches) if n_batches > 0 else 0.0,
        }
    
    def train(self) -> Dict[str, list]:
        """
        Run full training loop.
        
        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training on {self.device}")
        
        # Check for model size (handle DataParallel unwrapping for checking)
        model_to_check = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        logger.info(f"Model parameters: {sum(p.numel() for p in model_to_check.parameters()):,}")
        
        logger.info(f"Config: {self.config}")
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate()
            
            # Record history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['picker_loss'].append(train_metrics['picker_loss'])
            self.history['detector_loss'].append(train_metrics['detector_loss'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            if val_metrics:
                self.history['val_loss'].append(val_metrics['loss'])
            
            if self.config.use_dwa:
                self.history['dwa_weights'].append(self.loss_fn.get_weights().tolist())
            
            # Logging
            epoch_time = time.time() - epoch_start
            train_seq_len = max(int(round(train_metrics.get('seq_len', 0))), 1)
            train_per_ts = train_metrics['loss'] / train_seq_len if train_seq_len > 0 else train_metrics['loss']
            if val_metrics:
                val_seq_len = max(int(round(val_metrics.get('seq_len', 0))), 1)
                val_per_ts = val_metrics['loss'] / val_seq_len if val_seq_len > 0 else val_metrics['loss']
                val_str = f"Val Loss: {val_metrics['loss']:.4f} (per_ts: {val_per_ts:.6f})"
            else:
                val_str = "No validation"
            dwa_str = f"DWA: {self.loss_fn.get_weights().tolist()}" if self.config.use_dwa else ""
            
            logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} (per_ts: {train_per_ts:.6f}) | "
                f"{val_str} | {dwa_str} | Time: {epoch_time:.1f}s"
            )

            # Scheduler step (epoch-based)
            if self.scheduler is not None and not self.scheduler_step_per_batch:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    metric = val_metrics.get('loss') if val_metrics else train_metrics['loss']
                    self.scheduler.step(metric)
                else:
                    self.scheduler.step()
            
            # Checkpointing (skip best-model tracking during warmup —
            # untrained models produce artificially low balanced CE loss)
            if val_metrics:
                val_loss = val_metrics['loss']
                if epoch < self.config.warmup_epochs:
                    logger.info(
                        "Warmup epoch %d/%d — skipping best-model tracking (val_loss=%.6f)",
                        epoch + 1, self.config.warmup_epochs, val_loss,
                    )
                elif val_loss < self.best_val_loss - self.config.min_delta:
                    prev_best = self.best_val_loss
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                    epoch_dir = f"epoch_{epoch + 1:04d}"
                    self.save_checkpoint(os.path.join(epoch_dir, 'best.pt'))
                    logger.info(
                        "val_loss improved from %.6f to %.6f, saved %s",
                        prev_best,
                        val_loss,
                        os.path.join(self.config.checkpoint_dir, epoch_dir, 'best.pt'),
                    )
                else:
                    self.epochs_without_improvement += 1
                    logger.info(
                        "Validation loss %.4f did not improve from %.4f for %d epochs.",
                        val_loss,
                        self.best_val_loss,
                        self.epochs_without_improvement,
                    )

                # Early stopping (only after warmup)
                if epoch >= self.config.warmup_epochs and self.config.patience and self.config.patience > 0:
                    if self.epochs_without_improvement >= self.config.patience:
                        logger.info(f"Early stopping after {epoch + 1} epochs")
                        break
            
            # Periodic checkpoints
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f'epoch_{epoch + 1:04d}.pt')
        
        # Save final model
        self.save_checkpoint('final.pt')
        
        return self.history
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        path = os.path.join(self.config.checkpoint_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Unwrap torch.compile, DDP, and DataParallel wrappers before saving
        model_to_save = self.model
        if hasattr(model_to_save, '_orig_mod'):           # torch.compile
            model_to_save = model_to_save._orig_mod
        if isinstance(model_to_save, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            model_to_save = model_to_save.module
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
        except pickle.UnpicklingError as exc:
            msg = str(exc)
            if "Weights only load failed" not in msg:
                raise
            warnings.warn(
                "Falling back to torch.load(weights_only=False) for checkpoint compatibility. "
                "Only do this for trusted checkpoints.",
                RuntimeWarning,
            )
            try:
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            except TypeError:
                checkpoint = torch.load(path, map_location=self.device)
        
        # Unwrap DataParallel if loading
        model_to_load = self.model.module if isinstance(self.model, nn.DataParallel) else self.model

        # Robust state_dict loading: normalize away torch.compile (`_orig_mod.`)
        # and DDP (`module.`) prefixes from both sides, then remap to the
        # current model's actual key names. This handles all combinations of
        # save/load wrapping (compiled vs eager, single-GPU vs DDP).
        state_dict = checkpoint['model_state_dict']

        def _normalize_key(k: str) -> str:
            # Strip in order from longest to shortest to avoid partial matches
            for prefix in ('_orig_mod.module.', 'module._orig_mod.', '_orig_mod.', 'module.'):
                if k.startswith(prefix):
                    return k[len(prefix):]
            return k

        model_state = model_to_load.state_dict()
        target_by_norm = {_normalize_key(k): k for k in model_state.keys()}

        remapped: Dict[str, torch.Tensor] = {}
        unmatched_ckpt = []
        for k, v in state_dict.items():
            norm_k = _normalize_key(k)
            if norm_k in target_by_norm:
                remapped[target_by_norm[norm_k]] = v
            else:
                unmatched_ckpt.append(k)

        unmatched_model = [k for k in model_state.keys() if k not in remapped]

        if unmatched_ckpt:
            logger.warning(f"Checkpoint had {len(unmatched_ckpt)} keys not in current model "
                           f"(first 3: {unmatched_ckpt[:3]})")
        if unmatched_model:
            logger.warning(f"Current model has {len(unmatched_model)} keys not in checkpoint "
                           f"(first 3: {unmatched_model[:3]}) — these will keep their initial values")

        # strict=False to tolerate num_batches_tracked and minor schema drift
        missing, unexpected = model_to_load.load_state_dict(remapped, strict=False)
        logger.info(f"Loaded {len(remapped)} state_dict tensors "
                    f"({len(missing)} missing, {len(unexpected)} unexpected)")
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")


if __name__ == '__main__':
    # Quick test
    import numpy as np
    from redpan_motion.models.mtan_r2unet import MTAN_R2UNet
    from redpan_motion.data.dataset import NumpySeismicDataset, create_dataloader
    
    # Create dummy data
    n_samples = 100
    waveforms = np.random.randn(n_samples, 3, 6000).astype(np.float32)
    labels = np.zeros((n_samples, 6000, 3), dtype=np.float32)
    labels[..., 2] = 1.0  # All noise
    masks = np.zeros((n_samples, 6000, 2), dtype=np.float32)
    masks[..., 1] = 1.0  # All unmask
    
    # Create dataset and loader
    dataset = NumpySeismicDataset(waveforms, labels, masks)
    loader = create_dataloader(dataset, batch_size=8, num_workers=0)
    
    # Create model and trainer
    model = MTAN_R2UNet()
    config = TrainingConfig(
        epochs=2,
        batch_size=8,
        log_every=5,
        checkpoint_dir='/tmp/test_checkpoints',
    )
    
    trainer = REDPANTrainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        config=config,
    )
    
    # Train for a few epochs
    history = trainer.train()
    
    print(f"\nTraining complete!")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    print("\n✓ Trainer test passed!")
