"""
Model optimization utilities for PyTorch RED-PAN.

Provides optimization functions for faster training and inference.
"""

import torch
import torch.nn as nn
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)


def optimize_for_inference(model: nn.Module, use_compile: bool = True) -> nn.Module:
    """
    Optimize model for inference.
    
    Applies:
    1. Fused Conv + BatchNorm (when in eval mode)
    2. torch.compile for graph optimization (PyTorch 2.0+)
    
    Args:
        model: The model to optimize
        use_compile: Whether to use torch.compile
    
    Returns:
        Optimized model
    """
    model.eval()
    
    # Fuse Conv + BN pairs
    model = _fuse_conv_bn(model)
    
    # Apply torch.compile if available (PyTorch 2.0+)
    if use_compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("Applied torch.compile optimization")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")
    
    return model


def _fuse_conv_bn(model: nn.Module) -> nn.Module:
    """
    Fuse Conv1d + BatchNorm1d pairs for faster inference.
    
    This is only effective when model.eval() has been called.
    """
    from torch.nn.utils.fusion import fuse_conv_bn_eval
    
    fused_count = 0
    
    # Find and fuse Conv + BN pairs
    for name, module in model.named_modules():
        if hasattr(module, 'conv') and hasattr(module, 'bn'):
            if isinstance(module.conv, nn.Conv1d) and isinstance(module.bn, nn.BatchNorm1d):
                try:
                    fused_conv = fuse_conv_bn_eval(module.conv, module.bn)
                    module.conv = fused_conv
                    module.bn = nn.Identity()
                    fused_count += 1
                except Exception:
                    pass
    
    if fused_count > 0:
        logger.info(f"Fused {fused_count} Conv+BN pairs")
    
    return model


def optimize_for_training(model: nn.Module, use_compile: bool = True) -> nn.Module:
    """
    Optimize model for training.
    
    Args:
        model: The model to optimize
        use_compile: Whether to use torch.compile
    
    Returns:
        Optimized model
    """
    if use_compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode="default")
            logger.info("Applied torch.compile for training")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")
    
    return model


def enable_gradient_checkpointing(model: nn.Module) -> nn.Module:
    """
    Enable gradient checkpointing to reduce memory usage.
    
    Trades compute for memory by recomputing activations during backward pass.
    
    Args:
        model: The model to modify
    
    Returns:
        Model with checkpointing enabled
    """
    from torch.utils.checkpoint import checkpoint
    
    # Store original forward methods
    original_forwards = {}
    
    def make_checkpointed_forward(module, original_forward):
        def forward(*args, **kwargs):
            return checkpoint(original_forward, *args, use_reentrant=False, **kwargs)
        return forward
    
    # Apply to RRConv blocks (most memory intensive)
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'RRConvBlock':
            original_forwards[name] = module.forward
            module.forward = make_checkpointed_forward(module, module.forward)
    
    logger.info(f"Enabled gradient checkpointing for {len(original_forwards)} blocks")
    
    return model


def get_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    use_fused: bool = True,
    optimizer_type: str = 'adamw',
) -> torch.optim.Optimizer:
    """
    Create an optimized optimizer.
    
    Args:
        model: Model to optimize
        lr: Learning rate
        weight_decay: Weight decay
        use_fused: Use fused CUDA kernel (PyTorch 2.0+)
        optimizer_type: 'adamw', 'adam', or 'sgd'
    
    Returns:
        Configured optimizer
    """
    params = [p for p in model.parameters() if p.requires_grad]
    
    if optimizer_type == 'adamw':
        # Check if fused is available and model is on CUDA
        fused_available = 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames
        params_on_cuda = all(p.is_cuda for p in params)
        
        if use_fused and fused_available and torch.cuda.is_available() and params_on_cuda:
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, fused=True)
            logger.info("Using fused AdamW optimizer")
        else:
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
            
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
        
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = 'cosine',
    epochs: int = 100,
    warmup_epochs: int = 10,
    min_lr: float = 1e-6,
    steps_per_epoch: Optional[int] = None,
) -> Any:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer instance
        scheduler_type: 'cosine', 'plateau', 'step', or 'linear'
        epochs: Total epochs
        warmup_epochs: Warmup epochs
        min_lr: Minimum learning rate
    
    Returns:
        Learning rate scheduler
    """
    if scheduler_type == 'cosine':
        # Cosine annealing with warmup. If steps_per_epoch is provided, operate in step units.
        if steps_per_epoch is not None:
            total_steps = max(1, epochs * steps_per_epoch)
            warmup_steps = max(0, warmup_epochs * steps_per_epoch)
            main_steps = max(1, total_steps - warmup_steps)
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=main_steps, eta_min=min_lr
            )
            if warmup_steps > 0:
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.1, total_iters=warmup_steps
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, main_scheduler],
                    milestones=[warmup_steps]
                )
            else:
                scheduler = main_scheduler
            scheduler.step_per_batch = True
        else:
            # Epoch-based fallback
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=min_lr
            )
            if warmup_epochs > 0:
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.01, total_iters=warmup_epochs
                )
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer, 
                    schedulers=[warmup_scheduler, main_scheduler],
                    milestones=[warmup_epochs]
                )
            else:
                scheduler = main_scheduler
            scheduler.step_per_batch = False
            
    elif scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=min_lr
        )
        scheduler.step_per_batch = False
        
    elif scheduler_type == 'step':
        if steps_per_epoch is not None:
            # Decay every 30 epochs, converted to steps
            step_size = max(1, 30 * steps_per_epoch)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=step_size, gamma=0.1
            )
            scheduler.step_per_batch = True
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1
            )
            scheduler.step_per_batch = False
        
    elif scheduler_type == 'linear':
        base_lr = optimizer.param_groups[0].get('lr', 1e-3)
        end_factor = min_lr / max(base_lr, 1e-12)
        if steps_per_epoch is not None:
            total_steps = max(1, epochs * steps_per_epoch)
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=end_factor, total_iters=total_steps
            )
            scheduler.step_per_batch = True
        else:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=end_factor, total_iters=epochs
            )
            scheduler.step_per_batch = False
        
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
    return scheduler


def profile_model(
    model: nn.Module,
    input_shape: tuple = (1, 3, 6000),
    device: Optional[torch.device] = None,
    warmup_runs: int = 5,
    timed_runs: int = 20,
) -> dict:
    """
    Profile model performance.
    
    Args:
        model: Model to profile
        input_shape: Input tensor shape
        device: Device to run on
        warmup_runs: Number of warmup runs
        timed_runs: Number of timed runs
    
    Returns:
        Dict with profiling results
    """
    import time
    
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    x = torch.randn(input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(x)
    
    # Synchronize
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(timed_runs):
            start = time.perf_counter()
            _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    
    # Memory stats
    if device.type == 'cuda':
        max_memory = torch.cuda.max_memory_allocated(device) / 1e6
        torch.cuda.reset_peak_memory_stats(device)
    else:
        max_memory = 0
    
    return {
        'device': str(device),
        'input_shape': input_shape,
        'mean_time_ms': sum(times) / len(times) * 1000,
        'std_time_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5 * 1000,
        'throughput_samples_per_sec': input_shape[0] / (sum(times) / len(times)),
        'max_memory_mb': max_memory,
        'parameters': sum(p.numel() for p in model.parameters()),
    }


if __name__ == '__main__':
    from redpan_motion.models.mtan_r2unet import MTAN_R2UNet
    
    print("Testing optimization utilities...")
    
    # Create model
    model = MTAN_R2UNet()
    
    # Profile before optimization
    print("\nBefore optimization:")
    results = profile_model(model)
    for k, v in results.items():
        print(f"  {k}: {v}")
    
    # Optimize
    model = optimize_for_inference(model, use_compile=False)  # Compile takes time
    
    # Profile after optimization
    print("\nAfter fusing Conv+BN:")
    results = profile_model(model)
    for k, v in results.items():
        print(f"  {k}: {v}")
    
    print("\n✓ Optimization utilities test passed!")
