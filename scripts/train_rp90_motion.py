#!/usr/bin/env python3
"""
Training script for RED-PAN RP90-Motion (mixed-stride MTAN R2U-Net with polarity).

Supports single-GPU and multi-GPU (DDP via torchrun).

Usage:
    # Single GPU
    python scripts/train_rp90_motion.py --config configs/train_rp90_motion.json

    # Multi-GPU (4× GPU)
    torchrun --nproc_per_node=4 scripts/train_rp90_motion.py --config configs/train_rp90_motion.json --num-workers 4
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import inspect as _inspect

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from redpan_motion.models import (  # noqa: E402
    build_mtan_r2unet_rp90_motion,
    build_edge_rp90,
)
from redpan_motion.data.dataset_v2 import MultiDatasetH5  # noqa: E402
from redpan_motion.training.losses import MultiTaskLossWithPolarity  # noqa: E402

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train RED-PAN RP90-Motion")
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def resolve_data_dirs(cfg) -> list:
    # The DATA_ROOT environment variable overrides the config's data_root.
    data_root = os.environ.get("DATA_ROOT") or cfg.get("data_root", "")
    if "data" in cfg and isinstance(cfg["data"], dict):
        dirs = []
        for name, info in cfg["data"].items():
            path = info.get("file_path", "").replace("DATA_ROOT", data_root)
            if os.path.isdir(path):
                dirs.append(path)
            else:
                logger.warning(f"  {name}: {path} not found, skipping")
        return dirs
    if "data_dirs" in cfg:
        return [d for d in cfg["data_dirs"] if os.path.isdir(d)]
    return []


def build_datasets(cfg, rank=0):
    data_length = cfg["input_size"][0]
    mask_window = cfg.get("mask_window", 40)
    # Separate polarity target width — narrower Gaussian gives the tanh head
    # sharper peaks to chase, recovering gradient drive lost to the graduated
    # target shape. Defaults to mask_window for backward compat.
    polarity_mask_window = cfg.get("polarity_mask_window", mask_window)
    seed = cfg.get("seed", 42)
    cat_weights = cfg.get("category_weights")
    data_dirs = resolve_data_dirs(cfg)
    if rank == 0:
        logger.info(f"Data dirs: {data_dirs}")
        if polarity_mask_window != mask_window:
            logger.info(
                f"mask_window: picker/detector={mask_window}, polarity={polarity_mask_window}")

    # Target shape for polarity auto-derived from loss type to keep them in sync.
    polarity_loss_type = cfg.get("polarity_loss_type", "bce")
    if polarity_loss_type == 'softmax_ce':
        polarity_target_type = 'softmax_ce'
    elif polarity_loss_type == 'softmax_ud':
        polarity_target_type = 'softmax_ud'
    else:
        polarity_target_type = 'signed'
    # PhaseNet+ style widths for softmax_ce / softmax_ud paths.
    polarity_label_width = cfg.get("polarity_label_width", 20)
    polarity_mask_width = cfg.get("polarity_mask_width", 30)
    if rank == 0 and polarity_target_type in ('softmax_ce', 'softmax_ud'):
        logger.info(
            f"Polarity ({polarity_target_type}) widths: label={polarity_label_width} samples, "
            f"mask={polarity_mask_width} samples (binary)")
    # Full-trace supervision of the impulsive head (v26): >0 → impulsive target is
    # 0 off-pick everywhere (clean ≈0 baseline + ≈1 bump at impulsive picks);
    # 0 (default) → pick-window-only (legacy: head floats at ~0.5 off-pick).
    impulsive_bg_weight = float(cfg.get("impulsive_bg_weight", 0.0))
    if rank == 0 and impulsive_bg_weight > 0.0:
        logger.info(f"impulsive_bg_weight={impulsive_bg_weight} — full-trace impulsive supervision (target 0 off-pick)")

    # Polarity oversample upweights polarity-bearing datasets (e.g. CEED)
    # within each category. Applied to train only — val keeps natural
    # composition so pol-val numbers remain comparable across runs.
    polarity_oversample = float(cfg.get("polarity_oversample", 1.0))
    if rank == 0 and polarity_oversample > 1.0:
        logger.info(f"polarity_oversample={polarity_oversample} (train only)")

    # Per-dataset highpass + category exclusion (v4 additions).
    dataset_hp_freqs = cfg.get("dataset_hp_freqs", {}) or {}
    exclude_categories = cfg.get("exclude_categories", []) or []
    # Per-dataset category whitelist (v19 addition). Format:
    #   {ds_name: [allowed_file_cats]}. Datasets absent → no filter.
    # Use to keep CEED full while restricting other sources to noise only.
    dataset_categories = cfg.get("dataset_categories", {}) or {}
    # Normalization mode (v16 addition: 'moving' for PhaseNet+/EQNet-style
    # sliding-window normalization; 'zscore' for legacy global z-score).
    normalize_mode = cfg.get("normalize_mode", "zscore")
    moving_filter_size = int(cfg.get("moving_filter_size", 1024))
    if rank == 0:
        if dataset_hp_freqs:
            logger.info(f"dataset_hp_freqs (zero-phase Butterworth HP): {dataset_hp_freqs}")
        if exclude_categories:
            logger.info(f"exclude_categories (skipped during _discover): {exclude_categories}")
        if dataset_categories:
            logger.info(f"dataset_categories (per-dataset file_cat whitelist): {dataset_categories}")
        logger.info(f"normalize_mode={normalize_mode} (moving_filter_size={moving_filter_size} samples)")

    train_ds = MultiDatasetH5(
        data_dirs=data_dirs, split='train',
        data_length=data_length, mask_window=mask_window,
        polarity_mask_window=polarity_mask_window,
        polarity_target_type=polarity_target_type,
        polarity_label_width=polarity_label_width,
        polarity_mask_width=polarity_mask_width,
        impulsive_bg_weight=impulsive_bg_weight,
        samples_per_epoch=cfg.get("samples_per_epoch", 500000),
        category_weights=cat_weights,
        polarity_oversample=polarity_oversample, seed=seed,
        dataset_hp_freqs=dataset_hp_freqs,
        exclude_categories=exclude_categories,
        dataset_categories=dataset_categories,
        normalize_mode=normalize_mode,
        moving_filter_size=moving_filter_size,
    )
    val_ds = MultiDatasetH5(
        data_dirs=data_dirs, split='val',
        data_length=data_length, mask_window=mask_window,
        polarity_mask_window=polarity_mask_window,
        polarity_target_type=polarity_target_type,
        polarity_label_width=polarity_label_width,
        polarity_mask_width=polarity_mask_width,
        impulsive_bg_weight=impulsive_bg_weight,
        samples_per_epoch=cfg.get("val_samples_per_epoch", 50000),
        category_weights=cat_weights, seed=seed + 1,
        dataset_hp_freqs=dataset_hp_freqs,
        exclude_categories=exclude_categories,
        dataset_categories=dataset_categories,
        normalize_mode=normalize_mode,
        moving_filter_size=moving_filter_size,
    )
    return train_ds, val_ds


POLARITY_PREFIXES = (
    "pol_init_rrconv", "pol_init_proj", "init_pol_mtan", "enc_pol_mtans",
    "dec_pol_mtans", "pol_z_block", "polarity_head", "impulsive_head",
)


def _is_polarity_name(name: str) -> bool:
    return any(name.startswith(p) for p in POLARITY_PREFIXES)


def freeze_backbone(model):
    """Freeze all non-polarity params. Returns (frozen_count, trainable_count)."""
    base = model.module if hasattr(model, "module") else model
    frozen = trainable = 0
    for name, p in base.named_parameters():
        if _is_polarity_name(name):
            p.requires_grad = True
            trainable += p.numel()
        else:
            p.requires_grad = False
            frozen += p.numel()
    return frozen, trainable


def set_train_mode_respecting_freeze(model, freeze):
    """Call inside train loop. Keeps frozen modules in eval() so BN stats don't drift."""
    model.train()
    if not freeze:
        return
    base = model.module if hasattr(model, "module") else model
    for name, m in base.named_modules():
        if name == "":
            continue
        if not _is_polarity_name(name):
            m.eval()


def _snapshot_bn_stats(model):
    snap = {}
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.running_mean is not None:
                snap[name] = (m.running_mean.detach().clone(),
                              m.running_var.detach().clone(),
                              m.num_batches_tracked.detach().clone())
    return snap


def _restore_bn_stats(model, snap):
    for name, m in model.named_modules():
        if name in snap:
            rm, rv, nb = snap[name]
            m.running_mean.copy_(rm)
            m.running_var.copy_(rv)
            m.num_batches_tracked.copy_(nb)


def _reset_bn_stats(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.reset_running_stats()


def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device, cfg,
                    ungated_weight=0.0, amp_dtype=torch.bfloat16, epoch=0):
    set_train_mode_respecting_freeze(model, cfg.get("freeze_backbone", False))
    total_loss = 0.0
    t_pick = t_pol = t_det = t_imp = 0.0
    n = 0
    nan_skips = 0
    use_pol = cfg.get("use_polarity", True)
    amp_enabled = cfg.get("amp", True)
    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0

    pbar = tqdm(loader, desc=f"Epoch {epoch}", unit="batch",
                disable=(rank != 0), leave=False)

    use_imp = cfg.get("use_impulsive_head", False) and use_pol
    use_ec = cfg.get("use_event_center_head", False)
    for batch in pbar:
        # Dataset returns 8-tuple by default; 9-tuple when use_event_center_head=True:
        #   (wf, z_raw, picker, pol, pol_w, imp, imp_w, det, [event_center])
        if use_ec:
            wf, z_raw, pick_t, pol_t, pol_mask, imp_t, imp_w, det_t, ec_t = batch
            ec_t = ec_t.to(device)
        else:
            wf, z_raw, pick_t, pol_t, pol_mask, imp_t, imp_w, det_t = batch
            ec_t = None
        wf = wf.to(device); z_raw = z_raw.to(device)
        pick_t = pick_t.to(device); pol_t = pol_t.to(device); pol_mask = pol_mask.to(device)
        imp_t = imp_t.to(device); imp_w = imp_w.to(device); det_t = det_t.to(device)

        wf = torch.nan_to_num(wf, nan=0.0, posinf=0.0, neginf=0.0)
        z_raw = torch.nan_to_num(z_raw, nan=0.0, posinf=0.0, neginf=0.0)
        pick_t = torch.nan_to_num(pick_t, nan=0.0, posinf=0.0, neginf=0.0)
        pol_t = torch.nan_to_num(pol_t, nan=0.0, posinf=0.0, neginf=0.0)
        det_t = torch.nan_to_num(det_t, nan=0.0, posinf=0.0, neginf=0.0)

        bn_snap = _snapshot_bn_stats(model)

        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
            if use_imp:
                if use_ec:
                    picker, polarity, impulsive, detector, ec_pred = model(wf, z_raw=z_raw)
                else:
                    picker, polarity, impulsive, detector = model(wf, z_raw=z_raw)
                    ec_pred = None
                ret = loss_fn(
                    picker, polarity, detector, pick_t, pol_t, det_t,
                    polarity_mask=pol_mask,
                    impulsive_pred=impulsive, impulsive_target=imp_t,
                    impulsive_weight=imp_w,
                    event_center_pred=ec_pred, event_center_target=ec_t,
                    return_individual=True,
                )
                if use_ec:
                    loss, pl, poll, impl, dl, ecl = ret
                else:
                    loss, pl, poll, impl, dl = ret
            elif use_pol:
                if use_ec:
                    picker, polarity, detector, ec_pred = model(wf, z_raw=z_raw)
                    loss, pl, poll, dl, ecl = loss_fn(
                        picker, polarity, detector, pick_t, pol_t, det_t,
                        polarity_mask=pol_mask,
                        event_center_pred=ec_pred, event_center_target=ec_t,
                        return_individual=True,
                    )
                else:
                    picker, polarity, detector = model(wf, z_raw=z_raw)
                    loss, pl, poll, dl = loss_fn(
                        picker, polarity, detector, pick_t, pol_t, det_t,
                        polarity_mask=pol_mask, return_individual=True,
                    )
            else:
                picker, _, detector = model(wf, z_raw=z_raw)
                loss, pl, poll, dl = loss_fn(
                    picker, None, detector, pick_t, pol_t, det_t,
                    polarity_mask=pol_mask, return_individual=True,
                )

        local_ok = torch.tensor(
            [1.0 if torch.isfinite(loss) else 0.0], device=device)
        if is_dist:
            dist.all_reduce(local_ok, op=dist.ReduceOp.MIN)
        all_ranks_ok = local_ok.item() > 0.0

        if not all_ranks_ok:
            nan_skips += 1
            _restore_bn_stats(model, bn_snap)
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            if cfg.get("gradient_clip"):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.get("gradient_clip"):
                nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"])
            optimizer.step()

        total_loss += loss.item()
        t_pick += pl.item(); t_pol += poll.item(); t_det += dl.item()
        if use_imp:
            t_imp += impl.item()
        n += 1

        # Update tqdm postfix
        if rank == 0 and n > 0:
            pbar.set_postfix(
                loss=f"{total_loss/n:.1f}",
                p=f"{t_pick/n:.1f}", pol=f"{t_pol/n:.1f}",
                **({'imp': f"{t_imp/n:.1f}"} if use_imp else {}),
                d=f"{t_det/n:.1f}",
                nan=nan_skips, refresh=False,
            )

    pbar.close()
    denom = max(n, 1)
    return (total_loss / denom, t_pick / denom, t_pol / denom, t_det / denom,
            t_imp / denom, nan_skips)


@torch.no_grad()
def validate(model, loader, loss_fn, device, cfg, ungated_weight=0.0,
             amp_dtype=torch.bfloat16):
    model.eval()
    total_loss = 0.0
    v_pick = v_pol = v_det = v_imp = 0.0
    n = 0
    skipped = 0
    use_pol = cfg.get("use_polarity", True)
    use_imp = cfg.get("use_impulsive_head", False) and use_pol
    use_ec = cfg.get("use_event_center_head", False)
    amp_enabled = cfg.get("amp", True)
    is_dist = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0

    pbar = tqdm(loader, desc="Validation", unit="batch",
                disable=(rank != 0), leave=False)

    for batch in pbar:
        if use_ec:
            wf, z_raw, pick_t, pol_t, pol_mask, imp_t, imp_w, det_t, ec_t = batch
            ec_t = ec_t.to(device)
        else:
            wf, z_raw, pick_t, pol_t, pol_mask, imp_t, imp_w, det_t = batch
            ec_t = None
        wf = wf.to(device); z_raw = z_raw.to(device)
        pick_t = pick_t.to(device); pol_t = pol_t.to(device); pol_mask = pol_mask.to(device)
        imp_t = imp_t.to(device); imp_w = imp_w.to(device); det_t = det_t.to(device)
        wf = torch.nan_to_num(wf, nan=0.0, posinf=0.0, neginf=0.0)
        z_raw = torch.nan_to_num(z_raw, nan=0.0, posinf=0.0, neginf=0.0)

        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
            if use_imp:
                picker, polarity, impulsive, detector = model(wf, z_raw=z_raw)
                loss, pl, poll, impl, dl = loss_fn(
                    picker, polarity, detector, pick_t, pol_t, det_t,
                    polarity_mask=pol_mask,
                    impulsive_pred=impulsive, impulsive_target=imp_t,
                    impulsive_weight=imp_w,
                    return_individual=True,
                )
            elif use_pol:
                picker, polarity, detector = model(wf, z_raw=z_raw)
                loss, pl, poll, dl = loss_fn(
                    picker, polarity, detector, pick_t, pol_t, det_t,
                    polarity_mask=pol_mask, return_individual=True,
                )
            else:
                picker, _, detector = model(wf, z_raw=z_raw)
                loss, pl, poll, dl = loss_fn(
                    picker, None, detector, pick_t, pol_t, det_t,
                    polarity_mask=pol_mask, return_individual=True,
                )
        if torch.isfinite(loss):
            total_loss += loss.item()
            v_pick += pl.item(); v_pol += poll.item(); v_det += dl.item()
            if use_imp:
                v_imp += impl.item()
            n += 1
        else:
            skipped += 1

        if rank == 0 and n > 0:
            pbar.set_postfix(
                loss=f"{total_loss/n:.1f}",
                p=f"{v_pick/n:.1f}", pol=f"{v_pol/n:.1f}",
                **({'imp': f"{v_imp/n:.1f}"} if use_imp else {}),
                d=f"{v_det/n:.1f}",
                nan=skipped, refresh=False,
            )

    pbar.close()
    denom = max(n, 1)
    return (total_loss / denom, v_pick / denom, v_pol / denom, v_det / denom,
            v_imp / denom, skipped)


def unwrap_model(model):
    """Get base model from DDP/DataParallel wrapper."""
    if hasattr(model, 'module'):
        return model.module
    return model


def main():
    args = parse_args()
    with open(args.config) as _cfg_f:
        cfg = json.load(_cfg_f)

    # ── DDP setup ──
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_distributed = local_rank >= 0
    rank = 0
    world_size = 1

    if is_distributed:
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    if rank == 0:
        logger.info(f"Device: {device}" +
                     (f" (DDP world_size={world_size})" if is_distributed else ""))

    # ── Output dir ──
    out_dir = Path(ROOT / cfg["output_dir"])
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        json.dump(cfg, open(out_dir / "config.json", "w"), indent=2)
        fh = logging.FileHandler(out_dir / f"training_{time.strftime('%Y%m%d_%H%M%S')}.log")
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(fh)

    # ── Model ──
    # Single production architecture: MTAN R2U-Net RP90-Motion (the former *_xl
    # variant is folded in). model_type "mtan_r2unet_rp90_motion[_xl]" both route
    # here; legacy flags (polarity_arch / impulsive / eqt / spectrogram / ...) are
    # no longer parameters — the architecture is fixed (deeper polarity head).
    model_type = cfg.get("model_type", "mtan_r2unet_rp90_motion")

    pol_out_ch = (
        3 if cfg.get("polarity_loss_type", "softmax_ce") == "softmax_ce"
        else 2 if cfg.get("polarity_loss_type", "softmax_ce") == "softmax_ud"
        else 1
    )
    if model_type == "edge_rp90_v1":
        # Edge-deployable model (redpan_motion.models.edge_rp90). Fixed
        # macro-architecture; only these knobs are config-driven.
        _sig = _inspect.signature(build_edge_rp90)
        build_kwargs = dict(
            input_size=tuple(cfg["input_size"]),
            block=cfg.get("block", "pconv"),
            polarity_head=cfg.get("polarity_head", "sign_split"),
            polarity_output_channels=pol_out_ch,
            dropout_rate=cfg.get("dropout_rate", 0.1),
            pad_mode=cfg.get("pad_mode", "reflect"),
            pretrained_weights=cfg.get("pretrained_weights"),
        )
        build_kwargs = {k: v for k, v in build_kwargs.items() if k in _sig.parameters}
        model = build_edge_rp90(**build_kwargs).to(device)
    else:
        _sig = _inspect.signature(build_mtan_r2unet_rp90_motion)
        build_kwargs = dict(
            input_size=tuple(cfg["input_size"]),
            nb_filters=cfg["nb_filters"],
            strides=cfg["strides"],
            kernel_size=cfg.get("kernel_size", 7),
            dropout_rate=cfg.get("dropout_rate", 0.1),
            rrconv_iters=cfg.get("rrconv_iters", 2),
            use_polarity=cfg.get("use_polarity", True),
            polarity_output_channels=pol_out_ch,
            pol_stream_width_mult=cfg.get("pol_stream_width_mult", 3),
            pol_init_rrconv_iters=cfg.get("pol_init_rrconv_iters", 4),
            pol_head_ps_att_width=cfg.get("pol_head_ps_att_width", 16),
            pad_mode=cfg.get("pad_mode", "reflect"),
            pretrained_weights=cfg.get("pretrained_weights"),
        )
        build_kwargs = {k: v for k, v in build_kwargs.items() if k in _sig.parameters}
        model = build_mtan_r2unet_rp90_motion(**build_kwargs).to(device)

    if rank == 0:
        logger.info(f"model_type={model_type}")

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model: {n_params:,} params, filters={cfg['nb_filters']}, strides={cfg['strides']}")

    # Optional backbone freeze (stage-2 polarity-only training).
    if cfg.get("freeze_backbone", False):
        frozen, trainable = freeze_backbone(model)
        if rank == 0:
            logger.info(
                f"freeze_backbone=True → frozen {frozen:,} params, "
                f"trainable {trainable:,} params (polarity stream + head)")

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # ── Data ──
    train_ds, val_ds = build_datasets(cfg, rank=rank)
    if rank == 0:
        logger.info(f"Train: {len(train_ds)}/epoch, Val: {len(val_ds)}/epoch")

    if is_distributed:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size,
                                           rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size,
                                          rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_ds, batch_size=cfg.get("batch_size", 128),
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.get("batch_size", 128),
        shuffle=False, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── Loss / Optimizer / Scheduler ──
    loss_fn = MultiTaskLossWithPolarity(
        weight_mode=cfg.get("weight_mode", "balanced"),
        weight_alpha=cfg.get("weight_alpha", 1.0),
        weight_max=cfg.get("weight_max", 50.0),
        polarity_weight=cfg.get("polarity_loss_weight", 1.0),
        impulsive_weight=cfg.get("impulsive_loss_weight", 1.0),
        use_dwa=cfg.get("dwa", True),
        dwa_temperature=cfg.get("dwa_temperature", 2.0),
        loss_reduction=cfg.get("loss_reduction", "sum_over_batch"),
        polarity_loss_type=cfg.get("polarity_loss_type", "softmax_ce"),
        use_impulsive=cfg.get("use_impulsive_head", False),
        use_event_center=cfg.get("use_event_center_head", False),
        use_eqt_heads=cfg.get("use_eqt_heads", False),
        use_eqt_picker_heads=cfg.get("use_eqt_picker_heads", False),
        use_eqt_detector_head=cfg.get("use_eqt_detector_head", False),
    )

    opt_name = cfg.get("optimizer", "adam").lower()
    lr = cfg.get("lr", 0.005)
    wd = cfg.get("weight_decay", 0)
    # Split params into two groups:
    #   no_decay: polarity_head.weight (graduated Gaussian target → moderate logits;
    #             weight decay starves the already-weak gradient drive), all biases,
    #             and BN affine params (standard practice).
    #   decay:   everything else (backbone, MTAN chains, picker/detector heads).
    # Polarity_head.bias stays in the *decay* group to pull it toward 0 and
    # suppress the uniform-offset drift we've observed on trained checkpoints.
    no_decay_params, decay_params = [], []
    # Iterate the unwrapped model so names don't carry the DDP `module.` prefix.
    # Makes the suffix-matching below explicit rather than coincidentally correct.
    for name, p in unwrap_model(model).named_parameters():
        if not p.requires_grad:
            continue
        is_bias = name.endswith(".bias")
        is_bn = (".bn." in name or name.startswith("bn.")) and name.endswith(".weight")
        is_pol_head_w = name == "polarity_head.weight"
        is_pol_head_b = name == "polarity_head.bias"
        is_imp_head_w = name == "impulsive_head.weight"
        is_imp_head_b = name == "impulsive_head.bias"
        # polarity_head.bias and impulsive_head.bias stay IN decay group
        # (pull their offsets toward 0 to keep output baselines centered).
        exclude_from_decay = is_pol_head_w or is_imp_head_w or is_bn or (
            is_bias and not (is_pol_head_b or is_imp_head_b))
        if exclude_from_decay:
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    param_groups = [
        {"params": decay_params,    "weight_decay": wd},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    if rank == 0:
        logger.info(
            f"Optimizer param groups: decay={sum(p.numel() for p in decay_params):,}  "
            f"no_decay={sum(p.numel() for p in no_decay_params):,} "
            f"(polarity_head.weight, biases except polarity, BN weights)")
    if opt_name == "adamw":
        optimizer = torch.optim.AdamW(param_groups, lr=lr)
    else:
        optimizer = torch.optim.Adam(param_groups, lr=lr)
    warmup = cfg.get("warmup_epochs", 10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.get("epochs", 200) - warmup,
        eta_min=cfg.get("min_lr", 1e-5),
    )
    # AMP dtype selection (matches redpan/torch/training/trainer.py logic):
    #   "auto"  → bf16 on Ampere+ (cc≥8.0), fp16+scaler otherwise
    #   "bf16"  → force bf16 (will emulate in software on cc<8.0 → VERY slow)
    #   "fp16"  → force fp16 + GradScaler
    #   "fp32"  → disable AMP
    amp_dtype_str = cfg.get("amp_dtype", "auto").lower()
    bf16_hw = (torch.cuda.is_available()
               and torch.cuda.is_bf16_supported()
               and torch.cuda.get_device_capability()[0] >= 8)

    if amp_dtype_str == "auto":
        if bf16_hw:
            amp_dtype = torch.bfloat16
            scaler = None
        else:
            amp_dtype = torch.float16
            scaler = torch.amp.GradScaler("cuda", enabled=cfg.get("amp", True))
    elif amp_dtype_str in ("bf16", "bfloat16"):
        if not bf16_hw and rank == 0:
            logger.warning(
                "amp_dtype=bf16 requested but GPU cc<8.0 — will run in SOFTWARE "
                "emulation and be extremely slow. Set amp_dtype=auto or fp16.")
        amp_dtype = torch.bfloat16
        scaler = None
    elif amp_dtype_str in ("fp16", "float16", "half"):
        amp_dtype = torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.get("amp", True))
    else:
        amp_dtype = torch.float32
        scaler = None

    if rank == 0:
        logger.info(
            f"AMP: dtype={amp_dtype}, scaler={'on' if scaler is not None else 'off'}, "
            f"bf16_hw_supported={bf16_hw}, cc={torch.cuda.get_device_capability()}")

    ungated_ep = cfg.get("polarity_ungated_epochs", 30)
    trans_ep = cfg.get("polarity_transition_epochs", 20)
    epochs = cfg.get("epochs", 200)
    # Checkpoint-selection criterion — drives both best.pt and early stopping.
    #   "total"    the multitask val loss (legacy behaviour)
    #   "pick_det" val picker + val detector only
    # On a trained RP90-Motion run the polarity term makes up more than half of
    # the total, so "total" picks checkpoints mostly on polarity rather than on
    # picking and detection quality. Use "pick_det" when
    # picking is the deliverable; "best_total.pt" is still written either way.
    best_metric = cfg.get("best_metric", "total")
    if best_metric not in ("total", "pick_det"):
        raise ValueError(
            f"best_metric must be 'total' or 'pick_det'; got {best_metric!r}")
    if rank == 0:
        logger.info(f"best_metric={best_metric} (drives best.pt + early stopping)")
    best_val = float("inf")    # selection metric
    best_total = float("inf")  # legacy total-loss reference tracker
    patience_ctr = 0
    start_epoch = 1

    # ── Resume ──
    resume_path = cfg.get("resume")
    if resume_path:
        ck = torch.load(resume_path, map_location=device, weights_only=False)  # trusted local ckpt
        base = unwrap_model(model)
        base.load_state_dict(ck["model_state_dict"])
        if cfg.get("resume_optimizer", True) and "optimizer_state_dict" in ck:
            optimizer.load_state_dict(ck["optimizer_state_dict"])
        if cfg.get("resume_scheduler", True) and "scheduler_state_dict" in ck:
            scheduler.load_state_dict(ck["scheduler_state_dict"])
        if scaler is not None and "scaler_state_dict" in ck:
            try:
                scaler.load_state_dict(ck["scaler_state_dict"])
            except Exception:
                pass
        start_epoch = int(ck.get("epoch", 0)) + 1
        # Prefer the stored selection metric so a resume keeps comparing like
        # with like; fall back to val_loss for pre-best_metric checkpoints.
        best_val = float(ck.get("val_sel", ck.get("val_loss", best_val)))
        best_total = float(ck.get("val_loss", best_total))
        if rank == 0:
            logger.info(f"Resumed from {resume_path} at epoch {start_epoch} "
                        f"(prev val={best_val:.6f})")

    # ── Training loop ──
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        if trans_ep > 0:
            uw = 1.0 if epoch <= ungated_ep else max(0.0, 1.0 - (epoch - ungated_ep) / trans_ep)
        else:
            uw = 1.0 if epoch <= ungated_ep else 0.0

        # Epoch data prep
        train_ds.set_epoch(epoch)
        # Val set is deterministic across epochs: re-seeding val_ds.rng with
        # epoch number would draw a different random sample each epoch, making
        # best-ckpt selection compare noisy val draws. Fix: only build val once
        # (already done in __init__ by _prepare_epoch(0)) and never reshuffle.
        # Safe to skip set_epoch for val.
        if is_distributed:
            train_sampler.set_epoch(epoch)

        tl, t_pick, t_pol, t_det, t_imp, nan_skips = train_one_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device, cfg, uw,
            amp_dtype=amp_dtype, epoch=epoch,
        )
        vl, v_pick, v_pol, v_det, v_imp, val_skips = validate(
            model, val_loader, loss_fn, device, cfg, uw, amp_dtype=amp_dtype,
        )

        # If validation went NaN (BN stats may still be poisoned despite guards),
        # reset BN running stats so next epoch repopulates from clean batches.
        if not np.isfinite(vl) or val_skips > 0:
            if rank == 0:
                logger.warning(
                    f"Val had {val_skips} non-finite batches (mean={vl}); "
                    "resetting BatchNorm running stats.")
            _reset_bn_stats(unwrap_model(model))

        # Update DWA with per-task means (only update if all finite).
        # In DDP each rank computes losses on its own data shard; we all-reduce
        # before updating DWA so all ranks share identical DWA state. Without
        # this, ranks drift to different weights and apply inconsistent
        # multitask balancing to their respective gradients.
        use_imp = cfg.get("use_impulsive_head", False) and cfg.get("use_polarity", True)
        if cfg.get("dwa", True):  # default must match loss constructor (line ~601)
            losses_to_avg = ([t_pick, t_pol, t_imp, t_det] if use_imp
                             else [t_pick, t_pol, t_det])
            if all(np.isfinite(x) for x in losses_to_avg):
                loss_tensor = torch.tensor(losses_to_avg, dtype=torch.float32,
                                           device=device)
                if is_distributed:
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                loss_fn.update_weights(loss_tensor.cpu())

        if epoch > warmup:
            scheduler.step()

        # ── Logging + checkpointing (rank 0 only) ──
        if rank == 0:
            epoch_time = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            seq_len = cfg["input_size"][0]
            train_per_ts = tl / seq_len
            val_per_ts = vl / seq_len
            dwa_str = f"DWA: {loss_fn.get_weights().tolist()}" if cfg.get("dwa") else ""

            if use_imp:
                parts_train = f"train(p/pol/imp/d)={t_pick:.3f}/{t_pol:.3f}/{t_imp:.3f}/{t_det:.3f}"
                parts_val = f"val(p/pol/imp/d)={v_pick:.3f}/{v_pol:.3f}/{v_imp:.3f}/{v_det:.3f}"
            else:
                parts_train = f"train(p/pol/d)={t_pick:.3f}/{t_pol:.3f}/{t_det:.3f}"
                parts_val = f"val(p/pol/d)={v_pick:.3f}/{v_pol:.3f}/{v_det:.3f}"
            logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {tl:.4f} (per_ts: {train_per_ts:.6f}) | "
                f"Val Loss: {vl:.4f} (per_ts: {val_per_ts:.6f}) | "
                f"LR: {lr:.6f} | "
                f"nan_skips={nan_skips}/{val_skips} | "
                f"{parts_train} | {parts_val} | "
                f"{dwa_str} | Time: {epoch_time:.1f}s"
            )

            val_pick_det = v_pick + v_det
            sel = vl if best_metric == "total" else val_pick_det

            def _make_ckpt():
                c = {
                    "epoch": epoch,
                    "model_state_dict": unwrap_model(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": vl, "val_pick_det": val_pick_det,
                    "val_sel": sel, "best_metric": best_metric,
                    "config": cfg,
                }
                if scaler is not None:
                    c["scaler_state_dict"] = scaler.state_dict()
                return c

            is_best = sel < best_val - cfg.get("min_delta", 1e-4)
            if epoch <= warmup:
                logger.info(
                    "Warmup epoch %d/%d — skipping best-model tracking "
                    "(val_loss=%.6f, val_pick_det=%.6f)",
                    epoch, warmup, vl, val_pick_det)
            elif is_best:
                prev_best = best_val
                best_val = sel; patience_ctr = 0
                d = out_dir / f"epoch_{epoch:04d}"; d.mkdir(exist_ok=True)
                torch.save(_make_ckpt(), d / "best.pt")
                logger.info(
                    "%s improved from %.6f to %.6f, saved %s",
                    best_metric, prev_best, sel, str(d / "best.pt"))
            else:
                patience_ctr += 1
                logger.info(
                    "%s %.4f did not improve from %.4f for %d epochs.",
                    best_metric, sel, best_val, patience_ctr)

            # Always keep a reference checkpoint under the legacy total-loss
            # criterion, so the two selections can be compared post-hoc without
            # re-running. Overwritten in place — no per-epoch directory.
            if epoch > warmup and vl < best_total - cfg.get("min_delta", 1e-4):
                best_total = vl
                torch.save(_make_ckpt(), out_dir / "best_total.pt")

            # Periodic checkpoint — carries the same per-task val breakdown, so
            # a post-hoc benchmark sweep can re-select on any criterion.
            if epoch % cfg.get("save_every", 10) == 0:
                torch.save(_make_ckpt(), out_dir / f"epoch_{epoch:04d}.pt")

        # ── Early-stop decision (all ranks must agree or NCCL hangs) ──
        stop_flag = torch.tensor(
            [1 if (rank == 0 and epoch > warmup
                   and patience_ctr >= cfg.get("patience", 50)) else 0],
            device=device, dtype=torch.int32)
        if is_distributed:
            dist.broadcast(stop_flag, src=0)
        if stop_flag.item() == 1:
            if rank == 0:
                logger.info(f"Early stopping after {epoch} epochs")
            break

    # ── Save final ──
    if rank == 0:
        torch.save(unwrap_model(model).state_dict(), out_dir / "final.pt")
        logger.info(f"Done → {out_dir / 'final.pt'}")

    if is_distributed:
        dist.barrier()  # all ranks wait for rank-0 save before tearing down NCCL
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
