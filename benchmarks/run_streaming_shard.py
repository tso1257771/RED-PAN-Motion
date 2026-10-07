#!/usr/bin/env python
"""Run one shard of the streaming replay (``run_streaming.py``) so a long replay can use several
processes: shard ``k`` of ``n`` handles events / noise records ``k, k+n, k+2n, ...`` of the same list
and writes the same per-event files into the same output directory (default as ``run_streaming.py``,
``noise_mask`` for ``--noise-mask mask`` with a 60 s model). Run ``n`` shards with
``--shard 0..n-1``; existing output files are skipped, so a shard can be restarted.

Example (the 60 s RED-PAN noise replay with the real mask, four processes):
    for k in 0 1 2 3; do python run_streaming_shard.py --model redpan --mode noise --noise-mask mask \\
        --shard $k --n-shards 4 & done; wait
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

from rpm_bench import config, models
from rpm_bench.streaming_utils import list_eq_events, list_noise_subdirs
from run_streaming import default_out_subdir, run_event_60s, run_event_90s


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    config.add_config_arg(p)
    p.add_argument("--model", required=True, choices=models.MODEL_KEYS)
    p.add_argument("--mode", required=True, choices=["eq", "noise"])
    p.add_argument("--shard", type=int, required=True, help="0-based shard index")
    p.add_argument("--n-shards", type=int, required=True)
    p.add_argument(
        "--noise-mask",
        default="legacy",
        choices=["legacy", "mask"],
        help="as in run_streaming.py (60 s models, noise mode)",
    )
    p.add_argument("--batch-size", type=int, default=64, help="forward batch size, 90 s models")
    p.add_argument("--out-dir", type=Path, default=None, help="default as run_streaming.py")
    p.add_argument("--device", default=models.default_device())
    args = p.parse_args()
    if not 0 <= args.shard < args.n_shards:
        p.error("--shard must be in [0, --n-shards)")
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s shard{args.shard} %(message)s")

    cfg = config.load(args.config)
    ev_root = cfg["eew_root"] / ("EQ_Mw4_to_6.9" if args.mode == "eq" else "noise")
    events = list_eq_events(str(ev_root)) if args.mode == "eq" else list_noise_subdirs(str(ev_root))
    if len(events) == 0:
        raise SystemExit(f"no {args.mode} events under {ev_root}; check config key eew_root")
    events = events[args.shard :: args.n_shards]
    is60 = args.model in models.SIXTY_S
    out_dir = args.out_dir or config.results_dir(
        cfg, "streaming", args.model, default_out_subdir(args.mode, args.noise_mask, is60)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    model = models.load_model(args.model, cfg, args.device)
    logging.info(
        "model=%s mode=%s shard %d/%d: %d events -> %s",
        args.model,
        args.mode,
        args.shard,
        args.n_shards,
        len(events),
        out_dir,
    )

    n_fail = 0
    for ei, evdir in enumerate(events):
        evid = os.path.basename(evdir)
        if args.mode == "noise":
            evid = f"{os.path.basename(os.path.dirname(evdir)).split('_')[-1]}_{evid}"
        out_csv = out_dir / f"{evid}.csv"
        if out_csv.exists():
            continue
        t0 = time.time()
        try:
            if model.is_redpan60:
                df = run_event_60s(model, args.mode, evdir, args.device, noise_mask=args.noise_mask)
            else:
                df = run_event_90s(model, args.mode, evdir, args.device, args.batch_size)
        except Exception:  # one unreadable event must not stop the shard
            n_fail += 1
            logging.exception("%s: failed", evid)
            continue
        if df is None:
            logging.warning("%s: no usable stations, skipping", evid)
            continue
        tmp = out_csv.with_name(out_csv.name + ".tmp")
        df.to_csv(tmp, index=False, sep="\t")
        os.replace(tmp, out_csv)
        if ei % 50 == 0:
            logging.info(
                "[%d/%d] %s: %d rows (%.1fs)", ei + 1, len(events), evid, len(df), time.time() - t0
            )
    if n_fail:
        logging.warning("%d events failed", n_fail)


if __name__ == "__main__":
    main()
