"""CLI: daily batch inference over a day archive, the PyTorch counterpart of the
original RED-PAN daily worker.

For one day it iterates every 3-component instrument, runs single-forward-pass
inference, applies mask-detection P-S pairing, and writes one redpan_picks CSV
per instrument. Thin wrapper over the redpan_motion package (no sys.path hacks);
for cell-by-cell interactive use see ``notebooks/daily_inference.py``.

Input layout (a CWA-style day archive; the archive itself is not public):
    <datadir>/<NETDIR>/A/<YYYY>/<JJJ>/<STA>/<STA>.<NET>.<LOC>.<CHN>.<YYYY>.<JJJ>
Output:
    <out>/<YYYY>/<JJJ>/picks_<NET>.<STA>.<LOC>.<CHN2>.csv

    PYTHONPATH=. python scripts/daily_inference.py --datadir /path/to/day_archive --year 2019 --jday 1 \
        [--xml-dir metadata/stationxml/stations] [--limit N] [--overwrite]
"""
import argparse
from pathlib import Path

import torch

from redpan_motion import (
    REDPANPredictor, load, highpass, picks_to_dataframe, group_instruments, CH_Z,
)

ROOT = str(Path(__file__).resolve().parent.parent)   # project root (this file is scripts/..)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datadir", required=True,
                    help="root of the day archive (see the layout above)")
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--jday", type=int, required=True)
    ap.add_argument("--out", default=f"{ROOT}/outputs/redpan_picks")
    ap.add_argument("--ckpt", default="redpan_motion",
                    help="a .pt path, or a shipped checkpoint: redpan_60s, redpan_motion, edge_rp90")
    ap.add_argument("--mode", default="single", help="single (default) or sliding")
    ap.add_argument("--xml-dir", default=None,
                    help="dir of <NET>.<STA>.xml for amplitudes (omit = no-amplitude fast path)")
    ap.add_argument("--limit", type=int, default=0, help="process at most N instruments (0=all)")
    ap.add_argument("--overwrite", action="store_true", help="re-process even if output exists")
    ap.add_argument("--two-pass-polarity", action="store_true",
                    help="second raw-input forward for polarity (~2x inference). "
                         "Default: single pass (band-pass picks + raw-Z polarity).")
    args = ap.parse_args()

    groups = group_instruments(args.datadir, args.year, args.jday)
    yr, jd = f"{args.year:04d}", f"{args.jday:03d}"
    if not groups:
        raise SystemExit(f"no 3-component instruments under {args.datadir} for {yr}/{jd}")
    out_dir = Path(args.out) / yr / jd
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(groups)} instruments for {yr}/{jd}  "
          f"amplitude={'on' if args.xml_dir else 'off'}  mode={args.mode}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pred = REDPANPredictor.from_checkpoint(args.ckpt, device=device)

    n_done = n_pick = n_skip = 0
    for i, ((net, sta, loc, chn2), gidx) in enumerate(groups.items(), 1):
        if args.limit and n_done >= args.limit:
            break
        out_file = out_dir / f"picks_{net}.{sta}.{loc}.{chn2}.csv"
        if out_file.exists() and not args.overwrite:
            continue
        xml = None
        if args.xml_dir:
            cand = Path(args.xml_dir) / f"{net}.{sta}.xml"
            xml = str(cand) if cand.exists() else None
        try:
            raw, wf_so, inv, ids, t0, st = load(gidx, xml)
            sid = ".".join(ids)
            dt = 1.0 / st[0].stats.sampling_rate
            if args.two_pass_polarity:
                # pass 1 highpass -> picks/detection; pass 2 raw -> polarity (dominated)
                picker, detector, _ = pred.predict_arrays(highpass(raw, 1.0, dt), mode=args.mode)
                _, _, polarity = pred.predict_arrays(raw.astype("float32"), mode=args.mode)
            else:
                # single forward: 1 Hz highpass picks/detection + raw-Z polarity together
                picker, detector, polarity = pred.predict_arrays(
                    highpass(raw, 1.0, dt), mode=args.mode, postprocess=False,
                    z_raw=raw[CH_Z])   # raw (unfiltered) vertical -> polarity first-motion
            df = picks_to_dataframe(picker, detector, polarity, t0, sid,
                                    raw_counts=raw, wf_sensonly=wf_so, inv=inv,
                                    dt=dt, amplitude=xml is not None)
        except Exception as e:               # one bad instrument must not stop the day
            n_skip += 1
            print(f"  [{i}/{len(groups)}] {net}.{sta}.{loc}.{chn2}: SKIP "
                  f"({type(e).__name__}: {e})")
            continue
        df.to_csv(out_file, index=False)
        n_done += 1
        n_pick += len(df) // 2
        print(f"  [{i}/{len(groups)}] {sid}: {len(df) // 2} pairs -> {out_file.name}")

    print(f"done: {n_done} instruments, {n_pick} P-S pairs, {n_skip} skipped -> {out_dir}")


if __name__ == "__main__":
    main()
