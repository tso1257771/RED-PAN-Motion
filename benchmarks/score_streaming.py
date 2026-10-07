#!/usr/bin/env python
"""Score the streaming replay (run_streaming.py outputs).

For each model, sweep P probability (0.1-0.8), mask probability (0.0-0.8) and minimum delay
(0.1-1.0 s): an earthquake station record is detected at its earliest window with a pick
satisfying the thresholds and 0 <= delay <= 3 s; a noise false positive is a distinct noise
trigger (network, station, record, trigger time). Reports

  best-F1    the best F1 over the whole grid, with its operating point and mean delay
  fire-ASAP  the best F1 with the minimum delay fixed at 0.1 s (the delay reported in the text)

on the full noise set (4,370 records) and on the 800-record subset of
``data/eew_matched800_noise.txt`` (the 800 noise records the original 60 s RED-PAN replay was
run on; scores on this subset are comparable with that replay).

Reads ``<results_root>/streaming/<model>/{eq,noise}/*.csv`` (``--noise-subdir noise_mask`` for the
60 s models' ``run_streaming.py --noise-mask mask`` variant).

Example:
    python score_streaming.py --models edge rpm redpan --out-json results/streaming_scores.json
"""

from __future__ import annotations

import argparse
import os
from glob import glob
from pathlib import Path

import pandas as pd
from rpm_bench import config
from rpm_bench.results import write_json

P_PROB = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
M_PROB = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
DELAY_MIN = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
DMAX = 3.0
NAMES = {
    "edge": "Edge-RED-PAN-Motion",
    "rpm": "RED-PAN-Motion",
    "redpan": "RED-PAN (2022 paper)",
    "redpan_240107": "RED-PAN (240107)",
}
MATCHED_FILE = Path(__file__).resolve().parent / "data" / "eew_matched800_noise.txt"


def gather(d, subset=None) -> pd.DataFrame:
    """All replay CSVs of a directory (only the file names in ``subset`` if given), with
    ``network``, ``evid`` and the station-record key ``ID`` added."""
    dfs = []
    for f in sorted(glob(os.path.join(d, "*.csv"))):
        bn = os.path.basename(f)
        if subset is not None and bn not in subset:
            continue
        net = bn.split("_")[0]
        ev = bn.split("_")[1].replace(".csv", "") if "_" in bn else bn.replace(".csv", "")
        x = pd.read_csv(f, sep="\t")
        x["network"] = net
        x["evid"] = ev
        x["ID"] = net + "_" + x["station_id"].astype(str) + "_" + ev
        dfs.append(x)
    if not dfs:
        raise FileNotFoundError(f"no replay CSVs in {d}")
    return pd.concat(dfs, ignore_index=True)


def epoch_seconds(times: pd.Series) -> pd.Series:
    """Seconds since 1970 of parsed datetimes. Cast to nanoseconds first: pandas 2 parses strings
    to datetime64[ns], pandas 3 to datetime64[us], and ``astype("int64")`` returns the raw count
    in that unit (identical to the manuscript code on pandas 2.2.2)."""
    return times.astype("datetime64[ns]").astype("int64") / 1e9


def score(eqdf, nzdf):
    """(earthquake station records, best-F1 record, fire-ASAP record) over the threshold grid;
    a record is (F1, recall, precision, mean delay, TP, FP, P threshold, mask threshold, min delay)."""
    total = eqdf.ID.nunique()
    eqf = eqdf[(eqdf.delay >= 0) & (eqdf.delay <= DMAX)]
    nzf = nzdf[(nzdf.delay >= 0) & (nzdf.delay <= DMAX)].copy()
    nzf["inf_t"] = epoch_seconds(pd.to_datetime(nzf["inference_endtime"]))
    best = best01 = None
    for pp in P_PROB:
        eqp = eqf[eqf.P_probability >= pp]
        nzp = nzf[nzf.P_probability >= pp]
        for mp in M_PROB:
            eqm = eqp[eqp.Mask_probability >= mp]
            nzm = nzp[nzp.Mask_probability >= mp]
            for dmin in DELAY_MIN:
                fe = eqm[eqm.delay >= dmin]
                trg = fe.loc[fe.groupby("ID")["delay"].idxmin()] if len(fe) else fe
                tp = len(trg)
                fnz = nzm[nzm.delay >= dmin]
                fp = (
                    (
                        fnz.assign(tt=(fnz.inf_t - fnz.delay).round(2))
                        .groupby(["network", "station_id", "evid", "tt"])
                        .ngroups
                    )
                    if len(fnz)
                    else 0
                )
                r = tp / total
                p = tp / (tp + fp) if tp + fp else 0.0
                f1 = 2 * p * r / (p + r) if p + r else 0.0
                rec = (f1, r, p, float(trg.delay.mean()) if tp else -1, tp, fp, pp, mp, dmin)
                if best is None or f1 > best[0]:
                    best = rec
                if abs(dmin - 0.1) < 1e-9 and (best01 is None or f1 > best01[0]):
                    best01 = rec
    return total, best, best01


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    config.add_config_arg(ap)
    ap.add_argument("--models", nargs="+", default=["edge", "rpm", "redpan"], choices=list(NAMES))
    ap.add_argument(
        "--noise-subdir",
        default="noise",
        help="sub-directory of streaming/<model>/ with the noise replay (default noise; "
        "noise_mask for run_streaming.py --noise-mask mask on the 60 s models)",
    )
    ap.add_argument("--out-json", type=Path, default=None, help="also write the scores as JSON")
    args = ap.parse_args()
    cfg = config.load(args.config)
    root = cfg["results_root"] / "streaming"
    with open(MATCHED_FILE) as fh:
        matched = {ln.strip() for ln in fh if ln.strip()}

    res = {}
    for noise_set, subset in (("full noise", None), ("matched 800", matched)):
        print(f"\n### {noise_set}")
        for key in args.models:
            eq = gather(str(root / key / "eq"))
            nz = gather(str(root / key / args.noise_subdir), subset)
            total, b, a = score(eq, nz)
            res.setdefault(noise_set, {})[key] = {
                "best": list(b),
                "asap": list(a),
                "n_eq": total,
                "n_noise": int(nz.ID.nunique()),
            }
            for lab, x in (("best-F1", b), ("fire-ASAP", a)):
                print(
                    f"{NAMES[key]:24s} {lab:9s} F1={x[0]:.4f} R={x[1]:.4f} P={x[2]:.4f} delay={x[3]:.3f}s "
                    f"FP={x[5]} @P{x[6]}/M{x[7]}/d{x[8]}  ({total} eq-sta, {nz.ID.nunique()} noise)"
                )
    if args.out_json:
        write_json(args.out_json, res)


if __name__ == "__main__":
    main()
