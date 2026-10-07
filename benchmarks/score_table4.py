#!/usr/bin/env python
"""Table IV (peak vs mean detector gate) and the S-P-adaptive threshold fit.

For each catalog S-P bin ([0,5), [5,10), [10,15), [15,20), [20,+) s) and each gate (mask PEAK or
MEAN over the trigger), sweep mask 0.2-0.9, P 0.1-0.7, S 0.1-0.6 and keep the triple with the best
macro joint F1 over the eight held-out earthquake sets. Earthquake TP: any trigger with mask >= m,
P >= p within 0.5 s, S >= s within 1.0 s. Joint FP: pooled-noise records with ANY trigger passing
mask >= m, inside-trigger max P >= p and max S >= s (the deployment rule). The mean-gate triples are
the ``redpan_motion.sp_thresholds`` tables (bins in the order above).

Held-out earthquake sets: static/<model>/{ceed_nc,ceed_sc,crew,geonet,instance,romplus,stead_h5,tw}.csv
(GeoNet = the 2013-2014 holdout, STEAD = the HDF5 test split). Noise: noise/<model>/noisefp_triggers_<POOL>.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rpm_bench import config
from rpm_bench.constants import P_TOL_SEC as P_TOL, S_TOL_SEC as S_TOL
from rpm_bench.results import HELDOUT_SETS, POOLS, noise_csv, static_csv, write_json

BINS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 1e9)]
MASKG = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
PG = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
SG = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def load_eq(cfg, model) -> dict:
    """Per held-out dataset: the earthquake trigger rows as arrays for the sweep."""
    cells = {}
    for ds in HELDOUT_SETS:
        d = pd.read_csv(static_csv(cfg, model, ds), low_memory=False)
        d = d[(d.label_type == "earthquake") & d["labelP_sec"].notna() & (d["ps_diff_sec"] > 0)]
        one = d["evid"].nunique() == len(d)
        cells[ds] = dict(
            one=one,
            codes=None if one else pd.factorize(d["evid"])[0],
            sp=d["ps_diff_sec"].values,
            mk_peak=d["mask_peak"].fillna(0).values,
            mk_mean=d["mask_mean"].fillna(0).values,
            nt=(d["n_triggers"].fillna(0).values >= 1),
            Pp=d["P_pick_prob"].fillna(0).values,
            prok=(d["P_residual_sec"].abs().fillna(np.inf).values <= P_TOL),
            Sp=d["S_pick_prob"].fillna(0).values,
            srok=(d["S_residual_sec"].abs().fillna(np.inf).values <= S_TOL),
        )
    return cells


def fp_grid(files, mask_col) -> np.ndarray:
    """Pooled-noise records with any trigger passing (mask, P, S), for every grid point."""
    frames = []
    for f in files:
        d = pd.read_csv(
            f, usecols=["hdf5_index", "trigger_idx", mask_col, "inside_max_P", "inside_max_S"]
        )
        d = d[d["trigger_idx"] >= 0]
        d["gid"] = f.name + ":" + d["hdf5_index"].astype(str)
        frames.append(d)
    a = pd.concat(frames, ignore_index=True)
    code = pd.factorize(a["gid"])[0]
    mk = a[mask_col].fillna(0).values
    iP = a["inside_max_P"].fillna(0).values
    iS = a["inside_max_S"].fillna(0).values
    FP = np.zeros((len(MASKG), len(PG), len(SG)), int)
    for mi, m in enumerate(MASKG):
        for pi, p_ in enumerate(PG):
            mp = (mk >= m) & (iP >= p_)
            for si, s_ in enumerate(SG):
                sel = mp & (iS >= s_)
                if sel.any():
                    FP[mi, pi, si] = np.unique(code[sel]).size
    return FP


def optimum(cells, FP, lo, hi, mask_key):
    """(macro F1, mask, P, S, macro recall, FP) of the best grid point for S-P in [lo, hi)."""
    sel = {ds: ((c["sp"] >= lo) & (c["sp"] < hi)) for ds, c in cells.items()}
    nby = {
        ds: (int(sel[ds].sum()) if c["one"] else int(np.unique(c["codes"][sel[ds]]).size))
        for ds, c in cells.items()
    }
    best = None
    for mi, m in enumerate(MASKG):
        for pi, p in enumerate(PG):
            for si, s in enumerate(SG):
                fp = FP[mi, pi, si]
                f1s, rs = [], []
                for ds, c in cells.items():
                    n = nby[ds]
                    if n == 0:
                        continue
                    ok = (
                        (c[mask_key] >= m)
                        & c["nt"]
                        & (c["Pp"] >= p)
                        & c["prok"]
                        & (c["Sp"] >= s)
                        & c["srok"]
                        & sel[ds]
                    )
                    tp = int(ok.sum()) if c["one"] else int(np.unique(c["codes"][ok]).size)
                    r = tp / n
                    prec = tp / (tp + fp) if (tp + fp) else 0.0
                    rs.append(r)
                    f1s.append(2 * prec * r / (prec + r) if (prec + r) else 0.0)
                f1 = float(np.mean(f1s))
                if best is None or f1 > best[0]:
                    best = (f1, round(m, 2), round(p, 2), round(s, 2), float(np.mean(rs)), int(fp))
    return best


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    config.add_config_arg(p)
    p.add_argument(
        "--models",
        nargs="+",
        default=["edge", "rpm", "redpan"],
        help="add redpan_240107 for the shipped redpan_60s table",
    )
    p.add_argument("--out-json", type=Path, default=None, help="also write the results as JSON")
    args = p.parse_args()
    cfg = config.load(args.config)
    res = {}
    for m in args.models:
        cells = load_eq(cfg, m)
        files = [noise_csv(cfg, m, pool) for pool in POOLS]
        FPp, FPm = fp_grid(files, "mask_peak"), fp_grid(files, "mask_mean")
        r = {"global": {}, "bins": {}}
        print(f"\n=== {m} ===")
        for gate, mk, FP in (("peak", "mk_peak", FPp), ("mean", "mk_mean", FPm)):
            g = optimum(cells, FP, 0, 1e9, mk)
            r["global"][gate] = g
            print(
                f"  GLOBAL [{gate}]: mask {g[1]:.2f} / P {g[2]} / S {g[3]}  F1={g[0]:.4f} recall={g[4]:.4f} FP={g[5]}"
            )
        for lo, hi in BINS:
            lab = f"[{lo:g},{hi:g})" if hi < 1e9 else f"[{lo:g},+)"
            bp = optimum(cells, FPp, lo, hi, "mk_peak")
            bm = optimum(cells, FPm, lo, hi, "mk_mean")
            r["bins"][lab] = {"peak": bp, "mean": bm}
            print(
                f"  {lab:>9} | PEAK {bp[1]:.2f}/{bp[2]}/{bp[3]} F1={bp[0]:.3f} FP={bp[5]} | "
                f"MEAN {bm[1]:.2f}/{bm[2]}/{bm[3]} F1={bm[0]:.3f} FP={bm[5]} | dF1={bm[0] - bp[0]:+.3f} "
                + (f"dFP={100 * (bm[5] - bp[5]) / bp[5]:+.0f}%" if bp[5] else "dFP=n/a")
            )
        r["sp_table_mean_gate"] = [list(r["bins"][b]["mean"][1:4]) for b in r["bins"]]
        print("  sp_thresholds bins (mean gate):", [tuple(t) for t in r["sp_table_mean_gate"]])
        res[m] = r
    if args.out_json:
        write_json(args.out_json, res)


if __name__ == "__main__":
    main()
