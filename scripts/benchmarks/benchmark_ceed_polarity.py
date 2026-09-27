"""CEED first-motion polarity benchmark for redpan_motion.

Answers the raw-vs-filtered question: does band-passing the picker/detector input degrade
first-motion polarity? (The polarity head always reads RAW Z via ``z_raw``.)

Per event: crop 9000 samples centred on the catalogue P, run ONE forward pass
with ``z_raw = raw Z`` (sign-preserved), decode polarity = argmax[N,U,D] at the P
sample, and compare to CEED metadata `trace_p_polarity` (U/D/N). Runs two
conditions — no-bandpass (raw x) and bandpass 3-45 Hz on x — both with raw-Z
polarity, so any difference isolates the bandpass effect on polarity.

    PYTHONPATH=. python scripts/benchmarks/benchmark_ceed_polarity.py --max-eq 3000
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import torch

from redpan_motion import REDPANPredictor, bandpass_for_model, CH_Z

CACHE = os.environ.get("CEED_CACHE", "/path/to/data/seisbench_cache/datasets/ceed")
# CEED_TEST_IDS: a .npy of the CEED test-split trace ids to evaluate on.
IDS = os.environ.get("CEED_TEST_IDS", "/path/to/ceed_test_ids.npy")
CKPT = str(Path(__file__).resolve().parents[2] / "checkpoints/redpan_motion/best.pt")
IN_SAMPLES = 9000
DT = 0.01
POL = {0: "N", 1: "U", 2: "D"}   # polarity head channel order [N, U, D]


def build_polarity_index(cache):
    """{trace_name: 'U'|'D'|'N'} from CEED metadata*.csv `trace_p_polarity`."""
    idx = {}
    for csv in sorted(glob.glob(f"{cache}/metadata*.csv")):
        df = pd.read_csv(csv, usecols=["trace_name", "trace_p_polarity"], low_memory=False)
        df = df.dropna(subset=["trace_p_polarity"])
        df = df[df["trace_p_polarity"].isin(["U", "D", "N"])]
        idx.update(zip(df["trace_name"].values, df["trace_p_polarity"].values))
    return idx


def crop_to_window(wf, p_sample, target_len):
    """(3, N) -> (3, target_len) with P centred; zero-pad. Returns (win, new_p)."""
    n = wf.shape[1]
    start = p_sample - target_len // 2
    out = np.zeros((3, target_len), np.float32)
    lo, hi = max(0, start), min(n, start + target_len)
    out[:, lo - start:lo - start + (hi - lo)] = wf[:, lo:hi]
    return out, p_sample - start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-eq", type=int, default=3000, help="0 = all labelled events")
    ap.add_argument("--seed", type=int, default=42,
                    help="shuffle seed — the test IDs are region-sorted, so an unshuffled "
                         "head() slice is a biased (often hard) sample")
    ap.add_argument("--ckpt", default=CKPT)
    args = ap.parse_args()

    ids = np.load(IDS, allow_pickle=False)
    ids = ids[np.random.default_rng(args.seed).permutation(len(ids))]   # representative sample
    pol_idx = build_polarity_index(CACHE)
    keep = np.array([pol_idx.get(str(r["trace_name"])) in ("U", "D", "N") for r in ids])
    ids = ids[keep]
    if args.max_eq:
        ids = ids[:args.max_eq]
    print(f"CEED polarity benchmark — {len(ids)} labelled events  (model={Path(args.ckpt).parent.name})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pred = REDPANPredictor.from_checkpoint(args.ckpt, device=device)

    h5cache = {}

    def get_h5(region, year):
        key = (region, year)
        if key not in h5cache:
            p = Path(CACHE) / f"waveforms{region}{year}.hdf5"
            h5cache[key] = h5py.File(p, "r") if p.exists() else None
        return h5cache[key]

    rows = []
    for r in ids:
        tn = str(r["trace_name"])
        h5 = get_h5(str(r["region"]), int(r["year"]))
        if h5 is None or tn not in h5:
            continue
        wf = h5[tn][:]
        if wf.shape[0] != 3:
            continue
        win, p = crop_to_window(wf.astype(np.float32), int(r["p_sample"]), IN_SAMPLES)
        z = win[CH_Z]                                # raw vertical -> polarity (sign preserved)
        # polarity always reads raw Z; only the picker/detector input x changes
        _, _, pol_raw = pred.predict_arrays(win, mode="single", z_raw=z)
        _, _, pol_bp = pred.predict_arrays(bandpass_for_model(win, DT), mode="single", z_raw=z)
        rows.append((pol_idx[tn],
                     POL[int(pol_raw[p].argmax())],
                     POL[int(pol_bp[p].argmax())]))

    df = pd.DataFrame(rows, columns=["label", "raw", "bp"])
    print(f"  evaluated {len(df)}   label dist {df['label'].value_counts().to_dict()}\n")
    print(f"  {'condition':12s} | U/D recall | sign-acc (confident U/D) | 3-way acc")
    print(f"  {'-'*12}-+------------+--------------------------+----------")
    ud = df[df["label"].isin(["U", "D"])]
    for cond, name in (("raw", "no-bandpass"), ("bp", "bandpass")):
        ud_recall = (ud[cond] == ud["label"]).mean()            # N-predictions count as misses
        conf = ud[ud[cond].isin(["U", "D"])]                    # model committed to U/D
        sign = (conf[cond] == conf["label"]).mean() if len(conf) else float("nan")
        acc3 = (df[cond] == df["label"]).mean()
        print(f"  {name:12s} |   {ud_recall:.3f}    |   {sign:.3f}  ({len(conf)}/{len(ud)})       |  {acc3:.3f}")
    print(f"\n  raw-vs-bandpass polarity agreement: {(df['raw'] == df['bp']).mean():.3f}")


if __name__ == "__main__":
    main()
