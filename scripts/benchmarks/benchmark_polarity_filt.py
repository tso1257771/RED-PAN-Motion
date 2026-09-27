"""First-motion polarity under hp1 / bp345 / raw on the 90 s H5 training datasets.

This reads the pre-windowed 9000-sample singleEQ traces +
metadata `polarity` (U/D[/N]), and for a shuffled sample decodes argmax[N,U,D] at the
catalogue P under each picker/detector filter (z_raw = raw Z is fixed, so any delta
isolates the filter's effect on the polarity head's *confidence*).

    PYTHONPATH=. python scripts/benchmarks/benchmark_polarity_filt.py --dataset tw --max-eq 5000
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import torch
from scipy.signal import butter, sosfiltfilt

from redpan_motion import REDPANPredictor, bandpass_for_model, CH_Z

DATA_ROOT = os.environ.get("DATA_ROOT", "/path/to/input_h5_90sec")
ROOTS = {
    "tw": f"{DATA_ROOT}/TW/TW_dataset_90s_singleEQ",
    "ceed_nc": f"{DATA_ROOT}/CEED_NC/CEED_NC_dataset_90s_singleEQ",
    "instance": f"{DATA_ROOT}/INSTANCE/INSTANCE_dataset_90s_singleEQ",
}
CKPT = str(Path(__file__).resolve().parents[2] / "checkpoints/redpan_motion/best.pt")
DT = 0.01
POL = {0: "N", 1: "U", 2: "D"}


def highpass(w, f=1.0):
    sos = butter(4, f / 50.0, btype="high", output="sos")
    return np.stack([sosfiltfilt(sos, w[c] - w[c].mean()) for c in range(3)]).astype("float32")


def parse_p(v):
    s = str(v).strip().strip("[]")
    return int(float(s.split(",")[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(ROOTS))
    ap.add_argument("--max-eq", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = ROOTS[args.dataset]
    m = pd.read_csv(root + "_metadata.csv", low_memory=False)
    m = m[(m["split"] == "test") & (m["polarity"].isin(["U", "D"]))]
    if args.max_eq and len(m) > args.max_eq:   # --max-eq 0 = whole test
        m = m.sample(args.max_eq, random_state=args.seed)
    print(f"polarity [{args.dataset}] — {len(m)} U/D-labelled test events (hp1 vs bp345 vs raw)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pred = REDPANPredictor.from_checkpoint(CKPT, device=device)
    filters = {"raw": lambda w: w, "hp1": lambda w: highpass(w, 1.0),
               "bp345": lambda w: bandpass_for_model(w, DT)}

    rows = []
    with h5py.File(root + ".h5", "r") as h5:
        wf_ds = h5["singleEQ/test/waveforms"]
        for _, r in m.iterrows():
            w = wf_ds[int(r["hdf5_index"])].astype("float32")  # (3, 9000)
            if w.shape[0] != 3:
                w = w.T
            p = parse_p(r["p_arrival_sample"])
            if not (0 <= p < w.shape[1]):
                continue
            row = {"gt": r["polarity"]}
            for k, fn in filters.items():
                _, _, pol = pred.predict_arrays(fn(w), mode="single", z_raw=w[CH_Z])
                row[k] = POL[int(pol[p].argmax())]
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"  evaluated {len(df)}   label dist {df['gt'].value_counts().to_dict()}")
    print(f"  {'filter':6s} | U/D recall | sign-acc (confident) | N-abstain")
    for k in filters:
        rec = (df[k] == df["gt"]).mean()
        conf = df[df[k].isin(["U", "D"])]
        sign = (conf[k] == conf["gt"]).mean() if len(conf) else float("nan")
        nab = (df[k] == "N").mean()
        print(f"  {k:6s} |   {rec:.3f}    |   {sign:.3f}  ({len(conf)}/{len(df)})   |  {nab:.3f}")


if __name__ == "__main__":
    main()
