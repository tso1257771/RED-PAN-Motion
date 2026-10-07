#!/usr/bin/env python
"""Build the official STEAD test noise as 90 s windows in the archive's HDF5 layout.

The STEAD noise in the 90 s archive comes from the archive builder (``P05_TFRecord_noise.py``):
noise traces outside STEAD ``test.npy``, split 80/20 into train/val, each padded in front with
3,000 samples of spectrum-matched noise. The 23,526 noise traces listed in ``test.npy`` were held
out and never built. This script builds them the same way, as the test split.

Input:  ``<stead_root>/{merge.csv, merge.hdf5, test.npy}`` (the original STEAD release).
Output (default ``stead_noise_test_dir``):
  STEAD_dataset_90s_noise_test.h5            noise/test/waveforms (23526, 3, 9000) float32, raw counts,
                                             channel order E, N, Z, chunks (100, 3, 9000)
  STEAD_dataset_90s_noise_test_metadata.csv
Samples 3000:9000 of each window are the original 60 s trace (the 60 s RED-PAN and the SeisBench
baselines read those). Padding is seeded (numpy seed 0): a rebuild matches the published file
(md5 0af03c69c2bde5159e980397dc01705d) to about 3e-5 counts; byte identity depends on the numpy and
scipy versions.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from rpm_bench import config

from redpan_motion.utils.waveform import find_reference_signal, generate_matching_noise

PAD, NPTS = 3000, 6000
T = PAD + NPTS


def pad_front(wf3):
    """Prepend PAD samples of spectrum-matched noise to each channel of a (3, NPTS) trace."""
    return np.stack(
        [
            np.concatenate(
                [
                    generate_matching_noise(
                        find_reference_signal(c, window_size=500, max_search=5000, min_unique=300),
                        PAD,
                    ),
                    c,
                ]
            )
            for c in wf3
        ]
    ).astype(np.float32)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    config.add_config_arg(p)
    p.add_argument(
        "--out-dir", type=Path, default=None, help="default: stead_noise_test_dir from the config"
    )
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = config.load(args.config)
    root, out = cfg["stead_root"], args.out_dir or cfg["stead_noise_test_dir"]
    np.random.seed(0)
    os.makedirs(out, exist_ok=True)
    df = pd.read_csv(root / "merge.csv", low_memory=False)
    # test.npy of the STEAD release is an object array of trace names, so it needs
    # allow_pickle; check its md5 (README, Data) before loading a copy from elsewhere.
    test = set(np.load(root / "test.npy", allow_pickle=True))
    names = df[(df.trace_category == "noise") & df.trace_name.isin(test)].trace_name.tolist()
    rows = []
    with (
        h5py.File(root / "merge.hdf5", "r") as src,
        h5py.File(out / "STEAD_dataset_90s_noise_test.h5", "w") as dst,
    ):
        ds = dst.create_dataset(
            "noise/test/waveforms",
            shape=(len(names), 3, T),
            maxshape=(None, 3, T),
            dtype="float32",
            chunks=(100, 3, T),
        )
        k = 0
        for n in names:
            wf = np.asarray(src["data/" + n][()], dtype=np.float32)
            if wf.shape != (NPTS, 3):
                logging.warning("skip %s: shape %s", n, wf.shape)
                continue
            ds[k] = pad_front(wf.T)
            rows.append(
                dict(
                    hdf5_index=k,
                    sample_id=f"noise_{n}",
                    category="noise",
                    sampling_category="noise",
                    split="test",
                    source_file=f"STEAD/merge.hdf5:data/{n}",
                    p_arrival_sample="[]",
                    s_arrival_sample="[]",
                    ps_diff_samples=0,
                    lightweight_storage=True,
                )
            )
            k += 1
            if k % 5000 == 0:
                logging.info("%d / %d", k, len(names))
        if k < len(names):
            ds.resize((k, 3, T))
    pd.DataFrame(rows).to_csv(out / "STEAD_dataset_90s_noise_test_metadata.csv", index=False)
    logging.info("wrote %d windows to %s", k, out)


if __name__ == "__main__":
    main()
