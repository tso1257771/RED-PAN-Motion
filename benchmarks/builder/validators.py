"""Post-build sanity checks for unified H5 builder output."""
from __future__ import annotations
from pathlib import Path
import re

import h5py
import numpy as np
import pandas as pd

from .schema import T_SAMPLES, N_CHANNELS


_PICKS_RE = re.compile(r"^\[(\s*\d+\s*(,\s*\d+\s*)*)?\]$")


def _parse_picks(s: str) -> list[int]:
    s = s.strip()
    if not _PICKS_RE.match(s):
        raise ValueError(f"malformed picks string: {s!r}")
    inner = s.strip("[]").strip()
    return [int(x) for x in inner.split(",") if x.strip()] if inner else []


def assert_h5_consistency(
    h5_path: Path,
    csv_path: Path,
    *,
    sample_check: int = 1000,
    rng_seed: int = 0,
) -> None:
    """Verify that an H5 + CSV pair conforms to the builder contract.

    Checks:
        1. CSV row count == sum of H5 group dataset sizes.
        2. Every (category, split) combo appears in both files.
        3. p/s arrival samples in CSV are within [0, T_SAMPLES) and parsable.
        4. Random `sample_check` waveforms from the H5 are finite and shape-correct.
        5. polarity column is in {'', 'U', 'D'} for polarity-bearing categories.
        6. content_hash is unique within (category, split).

    Raises:
        AssertionError on any violation.
    """
    h5_path = Path(h5_path); csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    with h5py.File(h5_path, "r") as h:
        # Walk all (category, split) pairs in H5 and count rows
        h5_counts: dict[tuple[str, str], int] = {}
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and name.endswith("/waveforms"):
                # name = '{category}/{split}/waveforms'
                parts = name.split("/")
                if len(parts) >= 3 and parts[-1] == "waveforms":
                    cat, split = parts[-3], parts[-2]
                    h5_counts[(cat, split)] = obj.shape[0]
                    assert obj.shape[1] == N_CHANNELS, \
                        f"{name}: expected channels={N_CHANNELS}, got {obj.shape[1]}"
                    assert obj.shape[2] == T_SAMPLES, \
                        f"{name}: expected T={T_SAMPLES}, got {obj.shape[2]}"
        h.visititems(visit)

        # 1: row counts match
        h5_total = sum(h5_counts.values())
        assert h5_total == len(df), \
            f"H5 row total ({h5_total}) != CSV rows ({len(df)})"

        # 2: every (cat, split) combo appears in both
        csv_combos = set(zip(df["category"], df["split"]))
        h5_combos = set(h5_counts.keys())
        missing_in_csv = h5_combos - csv_combos
        missing_in_h5 = csv_combos - h5_combos
        assert not missing_in_csv, f"CSV missing combos: {missing_in_csv}"
        assert not missing_in_h5, f"H5 missing combos: {missing_in_h5}"

        # 3: pick parseability + bounds
        for col in ("p_arrival_sample", "s_arrival_sample"):
            for s in df[col].fillna("[]"):
                picks = _parse_picks(str(s))
                for p in picks:
                    assert 0 <= p < T_SAMPLES, f"{col}={p} out of range in '{s}'"

        # 4: random waveform integrity
        rng = np.random.default_rng(rng_seed)
        n_sample = min(sample_check, len(df))
        if n_sample > 0:
            ix = rng.choice(len(df), size=n_sample, replace=False)
            for i in ix:
                row = df.iloc[i]
                cat, split, hidx = row["category"], row["split"], int(row["hdf5_index"])
                wf = h[f"{cat}/{split}/waveforms"][hidx]
                assert wf.shape == (N_CHANNELS, T_SAMPLES), \
                    f"row {i} hdf5_index={hidx}: shape {wf.shape}"
                assert np.isfinite(wf).all(), \
                    f"row {i} hdf5_index={hidx}: non-finite values"

        # 5: polarity dictionary (as schema.Polarity; "N" = no clear first motion, CEED)
        valid_pol = {"", "U", "D", "N"}
        bad = ~df["polarity"].fillna("").isin(valid_pol)
        assert not bad.any(), f"{bad.sum()} rows with bad polarity values"

        # 6: hash uniqueness within (cat, split)
        if "content_hash" in df.columns:
            for (cat, split), grp in df.groupby(["category", "split"]):
                dups = grp["content_hash"].duplicated().sum()
                assert dups == 0, f"{dups} duplicate content_hash in ({cat}, {split})"

    print(f"OK: {h5_path.name} consistent with {csv_path.name} "
          f"({h5_total} rows, {len(h5_counts)} (cat,split) combos)")
