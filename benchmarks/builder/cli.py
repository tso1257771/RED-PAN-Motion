"""Build one 90 s archive (HDF5 + metadata CSV) from a dataset's original release.

Each (dataset, category) maps to an adapter that windows the source traces to 90 s (9,000 samples
at 100 Hz, E/N/Z), pads with spectrum-matched noise where the source is shorter, assigns the
train/val/test split and writes ``<DATASET>_dataset_90s_<CATEGORY>.h5`` + ``_metadata.csv``.
Source locations come from the harness config (``--config`` / ``RPM_BENCH_CONFIG`` /
``RPM_BENCH_<KEY>``, see ``builder/sources.py``); outputs go to ``<build_root>/<DATASET>/``.
Every location can also be given on the command line.

Examples (run from ``benchmarks/``):
    python -m builder --list                                  # plan + resolved sources
    python -m builder --dataset STEAD --category noise --validate
    python -m builder --dataset CREW --category singleEQ --max-samples 200   # smoke run
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from . import sources
from .adapters.ceed_nc import CEEDNCSingleEQAdapter
from .adapters.ceed_sc import CEEDSCSingleEQAdapter
from .adapters.crew import CREWPonlyAdapter, CREWSingleEQAdapter, CREWSonlyAdapter
from .adapters.geonet import GeoNetNoiseAdapter, GeoNetSingleEQAdapter
from .adapters.instance import INSTANCENoiseAdapter, INSTANCESingleEQAdapter
from .adapters.obstransformer import OBSTransformerSingleEQAdapter
from .adapters.rocknet import RockNetNoiseAdapter
from .adapters.romplus import ROMPLUSPonlyAdapter, ROMPLUSSingleEQAdapter, ROMPLUSSonlyAdapter
from .adapters.stead import STEADNoiseAdapter, STEADSingleEQAdapter
from .adapters.tw import TWNoiseAdapter, TWPonlyAdapter, TWSingleEQAdapter, TWSonlyAdapter
from .validators import assert_h5_consistency
from .writer import build_h5

# (dataset, category) -> adapter class
ADAPTERS = {
    ("STEAD", "noise"): STEADNoiseAdapter,
    ("STEAD", "singleEQ"): STEADSingleEQAdapter,
    ("INSTANCE", "noise"): INSTANCENoiseAdapter,
    ("INSTANCE", "singleEQ"): INSTANCESingleEQAdapter,
    ("GeoNet", "noise"): GeoNetNoiseAdapter,
    ("GeoNet", "singleEQ"): GeoNetSingleEQAdapter,
    ("CREW", "singleEQ"): CREWSingleEQAdapter,
    ("CREW", "Ponly"): CREWPonlyAdapter,
    ("CREW", "Sonly"): CREWSonlyAdapter,
    ("OBSTransformer", "singleEQ"): OBSTransformerSingleEQAdapter,
    ("RockNet", "noise"): RockNetNoiseAdapter,
    ("ROMPLUS", "singleEQ"): ROMPLUSSingleEQAdapter,
    ("ROMPLUS", "Ponly"): ROMPLUSPonlyAdapter,
    ("ROMPLUS", "Sonly"): ROMPLUSSonlyAdapter,
    ("CEED_NC", "singleEQ"): CEEDNCSingleEQAdapter,
    ("CEED_SC", "singleEQ"): CEEDSCSingleEQAdapter,
    ("TW", "noise"): TWNoiseAdapter,
    ("TW", "singleEQ"): TWSingleEQAdapter,
    ("TW", "Ponly"): TWPonlyAdapter,
    ("TW", "Sonly"): TWSonlyAdapter,
}
assert set(ADAPTERS) == set(sources.PLAN)


def _read_trace_list(path: Path):
    """Trace names from a ``.npy`` (e.g. STEAD ``test.npy``) or a text file, one per line."""
    if path.suffix == ".npy":
        return [str(x) for x in np.load(path, allow_pickle=True).tolist()]
    return [ln.strip() for ln in open(path) if ln.strip()]


def make_adapter(key, src: sources.Sources, a):
    """The adapter for ``key`` with the per-dataset keyword arguments of the original CLI."""
    dataset, category = key
    cls = ADAPTERS[key]
    opt = src.options
    split = dict(
        p_train=a.p_train if a.p_train is not None else opt.get("p_train", 0.70),
        p_val=a.p_val if a.p_val is not None else opt.get("p_val", 0.15),
        seed=a.seed,
    )
    common = dict(max_samples=a.max_samples or None)
    if dataset == "STEAD":
        kw = dict(merge_hdf5=src.source, merge_csv=src.metadata, **split, **common)
        excl = a.exclude_traces or opt.get("exclude_test_npy")
        if category == "noise" and excl is not None:
            kw["exclude_traces"] = _read_trace_list(Path(excl))
        return cls(**kw)
    if dataset == "INSTANCE":
        return cls(
            events_hdf5=src.source,
            events_csv=src.metadata,
            noise_hdf5=src.extra_source,
            noise_csv=src.extra_metadata,
            **split,
            **common,
        )
    pol = a.with_polarity if a.with_polarity is not None else bool(opt.get("with_polarity", False))
    if dataset == "CEED_NC":
        return cls(h5_path=src.source, csv_path=src.metadata, with_polarity=pol, **common)
    if dataset == "CEED_SC":
        return cls(root=src.source, with_polarity=pol, **common)
    if dataset == "ROMPLUS":
        return cls(
            sac_root=src.source,
            metadata_csv=src.metadata,
            stratify_ps_bins=False,
            **split,
            **common,
        )
    if dataset == "CREW":
        kw = dict(crew_root=src.source, **split, **common)
        if category == "singleEQ":  # Pn+Sn picks only; Ponly/Sonly take no phase_pair
            kw["phase_pair"] = a.crew_phase_pair or opt.get("crew_phase_pair", "composite")
        return cls(**kw)
    if dataset == "OBSTransformer":  # event-hash 70/15/15 split; p_train/p_val not used
        return cls(training_hdf5=src.source, seed=a.seed, **common)
    if dataset == "RockNet":  # published partition files; Bernoulli fallback for the rest
        return cls(luhu_h5=src.source, partition_root=src.metadata, seed=a.seed, **common)
    if dataset == "TW":
        if category == "noise":
            return cls(wf_dir=src.source, pred_dir=src.metadata, **split, **common)
        if category == "Sonly":
            return cls(wf_dir=src.source, meta_dir=src.metadata, **split, **common)
        return cls(
            wf_dir=src.source,
            meta_dir=src.metadata,
            polarity_csv=src.extra_metadata,
            stratify_ps_bins=False,
            **split,
            **common,
        )
    if dataset == "GeoNet":  # year-based splits; 2013-2014 reserved for the holdout
        if category == "noise":
            return cls(noise_hdf5=src.source, noise_csv=src.metadata, **split, **common)
        return cls(events_hdf5_dir=src.source, events_csv_dir=src.metadata, **split, **common)
    raise NotImplementedError(key)


def print_plan(cfg):
    print(f"build_root = {cfg['build_root']}\n")
    for (ds, cat), _ in sources.PLAN.items():
        src = sources.plan_sources(cfg, ds, cat)
        print(f"{ds:15s} {cat:9s} -> {sources.output_paths(cfg, ds, cat)[0].name}")
        for name in ("source", "metadata", "extra_source", "extra_metadata"):
            v = getattr(src, name)
            if v is not None:
                print(f"    {name:15s} {'ok ' if Path(v).exists() else 'MISSING'} {v}")
        if src.options:
            print(f"    options         {src.options}")


def main():
    p = argparse.ArgumentParser(
        prog="python -m builder",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        default=None,
        help="YAML file with data locations (default: $RPM_BENCH_CONFIG; RPM_BENCH_<KEY> overrides)",
    )
    p.add_argument(
        "--list", action="store_true", help="print the build plan with resolved sources and exit"
    )
    p.add_argument("--dataset", choices=sorted({d for d, _ in ADAPTERS}))
    p.add_argument("--category", choices=sorted({c for _, c in ADAPTERS}))
    g = p.add_argument_group("source overrides (default: from the plan and config)")
    g.add_argument("--source", type=Path, default=None, help="primary data file or directory")
    g.add_argument("--metadata", type=Path, default=None, help="metadata CSV or directory")
    g.add_argument(
        "--extra-source", type=Path, default=None, help="secondary data file (INSTANCE noise)"
    )
    g.add_argument(
        "--extra-metadata",
        type=Path,
        default=None,
        help="secondary metadata (INSTANCE noise CSV, TW first-motion CSV)",
    )
    g.add_argument(
        "--exclude-traces",
        type=Path,
        default=None,
        help="STEAD noise: .npy or text list of trace names to leave out (default: <stead_root>/test.npy)",
    )
    o = p.add_argument_group(
        "outputs (default: <build_root>/<DATASET>/<DATASET>_dataset_90s_<CATEGORY>.*)"
    )
    o.add_argument("--out-h5", type=Path, default=None)
    o.add_argument("--out-csv", type=Path, default=None)
    o.add_argument("--overwrite", action="store_true", help="replace existing outputs")
    o.add_argument(
        "--validate", action="store_true", help="run assert_h5_consistency after the build"
    )
    b = p.add_argument_group("build options (defaults reproduce the archive build)")
    b.add_argument(
        "--max-samples", type=int, default=0, help="cap on written samples, 0 = all (smoke runs)"
    )
    b.add_argument(
        "--p-train",
        type=float,
        default=None,
        help="split P(train) (default 0.70; STEAD noise 0.80)",
    )
    b.add_argument(
        "--p-val", type=float, default=None, help="split P(val) (default 0.15; STEAD noise 0.20)"
    )
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--chunk-size", type=int, default=1024)
    b.add_argument(
        "--with-polarity",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="CEED: read first-motion polarity from the metadata (default on)",
    )
    b.add_argument(
        "--crew-phase-pair",
        choices=["composite", "mantle"],
        default=None,
        help="CREW singleEQ picks: 'mantle' = Pn+Sn only (default), 'composite' = first P/S",
    )
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()
    logging.basicConfig(level=a.log_level, format="%(asctime)s %(levelname)s %(message)s")
    cfg = sources.load(a.config)
    if a.list:
        print_plan(cfg)
        return
    if not (a.dataset and a.category):
        p.error("--dataset and --category are required (or --list)")
    key = (a.dataset, a.category)
    if key not in ADAPTERS:
        p.error(f"no archive for {key}; available: " + ", ".join(f"{d}/{c}" for d, c in ADAPTERS))
    plan = sources.plan_sources(cfg, *key)
    src = plan._replace(
        **{
            k: v
            for k, v in dict(
                source=a.source,
                metadata=a.metadata,
                extra_source=a.extra_source,
                extra_metadata=a.extra_metadata,
            ).items()
            if v is not None
        }
    )
    out_h5, out_csv = sources.output_paths(cfg, *key)
    out_h5, out_csv = a.out_h5 or out_h5, a.out_csv or out_csv
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    adapter = make_adapter(key, src, a)
    counts = build_h5(
        adapter,
        out_h5=out_h5,
        out_csv=out_csv,
        chunk_size=a.chunk_size,
        max_samples=a.max_samples or None,
        overwrite=a.overwrite,
    )
    print(f"\nBuild complete: {out_h5}\nPer-split: {counts}")
    if a.validate:
        assert_h5_consistency(out_h5, out_csv)


if __name__ == "__main__":
    main()
