#!/usr/bin/env python
"""Fig. 3: plot joint F1 against S-P time from ``f1_vs_sp_time.csv`` (written by score_fig3.py).

The manuscript's ``make_fig_f1_sp.py`` with only its input and output paths made arguments:
reads ``--csv`` (per model, per 5 s S-P bin: sample count, the S-P-adaptive per-bin thresholds,
and the joint F1 at those thresholds) and writes ``--out`` as .pdf and .png. Needs matplotlib.

Example:
    python plot_fig3.py --csv results/f1_vs_sp_time.csv --out results/fig_f1_sp
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rpm_bench import config


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    config.add_config_arg(p)
    p.add_argument(
        "--csv", type=Path, default=None, help="input (default <results_root>/f1_vs_sp_time.csv)"
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output path; .pdf and .png are written (default <results_root>/fig_f1_sp)",
    )
    args = p.parse_args()
    cfg = config.load(args.config)
    csv = args.csv or cfg["results_root"] / "f1_vs_sp_time.csv"
    out = args.out or cfg["results_root"] / "fig_f1_sp"
    out.parent.mkdir(parents=True, exist_ok=True)

    # ---- plotting below unchanged from make_fig_f1_sp.py (only the paths differ) ----
    import matplotlib
    import numpy as np
    import pandas as pd

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tab = pd.read_csv(csv)
    LABELS = ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30", ">30"]
    # Mask-having models only: the joint (mask+P+S) F1 needs a detection output, so
    # PhaseNet is excluded (it has no mask; its detection was a P/S proxy). RED-PAN 60s
    # collapses at long S-P (60 s window can't hold both phases); both EQTransformer
    # weight sets (STEAD, INSTANCE) degrade most, matching the two EQT rows in the
    # external table; the two 90 s RED-PAN models hold up best and nearly overlap.
    MODELS = [
        "EdgeRP90",
        "RED-PAN-Motion",
        "RED-PAN 60s",
        "EQTransformer (STEAD)",
        "EQTransformer (INSTANCE)",
    ]
    # CSV keys (pipeline labels) -> paper display names.
    DISPLAY = {
        "EdgeRP90": "Edge-RED-PAN-Motion",
        "RED-PAN-Motion": "RED-PAN-Motion",
        "RED-PAN 60s": "RED-PAN",
        "EQTransformer (STEAD)": "EQTransformer (STEAD)",
        "EQTransformer (INSTANCE)": "EQTransformer (INSTANCE)",
    }
    counts = tab[tab.model == "EdgeRP90"].set_index("bin")["n"].reindex(LABELS).astype(int)

    # fonttype 42 embeds TrueType (not Type-3) fonts, per IEEE graphics requirements.
    # Generated at the native IEEE column width (~3.5 in) so \includegraphics[width=
    # \columnwidth] applies NO downscaling and the type stays at its set size.
    plt.rcParams.update(
        {"font.size": 8, "font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42}
    )
    fig, ax1 = plt.subplots(figsize=(3.5, 2.25))
    x = np.arange(len(LABELS))
    ax1.bar(x, counts.values, width=0.66, color="#c9ccd1", edgecolor="#9aa0a6", zorder=1)
    ax1.set_yscale("log")
    ax1.set_ylabel("Records / bin (log)", color="#5f6368", fontsize=7.5)
    ax1.set_xlabel("S-P differential time (s)", fontsize=7.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(LABELS, fontsize=6.5)
    ax1.tick_params(axis="y", labelcolor="#5f6368", labelsize=6.5)
    # Per-bar numeric labels are omitted at column size: the log axis conveys the
    # magnitudes and the caption gives the total, so no label can overlap the curves.

    ax2 = ax1.twinx()
    colors = {
        "EdgeRP90": "#0d8f83",
        "RED-PAN-Motion": "#b9740b",
        "RED-PAN 60s": "#4f46e5",
        "PhaseNet (STEAD)": "#c0392b",
        "EQTransformer (STEAD)": "#7f8c8d",
        "EQTransformer (INSTANCE)": "#37474f",
    }
    markers = {
        "EdgeRP90": "o",
        "RED-PAN-Motion": "s",
        "RED-PAN 60s": "^",
        "PhaseNet (STEAD)": "D",
        "EQTransformer (STEAD)": "v",
        "EQTransformer (INSTANCE)": "X",
    }
    for m in MODELS:
        sub = tab[tab.model == m].set_index("bin").reindex(LABELS)
        ax2.plot(
            x,
            sub.F1.values,
            marker=markers[m],
            color=colors[m],
            lw=1.4,
            ms=3.5,
            zorder=3,
            label=DISPLAY[m],
        )
    ax2.set_ylabel("Joint F1 (adaptive thresholds)", fontsize=7.5)
    ax2.set_ylim(0, 1.08)
    ax2.tick_params(axis="y", labelsize=6.5)
    bar_handle = plt.Rectangle((0, 0), 1, 1, fc="#c9ccd1", ec="#9aa0a6")
    h2, l2 = ax2.get_legend_handles_labels()
    # Legend below the axes, outside the data, so it never overlaps the curves or bars.
    leg = ax2.legend(
        h2 + [bar_handle],
        l2 + ["records / bin"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=3,
        fontsize=6.5,
        frameon=False,
        borderaxespad=0.0,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=1.3,
        labelspacing=0.5,
    )
    # No tight_layout: it clips the rotated right-axis label when a legend sits outside
    # the axes. bbox_inches="tight" + bbox_extra_artists captures every artist instead.
    for ext in ("pdf", "png"):
        fig.savefig(
            out.with_suffix(f".{ext}"),
            dpi=200,
            bbox_inches="tight",
            bbox_extra_artists=(leg,),
            pad_inches=0.04,
        )
    print("wrote", out.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
