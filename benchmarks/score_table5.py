#!/usr/bin/env python
"""Table V: first-motion polarity on CEED test records with an up or down label.

Input: polarity/<model>_polarity_per_event.csv from run_polarity.py (3-class softmax [N, U, D] read
at the labeled P). A record is covered at threshold t if max(P(U), P(D)) >= t; its class is the
larger of the two (for t >= 0.5 this equals the [N, U, D] argmax, since the three sum to 1).
Coverage = covered fraction; accuracy and recall up / down over covered records. McNemar: records
covered by both models, exact binomial test on the discordant pairs (chi-square with continuity
correction also reported).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rpm_bench import config
from rpm_bench.results import polarity_csv, write_json
from scipy.stats import binomtest, chi2


def cover(ud, t):
    """(covered at threshold t, predicted class U / D) per record."""
    u, d = ud.polarity_u_prob.values, ud.polarity_d_prob.values
    return np.maximum(u, d) >= t, np.where(u >= d, "U", "D")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    config.add_config_arg(p)
    p.add_argument(
        "--models", nargs=2, default=["edge", "rpm"], help="two models (the McNemar pair)"
    )
    p.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.7])
    p.add_argument("--out-json", type=Path, default=None, help="also write the results as JSON")
    args = p.parse_args()
    cfg = config.load(args.config)
    D = {
        m: pd.read_csv(
            polarity_csv(cfg, m),
            usecols=["trace_name", "gt_polarity", "polarity_u_prob", "polarity_d_prob"],
        )
        for m in args.models
    }
    a, b = args.models
    if len(D[a]) != len(D[b]) or not (D[a].trace_name.values == D[b].trace_name.values).all():
        raise SystemExit("the two per-record files must list the same records in the same order")
    res = {
        "n_evaluated": len(D[a]),
        "label_counts": D[a].gt_polarity.value_counts().to_dict(),
        "table": {},
        "mcnemar": {},
    }
    U = {m: d[d.gt_polarity.isin(["U", "D"])].reset_index(drop=True) for m, d in D.items()}
    res["n_ud"] = len(U[a])
    res["frac_up"] = float((U[a].gt_polarity == "U").mean())
    print(
        f"evaluated {res['n_evaluated']:,}; U/D records {res['n_ud']:,} (up {100 * res['frac_up']:.1f}%)"
    )
    print(f"{'model, threshold':18s} coverage accuracy recall_up recall_down")
    for m in args.models:
        for t in args.thresholds:
            c, k = cover(U[m], t)
            g = U[m].gt_polarity.values
            v = dict(
                coverage=float(c.mean()),
                accuracy=float((k[c] == g[c]).mean()),
                recall_up=float((k[c & (g == "U")] == "U").mean()),
                recall_down=float((k[c & (g == "D")] == "D").mean()),
            )
            res["table"][f"{m}|{t}"] = v
            print(
                f"{m + ', ' + str(t):18s} {v['coverage']:8.4f} {v['accuracy']:8.4f} {v['recall_up']:9.4f} {v['recall_down']:11.4f}"
            )
    g = U[a].gt_polarity.values
    for t in args.thresholds:
        ca, ka = cover(U[a], t)
        cb, kb = cover(U[b], t)
        both = ca & cb
        oa, ob = (ka == g) & both, (kb == g) & both
        n1, n2 = int((oa & ~ob).sum()), int((~oa & ob).sum())
        rec = dict(
            n_both=int(both.sum()),
            acc_a=float(oa.sum() / both.sum()),
            acc_b=float(ob.sum() / both.sum()),
            only_a_correct=n1,
            only_b_correct=n2,
            p_exact=float(binomtest(n1, n1 + n2, 0.5).pvalue) if n1 + n2 else 1.0,
            p_chi2_cc=float(chi2.sf((abs(n1 - n2) - 1) ** 2 / (n1 + n2), 1)) if n1 + n2 else 1.0,
        )
        res["mcnemar"][str(t)] = rec
        print(
            f"McNemar t={t}: n_both={rec['n_both']:,} acc {a} {rec['acc_a']:.4f} / {b} {rec['acc_b']:.4f}; "
            f"only {a} {n1}, only {b} {n2}; p_exact={rec['p_exact']:.3g} (chi2 {rec['p_chi2_cc']:.3g})"
        )
    if args.out_json:
        write_json(args.out_json, res)


if __name__ == "__main__":
    main()
