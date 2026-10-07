"""Constants shared by the runners and scorers.

The values are those of the manuscript runs; changing any of them changes published numbers.
Recipe-specific parameters (trigger thresholds, sweeps, the streaming replay settings in
``streaming_utils``) stay next to the code that uses them.
"""

DT = 0.01  # sample interval (s), 100 Hz
SENTINEL = -999.0  # "no value" in the per-record CSVs (no label, no trigger, no pick)

P_TOL_SEC = 0.5  # a P pick is correct within 0.5 s of the label
S_TOL_SEC = 1.0  # an S pick is correct within 1.0 s of the label

# Table III (mask, P and S) and the INT8 check (P and S): one probability threshold for every output
COMMON_THR = 0.3
