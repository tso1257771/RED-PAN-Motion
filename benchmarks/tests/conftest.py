"""Make ``rpm_bench`` and the entry-point modules importable when pytest runs from any directory
(for example ``PYTHONPATH=. python -m pytest benchmarks/tests`` from the repository root)."""

import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[1]
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))
