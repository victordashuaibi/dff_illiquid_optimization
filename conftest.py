"""Pytest bootstrap.

Two responsibilities, in this order (env vars first):

1. Suppress the macOS OpenMP duplicate-runtime deadlock. ``torch`` ships
   with LLVM ``libomp``; ``xgboost`` ships with Intel ``libiomp``. When
   both are loaded into the same Python process and both call into
   OpenMP barriers (e.g., during ``xgb.fit`` after a ``torch`` import),
   the threads deadlock — confirmed via ``sample`` stack trace on this
   venv. Workaround: tell Intel OMP not to abort on duplicate, and pin
   the thread count to 1 so the two runtimes never race in a barrier.
   Both env vars must be set *before* numpy/torch/xgboost are imported,
   so they live here at the top of conftest.py rather than in any test.

2. Put the project root on ``sys.path`` so ``import src.*`` works.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
