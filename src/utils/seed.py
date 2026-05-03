"""Reproducibility utilities.

``set_seed`` covers the four RNGs we use in the DFF stack: stdlib
``random``, ``numpy``, ``torch`` (CPU + CUDA), plus the cuDNN flags. The
``PYTHONHASHSEED`` env var only affects subprocesses launched after
``set_seed`` returns — for full single-process determinism, also export
``PYTHONHASHSEED=42`` before invoking Python (or rely on the fact that
none of the deterministic code in this project depends on dict-iteration
order).
"""
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
