"""Small NVTX helpers for optional Nsight Systems profiling."""

import contextlib
import os

import torch


@contextlib.contextmanager
def nvtx_range(name):
    """Optional NVTX range enabled only when ENABLE_NSYS=1."""
    enabled = os.getenv("ENABLE_NSYS", "0") == "1" and torch.cuda.is_available()
    if enabled:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()
