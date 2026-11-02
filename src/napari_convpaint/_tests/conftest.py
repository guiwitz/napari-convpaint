import sys

if sys.platform == "win32":
    import torch  # must load before Qt/Napari on Windows

import os
import gc
import torch
import pytest

_mps_backend = getattr(torch.backends, "mps", None)  # absent before torch 1.12
_MPS_CLEANUP = (
    _mps_backend is not None
    and _mps_backend.is_built()
    and _mps_backend.is_available()
    and hasattr(torch, "mps")  # torch.mps added in torch 2.0
    and hasattr(torch.mps, "empty_cache")
    and os.environ.get("CONVPAINT_SKIP_MPS_CLEANUP") != "1"
)

# Fix for MPS memory leaks in tests:
@pytest.fixture(autouse=True)
def cleanup_mps_after_test():
    yield
    # delete any local refs in test code if possible, then:
    gc.collect()
    # empty PyTorch MPS cache if available
    if _MPS_CLEANUP:
        try:
            torch.mps.empty_cache()
        except Exception:
            pass