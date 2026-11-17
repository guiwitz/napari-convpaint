import sys

if sys.platform == "win32":
    import torch  # must load before Qt/Napari on Windows

import os
import gc
import torch
import pytest

# Keep the tests out of the real user cache dir, where Convpaint puts the default feature store
# folder and the list of the folders it stored features in (platformdirs is imported where it is
# used, so patching the function on the module reaches every caller).
@pytest.fixture(autouse=True)
def convpaint_cache_dir_in_tmp(tmp_path_factory, monkeypatch):
    import platformdirs
    cache_dir = tmp_path_factory.mktemp('convpaint_cache')
    monkeypatch.setattr(platformdirs, 'user_cache_dir', lambda *args, **kwargs: str(cache_dir))

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