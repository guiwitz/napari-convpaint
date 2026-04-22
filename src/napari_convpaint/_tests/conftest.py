import sys

if sys.platform == "win32":
    import torch  # must load before Qt/Napari on Windows

import gc
import torch
import pytest

# Run long-running widget operations (train / predict / predict_all) on the
# calling thread in tests. The test assertions check layer state immediately
# after _on_train() / _on_predict(), which assumes synchronous execution.
from napari_convpaint.convpaint_widget import ConvpaintWidget  # noqa: E402
ConvpaintWidget._sync_workers = True


# Belt-and-suspenders MPS cleanup between tests. The root-cause fix for the
# worker-pinning leak is in convpaint_widget (we use superqt.utils.thread_worker
# so napari's task_status manager doesn't retain workers); this fixture adds an
# extra gc.collect() + torch.mps.empty_cache() after each test so any lingering
# per-test allocations don't carry over into the next one.
@pytest.fixture(autouse=True)
def cleanup_mps_after_test():
    yield
    gc.collect()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
