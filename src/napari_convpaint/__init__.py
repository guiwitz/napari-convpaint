import os
# Ensure that any missing MPS kernels will fall back to CPU:
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"