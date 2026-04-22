import os
# Ensure that any missing MPS kernels will fall back to CPU:
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"


# Import the model, widget and feature extractor superclass to make them available at the package level
from .convpaint_model import ConvpaintModel
from .convpaint_widget import ConvpaintWidget
from .feature_extractor import FeatureExtractor
from .utils import CancelToken, CancelledError