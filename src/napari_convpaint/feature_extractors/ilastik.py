import itertools
import math
import numpy as np
import importlib.util
import warnings

# fastfilters truncates each Gaussian / derivative kernel at half-length
# `ceil((3 + order/2) * sigma)` (fir_kernel.c). `SingleFilter.kernel_size` in
# ilastik.napari.filters returns `half_length + 1`, so `kernel_size // 2`
# under-pads by ~2×. For compound filters the reaches chain: StructureTensor
# computes a gradient at `inner_scale` (order=1) then smooths with a Gaussian
# at `outer_scale` (order=0), and the two half-lengths *add*. DifferenceOfGaussians
# runs two independent Gaussian smoothings and takes their difference, so the
# reach is the max of the two.

def _gauss_half_length(sigma, order):
    """fastfilters' default Gaussian kernel half-length (each side, in pixels)
    for derivative order `order`. Source: fir_kernel.c."""
    return math.ceil((3.0 + 0.5 * order) * sigma)

def _filter_reach(f):
    """Boundary reach (pixels each side) of an ilastik FilterSet member."""
    name = type(f).__name__
    if name == "StructureTensorEigenvalues":
        return _gauss_half_length(f.inner_k * f.scale, 1) + _gauss_half_length(f.scale, 0)
    if name == "DifferenceOfGaussians":
        return max(_gauss_half_length(f.scale, 0), _gauss_half_length(f.inner_k * f.scale, 0))
    return _gauss_half_length(f.scale, f.order)

def import_ilastik_filters():
    try:
        from ilastik.napari.filters import (
            FilterSet,
            Gaussian,
            LaplacianOfGaussian,
            GaussianGradientMagnitude,
            DifferenceOfGaussians,
            StructureTensorEigenvalues,
            HessianOfGaussianEigenvalues,
        )
    except ImportError:
        return None

    return {
        "FilterSet": FilterSet,
        "Gaussian": Gaussian,
        "LaplacianOfGaussian": LaplacianOfGaussian,
        "GaussianGradientMagnitude": GaussianGradientMagnitude,
        "DifferenceOfGaussians": DifferenceOfGaussians,
        "StructureTensorEigenvalues": StructureTensorEigenvalues,
        "HessianOfGaussianEigenvalues": HessianOfGaussianEigenvalues,
    }

# Check availability and provide infos for ConvpaintModel

def ilastik_available():
    available = importlib.util.find_spec("ilastik") is not None
    # if not available:
    #     warnings.warn(
    #         "Ilastik is not installed and is not available as feature extractor.\n"
    #         "Run 'pip install napari-convpaint[ilastik]' to install it.\n"
    #         "Make sure to also have fastfilters installed ('conda install -c ilastik-forge fastfilters')."
    #     )
    return available

AVAILABLE_MODELS = ['ilastik_2d'] if ilastik_available() else []

STD_MODELS = {
    "ilastik": {"fe_name": "ilastik_2d"},
}

IMPORT_ERROR_MESSAGE = (
            "Ilastik is not installed and is not available as feature extractor.\n"
            "Run 'pip install napari-convpaint[ilastik]' to install it.\n"
            "Make sure to also have fastfilters installed ('conda install -c ilastik-forge fastfilters')."
        )

# Actual feature extractor implementation

from ..feature_extractor import FeatureExtractor

class IlastikFeatures(FeatureExtractor):
    """Feature extractor using the full filter set of the popular segmentation tool Ilastik."""
    def __init__(self, model_name='ilastik_2d', **kwargs):
        global FILTER_SET
        FILTER_SET = create_filter_set()
        super().__init__(model_name=model_name)
        self.padding = max(_filter_reach(f) for f in FILTER_SET.filters)

    def get_description(self):
        return "Extraction of image features using the full filter set of the popular segmentation tool Ilastik."
        
    def get_default_params(self, param=None):
        param = super().get_default_params(param=param)
        param.fe_layers = None
        return param

    def extract_features_from_plane(self, image, device=None, filter_set=None):
        if filter_set is None:
            filter_set = FILTER_SET
        
        # Given that we get single-channel images as input:
        features = filter_set.transform(image[0]) # Ilastik wants 2D
        features = np.moveaxis(features, -1, 0)

        return features
    
def create_filter_set():
    filters_mod = import_ilastik_filters()
    if filters_mod is None:
        raise ImportError(
            "Ilastik filters could not be imported. If called through ConvpaintModel, this should not happen as the availability of Ilastik is checked before. " +
            "Make sure to have ilastik.napari.filters installed and available in your environment."
        )

    FILTER_LIST = (
        filters_mod["Gaussian"],
        filters_mod["LaplacianOfGaussian"],
        filters_mod["GaussianGradientMagnitude"],
        filters_mod["DifferenceOfGaussians"],
        filters_mod["StructureTensorEigenvalues"],
        filters_mod["HessianOfGaussianEigenvalues"],
    )

    SCALE_LIST = (0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0)

    combos = itertools.product(range(len(FILTER_LIST)), range(len(SCALE_LIST)))

    filters = tuple(
        FILTER_LIST[row](SCALE_LIST[col])
        for row, col in combos
    )

    return filters_mod["FilterSet"](filters=filters)