from ilastik.napari.filters import (FilterSet,
                                    Gaussian,
                                    LaplacianOfGaussian,
                                    GaussianGradientMagnitude,
                                    DifferenceOfGaussians,
                                    StructureTensorEigenvalues,
                                    HessianOfGaussianEigenvalues)
import itertools
import numpy as np
from .conv_paint_feature_extractor import FeatureExtractor

AVAILABLE_MODELS = ['ilastik_2d']

# Define the filter set and scales
FILTER_LIST = (Gaussian,
               LaplacianOfGaussian,
               GaussianGradientMagnitude,
               DifferenceOfGaussians,
               StructureTensorEigenvalues,
               HessianOfGaussianEigenvalues)
SCALE_LIST = (0.3, 0.7, 1.0, 1.6, 3.5, 5.0, 10.0)
# Generate all combinations of FILTER_LIST and SCALE_LIST
ALL_FILTER_SCALING_COMBOS = list(itertools.product(range(len(FILTER_LIST)), range(len(SCALE_LIST))))
# Create a FilterSet with all combinations
FILTERS = tuple(FILTER_LIST[row](SCALE_LIST[col]) for row, col in sorted(ALL_FILTER_SCALING_COMBOS))
FILTER_SET = FilterSet(filters=FILTERS)

class IlastikFeatures(FeatureExtractor):
    def __init__(self, model_name='ilastik_2d', use_gpu=False, **kwargs):
        super().__init__(model_name=model_name, use_gpu=use_gpu)
        self.padding = FILTER_SET.kernel_size // 2

    def get_description(self):
        return "Extraction of image features using the full filter set of the popular segmentation tool Ilastik."
        
    def get_default_params(self, param=None):
        param = super().get_default_params(param=param)
        param.fe_layers = []
        return param

    def get_features_from_plane(self, image, filter_set=FILTER_SET):
        
        # Given that we get single-channel images as input:
        features = filter_set.transform(image[0]) # Ilastik wants 2D
        features = np.moveaxis(features, -1, 0)

        return features