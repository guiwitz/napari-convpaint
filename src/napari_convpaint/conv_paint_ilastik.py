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
    def __init__(self, model_name='ilastik_2d', use_cuda=False, **kwargs):
        super().__init__(model_name=model_name, use_cuda=use_cuda)
        self.padding = FILTER_SET.kernel_size // 2

    def get_description(self):
        return "Extraction of image features using the full filter set of the popular segmentation tool Ilastik."

    def get_features(self, image, filter_set=FILTER_SET, **kwargs):
        """
        Feature Extraction with Ilastik for single- or multi-channel images.
        INPUT:
            image (np.ndarray): image to predict on; shape (C, H, W) or (H, W, C) or (H, W)
            filter_set (FilterSet from ilastik.napari.filters): filter set to use for feature extraction
        OUTPUT:
            features (np.ndarray): feature map (H, W, F) with F being the number of features per pixel
        """
        # Extract features (depending on the number of channels)
        if image.ndim > 2:
            features = self.get_ila_features_multichannel(image)
        else:
            features = filter_set.transform(image)

        # Move the features to the first axis
        features = np.moveaxis(features, 2, 0)

        return features

    def get_ila_features_multichannel(self, image, filter_set=FILTER_SET):
        """
        Feature Extraction with Ilastik for multichannel images.
        Concatenates the feature maps of each channel.
        INPUT:
            image (np.ndarray): image to predict on; shape (C, H, W)
            filter_set (FilterSet from ilastik.napari.filters): filter set to use for feature extraction
        OUTPUT:
            features (np.ndarray): feature map (H, W, C) with C being the number of features per pixel
        """
        # Transform to (H, W, C) - expected by Ilastik
        # if len(image.shape) == 3 and image.shape[0] < 5:
        image = np.moveaxis(image, 0, -1)

        # Loop over channels, extract features and concatenate them
        for ch_idx in range(image.shape[2]):
            # channel_feature_map = np.zeros((100,100))
            channel_feature_map = filter_set.transform(image[:,:,ch_idx])
            if ch_idx == 0:
                feature_map = channel_feature_map
            else:
                feature_map = np.concatenate((feature_map, channel_feature_map), axis=2)

        return feature_map
        
    def get_default_params(self, param=None):
        param = super().get_default_params(param=param)
        param.fe_layers = []
        return param
