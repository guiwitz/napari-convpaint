import skimage
import numpy as np
from .conv_paint_feature_extractor import FeatureExtractor


# 1) LIST THE AVAILABLE MODELS HERE
AVAILABLE_MODELS = [] # List the available model names here


# 2) DEFINE THE INITIALIZATION, DESCRIPTION AND DEFAULT PARAMETERS
class GaussianFeatures(FeatureExtractor):
    # Define the initiation method
    def __init__(self, model_name='gaussian_features', use_gpu=False):
        # Sets some default parameters and chooses the itialization method
        super().__init__(model_name=model_name, use_gpu=use_gpu)

        # Define specifications for the feature extractor model, if necessary
        # self.padding = 0
        # self.patch_size = 1
        # self.num_input_channels = [1]
        # self.norm_imagenet = False

    def get_description(self):
        return "" # Briefly describe the feature extractor here

    def get_default_params(self, param=None):
        param = super().get_default_params(param=param)
        
        # Define here, which parameters shall be used as default parameters
        # param.fe_layers = []
        # param.fe_scalings = [1]
        # etc.

        return param


# 3) OPTIONAL METHODS:

# FOR MODELS THAT NEED TO CREATE A SEPARATE FE MODEL, DEFINE THE FOLLOWING METHOD

    # @staticmethod
    # def create_model(model_name):
    #     pass

# IF THE MODEL ABSOLUTELY REQUIRES SOME PARAMETERS TO BE SET, DEFINE THE FOLLOWING METHOD
    
        # def get_enforced_params(self, param=None):
        #     param = super().get_enforced_params(param=param)
            
        #     # Define here, which parameters shall be used as enforced parameters
        #     # param.fe_layers = []
        #     # param.fe_scalings = [1]
        #     # etc.
    
        #     return param


# 4) CHOSE BETWEEN THE FOLLOWING METHODS HOW TO IMPLEMENT THE FEATURE EXTRACTION
#    WITH INCREASING CONTROL AND COMPLEXITY:

# a) get_features_from_plane(self, image):
#    
#    Define how to extract features from a single plane of the image, assuming the number of
#    channels is compatible with the model (self.num_input_channels). Input = [C, H ,W]
#    Important: Output needs to be (a list of) 3D: [nb_features, H, W]
#    
#    If this is given, stacks are simply treated as independent planes and their features
#    stacked back along the z axis. Also, the channels are automatically repeated to suit the
#    required input channels, and the output of the channels are treated as different features.
#    Finally, the image is automatically scaled according to the scaling factors in the
#    parameters, reduced to patch size. The output is then scaled back to the original size.

# b) get_features(self, image):
#
#    Define how to extract features from an image stack (4D), assuming the number of input channels
#    is compatible with the model (self.num_input_channels). Input = [C, Z, H ,W]
#    Important: Output needs to be (a list of) 4D: [nb_features, Z, H, W]
#    
#    If this is given, the channels are automatically repeated to suit the required input channels,
#    and the output of the channels are treated as different features.
#    Finally, the image is automatically scaled according to the scaling factors in the
#    parameters, reduced to patch size. The output is then scaled back to the original size.

# c) get_features_from_channels(self, image):
#
#    Define how to extract features from an image stack (4D) with an arbitrary number of channels.
#    Input = [C, Z, H ,W]
#    Important: Output needs to be (a list of) 4D: [nb_features, Z, H, W]
#
#    If this is given, the image is automatically scaled according to the scaling factors in the
#    parameters, reduced to patch size. The output is then scaled back to the original size.
    
# d) get_feature_pyramid(self, data, param, patched=True):
#
#    Define the full feature extraction process, including the feature pyramid. Input = [C, Z, H ,W]
#    Important: Output needs to be 4D: [nb_features, Z, H, W]


# 5) TO USE THE FEATURE EXTRACTOR CLASS INSIDE THE CONVPAINT MODEL, YOU NEED TO FOLLOW THE FOLLOWING STEPS:

# - Import the AVAILABLE_MODELS and the class of the feature extractor
# - Implement to add the model to the FE_MODELS_TYPES_DICT in the _init_fe_models_dict method
# - Optional: Add it to the dictionary std_models, to enable loading it with an alias
