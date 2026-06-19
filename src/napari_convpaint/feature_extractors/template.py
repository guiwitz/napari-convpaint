from ..feature_extractor import FeatureExtractor


# 1) LIST THE AVAILABLE MODELS (REQUIRED):

# This can just be a model name (string) that is used to recognize your model,
# or multiple names to differentiate between potentially multiple different versions
AVAILABLE_MODELS = ["gaussian_features"] # List the available model names here

# OPTIONAL: Add options to the dictionary std_models, to enable loading this model by alias in the API
STD_MODELS = {
    "gaussian": {"fe_name": "gaussian_features"}, # Example: this allows to load the gaussian feature extractor by just setting fe_name = "gaussian" in the parameters
    # You can add more aliases; and you can specify more parameters in the dictionary for an alias, if you want to set some parameters as default for this alias. For example:
    "gaussian_2": {"fe_name": "gaussian_features", "image_downsample": 2} # A version with downsampling by 2 as default
}


# 2) DEFINE INITIALIZATION, DESCRIPTION AND DEFAULT PARAMETERS (TECHNICALLY OPTIONAL, BUT STRONGLY RECOMMENDED):

class GaussianFeatures(FeatureExtractor):
    # Define the initiation method
    def __init__(self, model_name='gaussian_features'):
        # __init__ of FeatureExtractor superclass sets some default parameters and chooses the itialization method
        super().__init__(model_name=model_name)

        # Define specifications for the feature extractor model, if necessary; examples:
        # self.padding = 0 # If the model needs a certain padding around the extracted pixel, set it here. This is used to calculate the necessary padding for tiling.
        # self.patch_size = 1 # If the model produces features at a lower resolution than the input image, e.g. because of pooling or ViT patching, set the patch size here. This is used to calculate the necessary padding and alignment for tiling.
        # self.has_global_context = False # True if the model's features at a pixel depend on the whole image (e.g. ViT-style global attention) rather than just a local neighborhood (e.g. CNN with small kernels and little pooling)
        # self.num_input_channels = [1]
        # self.norm_mode = "default" # or "imagenet" or "percentile"
        # self.rgb_input = False # True if the model takes RGB input

    def get_description(self):
        # Briefly describe the feature extractor here; this will be displayed in the UI
        return ""

    def get_default_params(self, param=None):
        # Get the default parameters for the feature extractor; this allows you to only set the parameters that are relevant for your model
        param = super().get_default_params(param=param)
        
        # Define here, which parameters shall be used as default parameters
        # param.fe_layers = []
        # param.fe_scalings = [1]
        # etc.

        return param


# 3) METHODS FOR MORE ADVANCED FEATURE EXTRACTORS (OPTIONAL):

    # FOR MODELS THAT NEED TO CREATE AN UNDERLYING FE MODEL (E.G. TORCH MODEL), DEFINE THE FOLLOWING METHOD

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


# 4) IMPLEMENT THE FEATURE EXTRACTION

# CHOOSE A POINT OF ENTRY BY OVERRIDING ONE OF THE METHODS BELOW WITH INCREASING CONTROL AND COMPLEXITY

# IMPORTANT for GPU support:
# Independent of the extraction method chosen, if want the feature extractor to be able to use GPU,
# the class needs to implement a way to move its internal model (if applicable) and the tensors to the
# appropriate device (CPU or GPU, with cuda or mps).
# For standard torch models, you can use the move_model_to_device() method of the base FeatureExtractor class
# to move the model. For other models, you need to implement a similar logic of moving model and tensors.

# a) extract_features_from_plane(self, image, device):
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

# b) extract_features_from_stack(self, image, device):
#
#    Define how to extract features from an image stack (4D), assuming the number of input channels
#    is compatible with the model (self.num_input_channels). Input = [C, Z, H ,W]
#    Important: Output needs to be (a list of) 4D: [nb_features, Z, H, W]
#    
#    If this is given, the channels are automatically repeated to suit the required input channels,
#    and the output of the channels are treated as different features.
#    Finally, the image is automatically scaled according to the scaling factors in the
#    parameters, reduced to patch size. The output is then scaled back to the original size.

# c) extract_features_from_multichannel_stack(self, image, rgb_data, device):
#
#    Define how to extract features from an image stack (4D) with an arbitrary number of channels.
#    Input = [C, Z, H ,W]
#    Important: Output needs to be (a list of) 4D: [nb_features, Z, H, W]
#
#    If this is given, the image is automatically scaled according to the scaling factors in the
#    parameters, reduced to patch size. The output is then scaled back to the original size.
    
# d) extract_features_pyramid(self, data, param, patched=True, device):
#
#    Define the full feature extraction process, including the feature pyramid. Input = [C, Z, H ,W]
#    Important: Output needs to be 4D: [nb_features, Z, H, W]