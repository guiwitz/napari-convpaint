from collections import OrderedDict

import numpy as np
import pandas as pd
import skimage
import torch
from torch import nn
from torch.nn.functional import interpolate as torch_interpolate
from torchvision import models
from .conv_paint_utils import get_device, scale_img
from .conv_paint_feature_extractor import FeatureExtractor
from .conv_paint_param import Param


AVAILABLE_MODELS = ['vgg16', 'efficient_netb0', 'convnext']

class Hookmodel(FeatureExtractor):
    """
    Class to extract features from a pytorch model using hooks on chosen layers.

    Parameters:
    ----------
    model_name : str
        Name of model to use. Currently only 'vgg16' and 'efficient_netb0' are supported.
    model : torch model, optional
        Model to extract features from, by default None
    use_cuda : bool, optional
        Use cuda, by default False
    layers : list of str or int, optional
        List of layer keys (if string) or indices (int) to extract features from, by default None

    Attributes
    ----------
    model : torch model
        Model to extract features from
    outputs : list of torch tensors
        List of outputs from hooks
    features_per_layer : list of int
        Number of features per hooked layer
    module_list : list of lists
        List of hooked layers, their names, and their indices in the model (if applicable)
    """

    def __init__(self, model_name='vgg16', model=None, use_cuda=None, layers=None):
        
        super().__init__(model_name=model_name, model=model, use_cuda=use_cuda)

        # SET DEVICE
        self.device = get_device(use_cuda)
        self.model = self.model.to(self.device)
        self.model.eval()

        # INITIALIZATION OF LAYER HOOKS
        self.update_layer_dict()

        self.outputs = []
        if layers is not None:
            self.register_hooks(layers)
        else:
            self.register_hooks(self.get_default_params().fe_layers)

    @staticmethod
    def create_model(model_name):

        # CREATE VGG16 MODEL
        if model_name == 'vgg16':
            return models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            # self.transform = models.VGG16_Weights.IMAGENET1K_V1.transforms()

        # CREATE EFFICIENTNETB0 MODEL
        elif model_name == 'efficient_netb0':
            return models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # CREATE ConvNeXt MODEL
        elif model_name == 'convnext':
            return models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)

    def get_description(self):
        if self.model_name == 'vgg16':
            desc = "CNN model. First layers extract low-level features. Add pyramid scalings to include broader context."
            desc += "\nGood for: differentiating textures, colours, brightness etc."
        elif self.model_name == 'efficient_netb0':
            desc = "EfficientNet model trained on ImageNet data."
        elif self.model_name == 'convnext':
            desc = "ConvNeXt model trained on ImageNet data."
        return desc

    def get_default_params(self, param=None):
        
        param = super().get_default_params(param=param)
        
        # self.update_layer_dict() # Is done at initialization (and should not change later)
        
        if self.model_name == 'vgg16':
            param.fe_scalings = [1,2,4]
            param.fe_layers = self.selectable_layer_keys[:1] # Use the first layer by default
        elif self.model_name == 'efficient_netb0':
            param.fe_scalings = [1,2]
            param.fe_layers = self.selectable_layer_keys[:1] # Use the first layer by default
        elif self.model_name == 'convnext':
            param.fe_scalings = [1,2,4]
            param.fe_layers = self.selectable_layer_keys[:2]

        param.tile_annotations = True # Overwrite non-FE settings

        return param

    def get_layer_keys(self):
        return self.selectable_layer_keys

    def update_layer_dict(self):
        """Create a flat list of all modules as well as a dictionary of modules with 
        keys describing the layers."""

        named_modules = list(self.model.named_modules())
        # Remove modules with submodules
        self.named_modules = [n for n in named_modules if len(list(n[1].named_modules())) == 1]
        # Create dictionaries for easy access
        self.module_id_dict = dict([(x[0] + ' ' + x[1].__str__(), x[0]) for x in self.named_modules])
        self.module_dict = dict([(x[0] + ' ' + x[1].__str__(), x[1]) for x in self.named_modules])
        
        self.selectable_layer_dict = dict([(x[0] + ' ' + x[1].__str__(), x[1]) for x in self.named_modules if isinstance(x[1], nn.Conv2d)])
        self.selectable_layer_keys = list(self.selectable_layer_dict.keys())

    def layers_to_keys(self, layers):
        if all([isinstance(x, int) for x in layers]):
            return [self.selectable_layer_keys[x] for x in layers]
        else:
            return layers
    
    def get_padding(self):
        ks = self.get_max_kernel_size()
        return ks // 2

    def get_max_kernel_size(self):
        """
        Given a hookmodel, find the maximum kernel size needed for the deepest layer.
        
        Parameters: None
        ----------
        
        Returns
        -------
        max_kernel_size: int
            maximum kernel size needed for the deepest layer
        """
        # Initialize variables
        max_kernel_size = 1
        current_total_pool = 1

        if len(self.selected_layers) == 0:
            # no layers selected yet
            return 0
        
        # Find out which is the deepest layer
        latest_layer = self.module_dict[self.selected_layers[-1]]
        # Iterate over all layers to find the maximum kernel size
        for curr_layer_key, curr_layer in self.module_dict.items():
            # If a maxpool layer is involved, kernel size needs to be multiplied for all future convolutions
            if "MaxPool2d" in str(curr_layer) and hasattr(curr_layer, 'kernel_size'):
                current_total_pool *= curr_layer.kernel_size
            # For each convolution, multiply the kernel size with the current total pool
            elif "Conv2d" in str(curr_layer) and hasattr(curr_layer, 'kernel_size'):
                max_kernel_size = current_total_pool * curr_layer.kernel_size[0]
            # Only iterate until the latest selected layer
            if curr_layer == latest_layer:
                break
        return max_kernel_size
    
    def get_num_input_channels(self):
        return [self.named_modules[0][1].in_channels]
    
    def extract_features_new(self, image):
        # Convert image to numpy array and ensure correct data type
        image = np.asarray(image, dtype=np.float32)

        self.outputs = []
        with torch.no_grad():
            # Treat z as batch dimension (temprorarily)
            ch_torch = torch.tensor(np.moveaxis(image, 1, 0))
            try:
                _ = self(ch_torch) # Forward pass through the model
            except AssertionError as ea:
                pass # Stop at hook
            except Exception as ex:
                raise ex
            
        # Move the z dimension back to the second position (and features to first)
        outputs = [o.permute(1, 0, 2, 3) for o in self.outputs]

        return outputs

    def __call__(self, tensor_image):
        tensor_image_dev = tensor_image.to(self.device)
        return self.model(tensor_image_dev)

    def hook_normal(self, module, input, output):
        # print("extracting with normal layer")
        self.outputs.append(output)

    def hook_last(self, module, input, output):
        # print("extracting with last layer")
        self.outputs.append(output)
        assert False

    def register_hooks(self, selected_layers):  # , selected_layer_pos):
        selected_layers = self.layers_to_keys(selected_layers)
        self.features_per_layer = []
        self.selected_layers = selected_layers.copy()
        for ind in range(len(selected_layers)):
            self.features_per_layer.append(
                self.module_dict[selected_layers[ind]].out_channels)
            if ind == len(selected_layers) - 1:
                # print(f"registering LAST hook for layer {selected_layers[ind]}")
                self.module_dict[selected_layers[ind]].register_forward_hook(self.hook_last)
            else:
                # print(f"registering hook for layer {selected_layers[ind]}")
                self.module_dict[selected_layers[ind]].register_forward_hook(self.hook_normal)