import numpy as np
import torch
from torch import nn
from ..utils import get_device_from_torch_model, guided_model_download

def import_models():
    try:
        from torchvision import models
    except ImportError:
        models = None
    return models

AVAILABLE_MODELS = ['vgg16', 'efficient_netb0', 'convnext']

STD_MODELS = {
    "vgg": {"fe_name": "vgg16"},
    "vgg-m": {
        "fe_name": "vgg16",
        "fe_layers": [
            "features.0 Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))",
            "features.2 Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))",
            "features.5 Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))",
        ],
        "fe_scalings": [1, 2, 4],
    },
    "vgg-l": {
        "fe_name": "vgg16",
        "fe_layers": [
            "features.0 Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))",
            "features.2 Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))",
            "features.5 Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))",
            "features.7 Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))",
            "features.10 Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))",
        ],
        "fe_scalings": [1, 2, 4, 8],
    },
}

from ..feature_extractor import FeatureExtractor

class Hookmodel(FeatureExtractor):
    """
    Class to extract features from a pytorch model using hooks on chosen layers.

    Parameters:
    ----------
    model_name : str
        Name of model to use. Currently only 'vgg16' and 'efficient_netb0' are supported.
    model : torch model, optional
        Model to extract features from, by default None
    layers : list of str or int, optional
        List of layer keys (if string) or indices (int) to extract features from, by default None

    Attributes:
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

    def __init__(self, model_name='vgg16', model=None, layers=None, **kwargs):
        
        super().__init__(model_name=model_name, model=model)

        self.norm_mode = "imagenet"  # All supported nn models expect ImageNet normalization
        self.rgb_input = True  # All supported nn models expect RGB input

        # REGISTER DEVICE OF THE MODEL
        self.device = get_device_from_torch_model(self.model)

        # INITIALIZATION OF LAYER HOOKS
        self.init_layer_dict()

        self.outputs = []
        # Cached (RF, stride, has_global_context) — invalidated when hooks are (re)registered.
        self._fe_properties_cache = None
        if layers is not None:
            self.register_hooks(layers)
        else:
            self.register_hooks(self.get_default_params().fe_layers)

    @staticmethod
    def create_model(model_name):

        models = import_models()
        if models is None:
            raise ImportError(
            "Torchvision models could not be imported. If called through ConvpaintModel, this should not happen as the availability of Torchvision is checked before. " +
            "Make sure to have torchvision installed and available in your environment."
        )

        # CREATE VGG16 MODEL
        if model_name == 'vgg16':
            model_file = 'vgg16-397923af.pth'
            model_url = f'https://download.pytorch.org/models/{model_file}'
            model_path = guided_model_download(model_file, model_url)
            model = models.vgg16()

        # CREATE EFFICIENTNETB0 MODEL
        elif model_name == 'efficient_netb0':
            model_file = 'efficientnet_b0_rwightman-7f5810bc.pth'
            model_url = f'https://download.pytorch.org/models/{model_file}'
            model_path = guided_model_download(model_file, model_url)
            model = models.efficientnet_b0()
        
        # CREATE ConvNeXt MODEL
        elif model_name == 'convnext':
            model_file = 'convnext_base-6075fbad.pth'
            model_url = f'https://download.pytorch.org/models/{model_file}'
            model_path = guided_model_download(model_file, model_url)
            model = models.convnext_base()
        
        else:
            raise ValueError(f"Model {model_name} is not supported. Available models: {AVAILABLE_MODELS}")

        # Load the model's state_dict
        state_dict = torch.load(model_path, weights_only=True)
        model.load_state_dict(state_dict)
        
        # Set model to evaluation mode. Device is selected at feature extraction time.
        model.eval()

        return model

    def get_description(self):
        if self.model_name == 'vgg16':
            desc = "CNN model trained on ImageNet data. First layers extract low-level features."
            desc += "\nAdd pyramid scalings to include broader context."
            desc += "\nGood for: differentiating textures, colours, brightness etc."
        elif self.model_name == 'efficient_netb0':
            desc = "EfficientNet model trained on ImageNet data."
        elif self.model_name == 'convnext':
            desc = "ConvNeXt model trained on ImageNet data."
        return desc

    def get_default_params(self, param=None):
        
        param = super().get_default_params(param=param)
        
        # self.init_layer_dict() # Is done at initialization (and should not change later)
        
        if self.model_name == 'vgg16':
            param.fe_scalings = [1,2,4]
            param.fe_layers = self.selectable_layer_keys[:1] # Use the first layer by default
        elif self.model_name == 'efficient_netb0':
            param.fe_scalings = [1,2]
            param.fe_layers = self.selectable_layer_keys[:1] # Use the first layer by default
        elif self.model_name == 'convnext':
            param.fe_scalings = [1,2]
            param.fe_layers = self.selectable_layer_keys[:3] # Use the first 3 layers by default

        param.tile_annotations = True # Overwrite non-FE settings

        return param

    def get_layer_keys(self):
        return self.selectable_layer_keys

    def init_layer_dict(self):
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
        # Half receptive-field bound: minimum context around each pixel for
        # its deepest selected feature to equal the infinite-image answer.
        return self._fe_properties()[0] // 2

    def get_patch_size(self):
        # Product of MaxPool kernel sizes and Conv2d strides up to the deepest
        # selected layer. Repurposed as the input-grid alignment requirement:
        # input dims must be a multiple of this for tile features to match
        # whole-image features. Combined with `gives_patched_features=False`
        # below, features still come back at input resolution.
        return self._fe_properties()[1]

    def gives_patched_features(self):
        # `patch_size` here encodes the alignment grid, not the output
        # resolution — features are returned at input resolution.
        return False

    def has_global_context(self):
        # True if an AdaptiveAvgPool2d / AdaptiveMaxPool2d sits on the path to
        # the deepest selected layer (e.g. SE blocks in EfficientNet) — no
        # finite padding makes tile features match whole-image features.
        return self._fe_properties()[2]

    def _fe_properties(self):
        """Cached `(receptive_field, alignment_stride, has_global_context)`
        from a single module-dict walk. Invariant once `register_hooks` ran."""
        if self._fe_properties_cache is None:
            self._fe_properties_cache = self._compute_fe_properties()
        return self._fe_properties_cache

    def _compute_fe_properties(self):
        """Walk the network in execution order, accumulating receptive field,
        total stride, and whether any global-context op was encountered, up to
        and including the deepest selected layer."""
        if len(self.selected_layers) == 0:
            return 1, 1, False
        rf = 1
        stride = 1
        has_global = False
        latest_layer = self.module_dict[self.selected_layers[-1]]
        for _, curr_layer in self.module_dict.items():
            if isinstance(curr_layer, (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)):
                has_global = True
            k = s = None
            if isinstance(curr_layer, nn.Conv2d):
                k = curr_layer.kernel_size[0] if isinstance(curr_layer.kernel_size, tuple) else curr_layer.kernel_size
                s = curr_layer.stride[0] if isinstance(curr_layer.stride, tuple) else curr_layer.stride
            elif isinstance(curr_layer, (nn.MaxPool2d, nn.AvgPool2d)) and hasattr(curr_layer, 'kernel_size'):
                ks = curr_layer.kernel_size
                st = curr_layer.stride if curr_layer.stride is not None else ks
                k = ks[0] if isinstance(ks, tuple) else ks
                s = st[0] if isinstance(st, tuple) else st
            if k is not None:
                rf = rf + (k - 1) * stride
                stride = stride * s
            if curr_layer is latest_layer:
                break
        return rf, stride, has_global

    def get_num_input_channels(self):
        return [self.named_modules[0][1].in_channels]
    
    def extract_features_from_stack(self, image, device=torch.device("cpu")):
        self.move_model_to_device(device)

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
        self._fe_properties_cache = None
        for ind in range(len(selected_layers)):
            self.features_per_layer.append(
                self.module_dict[selected_layers[ind]].out_channels)
            if ind == len(selected_layers) - 1:
                # print(f"registering LAST hook for layer {selected_layers[ind]}")
                self.module_dict[selected_layers[ind]].register_forward_hook(self.hook_last)
            else:
                # print(f"registering hook for layer {selected_layers[ind]}")
                self.module_dict[selected_layers[ind]].register_forward_hook(self.hook_normal)