import numpy as np
from .dino import DinoFeatures
from .nnlayers import Hookmodel
from .gaussian import GaussianFeatures
# from .cellpose import CellposeFeatures
from math import lcm

# AVAILABLE_MODELS = ['combo_dino_vgg', 'combo_dino_ilastik', 'combo_dino_gauss', 'combo_dino_cellpose', 'combo_vgg_ilastik']
AVAILABLE_MODELS = ['combo_dino_vgg', 'combo_dino_gauss']

COMBOS = {'combo_dino_vgg': {'constructors': [Hookmodel, DinoFeatures],
                             'model names': ['vgg16', 'dinov2_vits14_reg'],
                             'description': "Combining a default VGG16 with DINOv2."},
        #   'combo_dino_ilastik': {'constructors': [IlastikFeatures, DinoFeatures],
        #                      'model names': ['ilastik_2d', 'dinov2_vits14_reg'],
        #                      'description': "Combining Ilastik with DINOv2."},
        #   'combo_vgg_ilastik': {'constructors': [IlastikFeatures, Hookmodel],
        #                     'model names': ['ilastik_2d', 'vgg16'],
        #                     'description': "Combining Ilastik with a default VGG16."},
          'combo_dino_gauss': {'constructors': [GaussianFeatures, DinoFeatures],
                               'model names': ['gaussian_features', 'dinov2_vits14_reg'],
                               'description': "Combining a Gaussian filter with DINOv2."},
        #   'combo_dino_cellpose': {'constructors': [CellposeFeatures, DinoFeatures],
        #                           'model names': ['cellpose', 'dinov2_vits14_reg'],
        #                           'description': "Combining Cellpose with DINOv2."}
                       }

# For models that are optional, we need to handle ImportError
try:
    from .ilastik import AVAILABLE_MODELS as Ilastik_models, IlastikFeatures
    if Ilastik_models:
        COMBOS['combo_dino_ilastik'] = {'constructors': [IlastikFeatures, DinoFeatures],
                                'model names': ['ilastik_2d', 'dinov2_vits14_reg'],
                                'description': "Combining Ilastik with DINOv2."}
        AVAILABLE_MODELS.append('combo_dino_ilastik')
except Exception as e:
    print(f"Ilastik is not available, and therefore combo feature extractors involving Ilastik are not available.")
    pass

from ..feature_extractor import FeatureExtractor

class ComboFeatures(FeatureExtractor):
    """
    Class for combining two feature extractors.

    Parameters
    ----------
    model_name : str
        Name of the model, e.g. 'combo_dino_vgg'.

    Attributes
    ----------
    model : tuple
        Tuple of two FeatureExtractor objects.
    patch_size : int
        Patch size of the combined feature extractor.
    padding : int
        Padding of the combined feature extractor.
    """
    def __init__(self, model_name='combo_dino_vgg', **kwargs):
        
        super().__init__(model_name=model_name)

        self.model1, self.model2 = self.model

    @staticmethod
    def create_model(model_name):
        constructors = COMBOS[model_name]['constructors']
        names = COMBOS[model_name]['model names']
        if len(constructors) != 2:
            raise ValueError(f"Expected 2 constructors, got {len(constructors)}")
        if len(names) != 2:
            raise ValueError(f"Expected 2 model names, got {len(names)}")
        model1 = constructors[0](model_name=names[0])
        model2 = constructors[1](model_name=names[1])
        return (model1, model2)

    def get_description(self):
        desc = COMBOS[self.model_name]['description']
        return desc

    def get_default_params(self, param=None):
        param = super().get_default_params(param=param)
        param.fe_name = self.model_name
        param.fe_layers = None # Layers are not set by default, and should be chosen by the user among the proposed ones (which depend on the model)
        param.fe_scalings = [1]
        param.fe_order = 0
        param.tile_image = False
        param.tile_annotations = False
        return param
    
    def get_padding(self):
        p1 = self.model1.get_padding()
        p2 = self.model2.get_padding()
        self.padding = max(p1, p2)
        return self.padding
    
    def get_patch_size(self):
        ps1 = self.model1.get_patch_size()
        ps2 = self.model2.get_patch_size()
        self.patch_size = lcm(ps1, ps2)
        return self.patch_size
    
    def gives_patched_features(self):
        # Since we want to combine, we always rescale to image size, even if the models are patched
        # So, the combo FE itself is not patched, even if it works with a patch_size to comply with the models
        return False

    def extract_features_pyramid(self, image, param, patched=False, device=None, cancel_token=None):
        def1 = self.model1.get_default_params(param)
        features1 = self.model1.extract_features_pyramid(image, def1, patched=False, device=device, cancel_token=cancel_token)
        def2 = self.model2.get_default_params(param)
        features2 = self.model2.extract_features_pyramid(image, def2, patched=False, device=device, cancel_token=cancel_token)
        features = np.concatenate((features1, features2), axis=0)
        return features