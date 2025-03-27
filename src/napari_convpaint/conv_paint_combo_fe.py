import numpy as np
from .conv_paint_utils import get_device
from .conv_paint_feature_extractor import FeatureExtractor
from .conv_paint_dino import DinoFeatures
from .conv_paint_nnlayers import Hookmodel
# from .conv_paint_ilastik import IlastikFeatures
from .conv_paint_gaussian import GaussianFeatures
# from .conv_paint_cellpose import CellposeFeatures
from math import gcd

# AVAILABLE_MODELS = ['combo_dino_vgg', 'combo_dino_ila', 'combo_dino_gauss', 'combo_dino_cellpose', 'combo_vgg_ila']
AVAILABLE_MODELS = ['combo_dino_vgg', 'combo_dino_gauss']

COMBOS = {'combo_dino_vgg': {'constructors': [Hookmodel, DinoFeatures],
                             'model names': ['vgg16', 'dinov2_vits14_reg'],
                             'description': "Combining a default VGG16 with DINOv2."},
        #   'combo_dino_ila': {'constructors': [IlastikFeatures, DinoFeatures],
        #                      'model names': ['ilastik_2d', 'dinov2_vits14_reg'],
        #                      'description': "Combining Ilastik with DINOv2."},
        #   'combo_vgg_ila': {'constructors': [IlastikFeatures, Hookmodel],
        #                     'model names': ['ilastik_2d', 'vgg16'],
        #                     'description': "Combining Ilastik with a default VGG16."},
          'combo_dino_gauss': {'constructors': [GaussianFeatures, DinoFeatures],
                               'model names': ['gaussian', 'dinov2_vits14_reg'],
                               'description': "Combining a Gaussian filter with DINOv2."},
        #   'combo_dino_cellpose': {'constructors': [CellposeFeatures, DinoFeatures],
        #                           'model names': ['cellpose', 'dinov2_vits14_reg'],
        #                           'description': "Combining Cellpose with DINOv2."}
                       }

class ComboFeatures(FeatureExtractor):
    """
    Class for combining two feature extractors.

    Parameters
    ----------
    model_name : str
        Name of the model, e.g. 'combo_dino_vgg'.
    use_cuda : bool
        If True, use the GPU for feature extraction.

    Attributes
    ----------
    model : tuple
        Tuple of two FeatureExtractor objects.
    patch_size : int
        Patch size of the combined feature extractor.
    padding : int
        Padding of the combined feature extractor.
    device : str
        Device used for feature extraction.
    """
    def __init__(self, model_name='combo_dino_vgg', use_cuda=False):
        
        # Sets self.model_name and self.use_cuda and creates the model
        super().__init__(model_name=model_name, use_cuda=use_cuda)
        
        self.model1, self.model2 = self.model
        ps1 = self.model1.get_patch_size()
        ps2 = self.model2.get_patch_size()
        self.patch_size = abs(ps1 * ps2) // gcd(ps1, ps2) # = lcm(ps1, ps2)
        p1 = self.model1.get_padding()
        p2 = self.model2.get_padding()
        self.padding = max(p1, p2)

        for fe in self.model:
            if fe.model is None:
                continue
            if use_cuda:
                self.device = get_device()
                fe.model.to(self.device)
            else:
                self.device = 'cpu'
            fe.model.eval()

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
        param.fe_use_cuda = self.use_cuda
        param.fe_layers = [0]
        param.fe_scalings = [1]
        param.fe_order = 0
        param.tile_image = False
        param.tile_annotations = False
        return param
    
    def get_features_scaled(self, image, param):
        '''Given an CxWxH image, extract features.
        Returns features with dimensions nb_features x H x W'''
        def1 = self.model2.get_default_params(param)
        dino_features = self.model2.get_features_scaled(image, def1)
        def2 = self.model1.get_default_params(param)
        vgg_features = self.model1.get_features_scaled(image, def2)
        features = np.concatenate((dino_features, vgg_features), axis=0)
        return features
