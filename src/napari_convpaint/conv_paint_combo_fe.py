import numpy as np
from .conv_paint_utils import get_device
from .conv_paint_feature_extractor import FeatureExtractor
from .conv_paint_dino import DinoFeatures
from .conv_paint_nnlayers import Hookmodel
from .conv_paint_gaussian import GaussianFeatures
# from .conv_paint_cellpose import CellposeFeatures
from math import lcm

# AVAILABLE_MODELS = ['combo_dino_vgg', 'combo_dino_ilastik', 'combo_dino_gauss', 'combo_dino_cellpose', 'combo_vgg_ilastik']
AVAILABLE_MODELS = ['combo_dino_vgg', 'combo_dino_gauss', 'combo_dino_ilastik']

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
                               'model names': ['gaussian', 'dinov2_vits14_reg'],
                               'description': "Combining a Gaussian filter with DINOv2."},
        #   'combo_dino_cellpose': {'constructors': [CellposeFeatures, DinoFeatures],
        #                           'model names': ['cellpose', 'dinov2_vits14_reg'],
        #                           'description': "Combining Cellpose with DINOv2."}
                       }

# For models that are optional, we need to handle ImportError
try:
    from .conv_paint_ilastik import IlastikFeatures
    COMBOS['combo_dino_ilastik'] = {'constructors': [IlastikFeatures, DinoFeatures],
                             'model names': ['ilastik_2d', 'dinov2_vits14_reg'],
                             'description': "Combining Ilastik with DINOv2."}
except ImportError:
    pass

class ComboFeatures(FeatureExtractor):
    """
    Class for combining two feature extractors.

    Parameters
    ----------
    model_name : str
        Name of the model, e.g. 'combo_dino_vgg'.
    use_gpu : bool
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
    def __init__(self, model_name='combo_dino_vgg', use_gpu=False):
        
        # Sets self.model_name and self.use_gpu and creates the model
        # Use "use_gpu=False" to force CPU mode; and move them manually below
        super().__init__(model_name=model_name, use_gpu=False)

        self.model1, self.model2 = self.model
        self.feature_extraction = False

        # Move all models to the device manually to ensure they are all on the same...
        self.use_gpu = use_gpu
        self.device = get_device(use_gpu)
        for fe in self.model:
            if fe.model is None:
                continue
            fe.model = fe.model.to(self.device)
            fe.model.eval()
            fe.device = self.device
            fe.use_gpu = self.use_gpu

    @staticmethod
    def create_model(model_name, use_gpu=False):
        constructors = COMBOS[model_name]['constructors']
        names = COMBOS[model_name]['model names']
        if len(constructors) != 2:
            raise ValueError(f"Expected 2 constructors, got {len(constructors)}")
        if len(names) != 2:
            raise ValueError(f"Expected 2 model names, got {len(names)}")
        model1 = constructors[0](model_name=names[0], use_gpu=use_gpu)
        model2 = constructors[1](model_name=names[1], use_gpu=use_gpu)
        return (model1, model2)

    def get_description(self):
        desc = COMBOS[self.model_name]['description']
        return desc

    def get_default_params(self, param=None):
        param = super().get_default_params(param=param)
        param.fe_name = self.model_name
        param.fe_use_gpu = self.use_gpu
        param.fe_layers = [0]
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

    def get_feature_pyramid(self, image, param, patched=False):
        def1 = self.model1.get_default_params(param)
        features1 = self.model1.get_feature_pyramid(image, def1, patched=False)
        def2 = self.model2.get_default_params(param)
        features2 = self.model2.get_feature_pyramid(image, def2, patched=False)
        features = np.concatenate((features1, features2), axis=0)
        return features