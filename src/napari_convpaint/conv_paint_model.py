
import pickle
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import warnings

from napari_convpaint.conv_paint_nnlayers import AVAILABLE_MODELS as NN_MODELS
from napari_convpaint.conv_paint_gaussian import AVAILABLE_MODELS as GAUSSIAN_MODELS
from napari_convpaint.conv_paint_dino import AVAILABLE_MODELS as DINO_MODELS
from napari_convpaint.conv_paint_gaussian import GaussianFeatures
from napari_convpaint.conv_paint_dino import DinoFeatures
from .conv_paint_nnlayers import Hookmodel
from .conv_paint_param import Param
from . import conv_paint_utils



class ConvpaintModel:

    def __init__(self, model_path=None):
        self.init_models_dict()
        self.classifier = None
        self.param = None
        self.fe_model = None
        self.fe_model_state = None
        # self.total_pad = None # Parameter to keep track of the overall padding needed (considering all scalings and padding)
        if model_path is not None:
            self.load_model(model_path)
        else:
            self.set_default_model()

    def init_models_dict(self):
        # Initialize the MODELS TO TYPES dictionary with the models that are always available
        self.ALL_MODELS_TYPES_DICT = {x: Hookmodel for x in NN_MODELS}
        self.ALL_MODELS_TYPES_DICT.update({x: GaussianFeatures for x in GAUSSIAN_MODELS})
        self.ALL_MODELS_TYPES_DICT.update({x: DinoFeatures for x in DINO_MODELS})

        # Try to import CellposeFeatures and update the MODELS TO TYPES  dictionary if successful
        # Cellpose is only installed with pip install napari-convpaint[cellpose]
        try:
            from napari_convpaint.conv_paint_cellpose import AVAILABLE_MODELS as CELLPOSE_MODELS
            from napari_convpaint.conv_paint_cellpose import CellposeFeatures
            self.ALL_MODELS_TYPES_DICT.update({x: CellposeFeatures for x in CELLPOSE_MODELS})
        except ImportError:
            # Handle the case where CellposeFeatures or its dependencies are not available
            print("Info: Cellpose is not installed and is not available as feature extractor.\n"
                "Run 'pip install napari-convpaint[cellpose]' to install it.")
            
    def set_default_model(self):
        self.param = Param()

        # ToDo: DEFINE DEFAULT CONVPAINT MODEL HERE

        self.set_fe_model(self.param)

    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        self.param = data['param']
        self.set_fe_model(self.param) # Resets classifier and model_state
        self.classifier = data['classifier']
        if 'model_state' in data:
            self.fe_model_state = data['model_state']
        # else:
        #     self.fe_model_state = None

    def save_model(self, model_path):
        with open(model_path, 'wb') as f:
            data = {
                'classifier': self.classifier,
                'param': self.param,
                'model_state': self.fe_model.state_dict() if hasattr(self.fe_model, 'state_dict') else None
            }
            pickle.dump(data, f)

    def set_fe_model(self):
        """Set the model based on the given parameters."""
        # self.total_pad = self.param.fe_padding * self.param.image_downsample * np.max(self.param.fe_scalings)
        self.fe_model_state = None
        self.classifier = None # TODO: Decide if Classifier should really be reset here
        model_class = self.ALL_MODELS_TYPES_DICT[self.param.fe_name]
        self.fe_model = model_class(
            model_name=self.param.fe_name,
            use_cuda=self.param.fe_use_cuda
        )
        
        if isinstance(self.fe_model, Hookmodel):
            if self.param.fe_layers:
                self.fe_model.register_hooks(selected_layers=self.param.fe_layers)
            elif len(self.fe_model.named_modules) == 1:
                self.fe_model.register_hooks(selected_layers=[list(self.fe_model.module_dict.keys())[0]])

    def train_classifier(self, features, targets, iterations = 50, learning_rate = 0.1, depth = 5, use_rf=False):
        """Train a classifier given a set of features and targets."""
        if not use_rf:
            self.classifier = CatBoostClassifier(iterations=iterations, learning_rate=learning_rate,depth=depth)
            self.classifier.fit(features, targets)
        else:
                # train a random forest classififer
                classifier = RandomForestClassifier(n_estimators=100, n_jobs=-1)
                classifier.fit(features, targets)

    def pre_process_img_annots_lists(self, img_list, annots_list):
        # If single image and single annot are given, convert them to lists
        if not isinstance(img_list, list):
            img_list = [img_list]
        if not isinstance(annots_list, list):
            annots_list = [annots_list]
        
        # Preprocess each image-annot pair
        output_img_list, output_annots_list = [], []
        for img_stack, annots_stack in zip(img_list, annots_list):
            img_stack, annots_stack = self.pre_process_img_annots_stack(img_stack, annots_stack)
            output_img_list.append(img_stack)
            output_annots_list.append(annots_stack)
        
        return output_img_list, output_annots_list

    def pre_process_img_annots_stack(self, img_stack, annots_stack):
        input_sclaing = self.param.image_downsample
        kernel_size = self.fe_model.kernel_size
        patch_size = self.fe_model.patch_size
        max_scaling = np.max(self.param.fe_scalings)
        # Process input image
        processed_img_stack = conv_paint_utils.pre_process_img_stack(img_stack, input_sclaing, kernel_size, patch_size, max_scaling)
        # Preprocess input annots
        processed_annots_stack = conv_paint_utils.pre_process_annots_stack(annots_stack, input_sclaing, kernel_size, patch_size, max_scaling)

    def get_features_targets_img_stack(self, img_stack, annots_stack):
        # Extract stacks with annotations
        annot_stacks = np.unique(np.where(annots_stack > 0)[0])
        if len(annot_stacks) == 0:
            warnings.warn('No annotations found')
            return None, None
        np.moveaxis(img_stack, -3, 0)
        img_stack = img_stack[annot_stacks]
        np.moveaxis(img_stack, 0, -3)
        annots_stack = annots_stack[annot_stacks]
        # Preprocess 
        output_img_stack, output_annots_stack = [], []
        for img, annots in zip(img_stack, annots_stack):
            img_tiles, annots_tiles = self.pre_process_img_annots_for_training(img, annots)
            output_img_stack.append(img_tiles)
            output_annots_stack.append(annots_tiles)

        # Create a numpy array with stack dimension as second dimension
        output_img_stack = np.stack(output_img_stack, axis=1)
        output_annots_stack = np.stack(output_annots_stack, axis=1)
        return output_img_stack, output_annots_stack


    def get_features_targets_multi_image(self, image_list, annots_list):
        all_features, all_targets = [], []
        for img, annots in zip(image_list, annots_list):
            features, targets = self.get_features_targets_img_stack(img, annots)