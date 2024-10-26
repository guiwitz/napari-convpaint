import numpy as np
import pandas as pd
import skimage
import warnings
from catboost import CatBoostClassifier
from napari_convpaint.conv_paint_nnlayers import AVAILABLE_MODELS as NN_MODELS
from napari_convpaint.conv_paint_gaussian import AVAILABLE_MODELS as GAUSSIAN_MODELS
from napari_convpaint.conv_paint_dino import AVAILABLE_MODELS as DINO_MODELS
from napari_convpaint.conv_paint_gaussian import GaussianFeatures
from napari_convpaint.conv_paint_dino import DinoFeatures
from .conv_paint_nnlayers import Hookmodel
import pickle
from .conv_parameters import Param


# Initialize the ALL_MODELS dictionary with the models that are always available
ALL_MODELS = {x: Hookmodel for x in NN_MODELS}
ALL_MODELS.update({x: GaussianFeatures for x in GAUSSIAN_MODELS})
ALL_MODELS.update({x: DinoFeatures for x in DINO_MODELS})

# Try to import CellposeFeatures and update the ALL_MODELS dictionary if successful
# Cellpose is only installed with pip install napari-convpaint[cellpose]
try:
    from napari_convpaint.conv_paint_cellpose import AVAILABLE_MODELS as CELLPOSE_MODELS
    from napari_convpaint.conv_paint_cellpose import CellposeFeatures
    ALL_MODELS.update({x: CellposeFeatures for x in CELLPOSE_MODELS})
except ImportError:
    # Handle the case where CellposeFeatures or its dependencies are not available
    print("Cellpose is not installed and is not available as feature extractor.\n"
          "Run 'pip install napari-convpaint[cellpose]' to install it.")


def get_all_models():
    """Return a dictionary of all available models"""
    return ALL_MODELS


def load_model(model_path):
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    classifier = data['classifier']
    param = data['param']
    if 'model_state' in data:
        model_state = data['model_state']
    else:
        model_state = None
    model = create_model(param)
    return classifier, model, param, model_state

def save_model(model_path, classifier, model, param):
    with open(model_path, 'wb') as f:
        data = {
            'classifier': classifier,
            'param': param,
            'model_state': model.state_dict() if hasattr(model, 'state_dict') else None
        }
        pickle.dump(data, f)


def create_model(param:Param):
    """Create a model based on the given parameters."""
    model_class = get_all_models()[param.model_name]
    model = model_class(
        model_name=param.model_name,
        use_cuda=param.use_cuda
    )
    
    if isinstance(model, Hookmodel):
        if param.model_layers:
            model.register_hooks(selected_layers=param.model_layers)
        elif len(model.named_modules) == 1:
            model.register_hooks(selected_layers=[list(model.module_dict.keys())[0]])
    
    return model


def train_classifier(features, targets, iterations = 50, learning_rate = 0.1, depth = 5):
    """Train a classifier given a set of features and targets."""

    # train a random forest classififer
    #classifier = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    #classifier.fit(features, targets)

    #classifier = xgb.XGBClassifier(tree_method="hist", n_estimators=100, n_jobs=8)
    #classifier.fit(features, targets-1)

    classifier = CatBoostClassifier(iterations=iterations, learning_rate=learning_rate,depth=depth)
    classifier.fit(features, targets)

    return classifier


def get_features_current_layers(image, annotations, model, param:Param):
    """Given a potentially multidimensional image and a set of annotations,
    extract multiscale features from multiple layers of a model.
    Parameters
    ----------
    image : np.ndarray
        2D or 3D Image to extract features from. If 3D but annotations are 2D,
        image is treated as multichannel and not image series
    annotations : np.ndarray
        2D, 3D Annotations (1,2) to extract features from
    model : feature extraction model
        Model to extract features from the image
    scalings : list of ints
        Downsampling factors
    order : int, optional
        Interpolation order for low scale resizing, by default 0
    use_min_features : bool, optional
        Use minimal number of features, by default True
    image_downsample : int, optional
        Downsample image by this factor before extracting features, by default 1
    Returns
    -------
    features : pandas DataFrame
        Extracted features (rows are pixel, columns are features)
    targets : pandas Series
        Target values
    """

    if model is None:
        raise ValueError('Model must be provided')

    # get indices of first dimension of non-empty annotations. Gives t/z indices
    if annotations.ndim == 3:
        non_empty = np.unique(np.where(annotations > 0)[0])
        if len(non_empty) == 0:
            warnings.warn('No annotations found')
            return None, None
    elif annotations.ndim == 2:
        non_empty = [0]
    else:
        raise Exception('Annotations must be 2D or 3D')

    all_values = []
    all_targets = []

    # find maximal padding necessary
    padding = param.padding * np.max(param.scalings)

    # iterating over non_empty iteraties of t/z for 3D data
    for ind, t in enumerate(non_empty):

        if annotations.ndim == 2:
            if image.ndim == 3:
                current_image = np.pad(image, ((0,0), (padding,padding), (padding,padding)), mode='reflect')
            else:
                current_image = np.pad(image, padding, mode='reflect')
            current_annot = np.pad(annotations, padding, mode='constant')
        else:
            if image.ndim == 3:
                current_image = np.pad(image[t], padding, mode='reflect')
                current_annot = np.pad(annotations[t], padding, mode='constant')
            elif image.ndim == 4:
                current_image = np.pad(image[:, t], ((0,0),(padding, padding),(padding, padding)), mode='reflect')
                current_annot = np.pad(annotations[t], padding, mode='constant')

        annot_regions = skimage.morphology.label(current_annot > 0)

        if param.tile_annotations:
            boxes = skimage.measure.regionprops_table(annot_regions, properties=('label', 'bbox'))
        else:
            boxes = {'label': [1], 'bbox-0': [padding], 'bbox-1': [padding], 'bbox-2': [current_annot.shape[0]-padding], 'bbox-3': [current_annot.shape[1]-padding]}
        for i in range(len(boxes['label'])):
            # NOTE: This assumes that the image is already padded correctly, and the padded boxes cannot go out of bounds
            pad_size = param.padding
            x_min = boxes['bbox-0'][i]-pad_size
            x_max = boxes['bbox-2'][i]+pad_size
            y_min = boxes['bbox-1'][i]-pad_size
            y_max = boxes['bbox-3'][i]+pad_size

            # temporary check that bounds are not out of image
            x_min = max(0, x_min)
            x_max = min(current_image.shape[-2], x_max)
            y_min = max(0, y_min)
            y_max = min(current_image.shape[-1], y_max)

            im_crop = current_image[...,
                x_min:x_max,
                y_min:y_max
            ]
            annot_crop = current_annot[
                x_min:x_max,
                y_min:y_max
            ]
            extracted_features = model.get_features_scaled(image=im_crop,param=param)

            if param.image_downsample > 1:
                annot_crop = annot_crop[::param.image_downsample, :: param.image_downsample]

            #from the [features, w, h] make a list of [features] with len nb_annotations
            mask = annot_crop > 0
            nb_features = extracted_features.shape[0]

            extracted_features = np.moveaxis(extracted_features, 0, -1) #move [w,h,features]

            extracted_features = extracted_features[mask]
            all_values.append(extracted_features)

            targets = annot_crop[annot_crop > 0]
            targets = targets.flatten()
            all_targets.append(targets)


    all_values = np.concatenate(all_values, axis=0)
    features = pd.DataFrame(all_values)

    all_targets = np.concatenate(all_targets, axis=0)
    targets = pd.Series(all_targets)

    return features, targets
