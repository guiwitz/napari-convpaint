from collections import OrderedDict
from typing import Any
import warnings

import torch
import numpy as np
import skimage.transform
import pandas as pd

import torchvision.models as models
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def filter_image(image, model, scalings):
    """
    Filter an image with the first layer of a VGG16 model.
    Apply filter to each scale in scalings.

    Parameters
    ----------
    image: 2d array
        image to filter
    model: pytorch model
        first layer of VGG16
    scalings: list of ints
        downsampling factors

    Returns
    -------
    all_scales: list of 2d array
        list of filtered images. For each downsampling factor,
        there are N images, N being the number of filters of the model.

    """

    n_filters = 64
    # convert to np if dasks array
    image = np.asarray(image)
    
    all_scales=[]
    for s in scalings:
        im_tot = image[::s,::s].astype(np.float32)
        #im_tot = np.ones((3,im_tot.shape[0],im_tot.shape[1]), dtype=np.float32) * im_tot
        im_torch = torch.from_numpy(im_tot[np.newaxis, np.newaxis, ::])
        out = model.forward(im_torch)
        out_np = out.detach().numpy()
        if s > 1:
            out_np = skimage.transform.resize(
                out_np, (1, n_filters, image.shape[0],image.shape[1]), preserve_range=True)
        all_scales.append(out_np)
    return all_scales

def filter_image_multioutputs(image, hookmodel, scalings=[1]):
    """Recover the outputs of chosen layers of a pytorch model. Layers and model are
    specified in the hookmodel object.
    
    Parameters
    ----------
    image : np.ndarray
        2d Image to filter
    hookmodel : Hookmodel
        Model to extract features from
    scalings : list of ints, optional
        Downsampling factors, by default None

    Returns
    -------
    all_scales : list of np.ndarray
        List of filtered images. The number of image is the sum over 
        selected output layers of NxS where N is the number of filters
        of the layer and S the number of scaling factors.
        
    """
    
    input_channels = hookmodel.named_modules[0][1].in_channels
    image = np.asarray(image, dtype=np.float32)
    image = np.ones((input_channels, image.shape[0],image.shape[1]), dtype=np.float32) * image
    
    all_scales  = []
    for s in scalings:
        im_tot = image[:, ::s, ::s]
        im_torch = torch.from_numpy(im_tot[np.newaxis, ::])
        hookmodel.outputs = []
        try:
            _ = hookmodel(im_torch)
        except:
            pass
        
        for im in hookmodel.outputs:
            out_np = im.detach().numpy()
            if image.shape[1:3] != out_np.shape[2:4]:
                out_np = skimage.transform.resize(
                    out_np, (1, out_np.shape[1], image.shape[1],image.shape[2]), preserve_range=True)
            all_scales.append(out_np)
    
    return all_scales



def predict_image(image, model, classifier, scalings=[1]):
    """
    Given a filter model and a classifier, predict the class of 
    each pixel in an image.

    Parameters
    ----------
    image: 2d array
        image to segment
    model: Hookmodel
        model to extract features from
    classifier: sklearn classifier
        classifier to use for prediction
    scalings: list of ints
        downsampling factors

    Returns
    -------
    predicted_image: 2d array
        predicted image with classes

    """

    all_scales = filter_image_multioutputs(image, model, scalings=scalings)
    tot_filters = np.sum(model.features_per_layer) * len(all_scales) / len(model.features_per_layer)
    tot_filters = int(tot_filters)
    all_pixels = pd.DataFrame(np.reshape(np.concatenate(all_scales, axis=1), (tot_filters, image.shape[0]*image.shape[1])).T)
        
    predictions = classifier.predict(all_pixels)

    predicted_image = np.reshape(predictions, image.shape)
    return predicted_image

def load_single_layer_vgg16():
    """Load VGG16 model from torchvision, keep first layer only
    
    Parameters
    ----------

    Returns
    -------
    model: torch model
        model for pixel feature extraction
    
    """

    vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    model = nn.Sequential(OrderedDict([('conv1', nn.Conv2d(1,64,3,1,1))]))

    pretrained_dict = vgg16.state_dict()
    reduced_dict = model.state_dict()

    reduced_dict['conv1.weight'][:,0,:,:] = pretrained_dict['features.0.weight'][:,:,:,:].sum(axis=1)
    reduced_dict['conv1.bias'] = pretrained_dict['features.0.bias']

    model.load_state_dict(reduced_dict)
    
    return model

def get_multiscale_features(model, image, annot, scalings):
    """Given an image and a set of annotations, extract multiscale features
    
    Parameters
    ----------
    model : Hookmodel
        Model to extract features from
    image : np.ndarray
        Image to extract features from (currently 2D only)
    annot : np.ndarray
        Annotations (1,2) to extract features from (currently 2D only)

    Returns
    -------
    extracted_features : np.ndarray
        Extracted features. Dimensions npixels x nfeatures * nbscales
    """

    # get number of features
    
    max_features = np.max(model.features_per_layer)
    full_annotation = np.ones((max_features, image.shape[0], image.shape[1]),dtype=np.bool_)

    full_annotation = full_annotation * annot > 0

    all_scales = filter_image_multioutputs(image, model, scalings)
    all_values_scales=[]
    for ind, a in enumerate(all_scales):
        
        n_features = a.shape[1]
        extract = a[0, full_annotation[0:n_features]]
    
        all_values_scales.append(np.reshape(extract, (n_features, int(extract.shape[0]/n_features))).T)
    extracted_features = np.concatenate(all_values_scales, axis=1)
    
    return extracted_features
    
def get_features_current_layers(model, image, annotations, scalings=[1]):
    """Get a current set of features, targets from the currently selected
    files."""

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
    # iterating over non_empty iteraties of t/z for 3D data
    for ind, t in enumerate(non_empty):

        if image.ndim == 2:
            current_image = image
            current_annot = annotations
        else:
            current_image = image[t]
            current_annot = annotations[t]
            
        extracted_features = get_multiscale_features(model, current_image, current_annot, scalings)
        all_values.append(extracted_features)

    all_values = np.concatenate(all_values,axis=0)
    features = pd.DataFrame(all_values)
    target_im = current_annot[annotations>0]
    targets = pd.Series(target_im)

    return features, targets

def train_classifier(features, targets):
    """Train a random forest classifier given a set of features and targets."""

    # train model
    #split train/test
    X, X_test, y, y_test = train_test_split(features, targets, 
                                        test_size = 0.2, 
                                        random_state = 42)

    #train a random forest classififer
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X, y)

    return random_forest


# subclass of pytorch model that includes hooks and outputs for certain layers
class Hookmodel():
    """Class to extract features from a pytorch model using hooks on chosen layers.

    Parameters
    ----------
    model_name : str
        Name of model to use. Currently only 'vgg16' and 'single_layer_vgg16' are supported.

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
    
    def __init__(self, model_name='vgg16'):

        if model_name == 'vgg16':
            self.model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        elif model_name == 'efficient_netb0':
            self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        elif model_name == 'single_layer_vgg16':
            self.model = load_single_layer_vgg16()
        
        self.outputs = []
        self.features_per_layer = []

        self.get_layer_dict()

        if model_name == 'single_layer_vgg16':
            self.register_hooks(
                [x[1] for x in self.module_list], [x[2] for x in self.module_list])

    def __call__(self, tensor_image):
        
        return self.model(tensor_image)
        
    def hook_normal(self, module, input, output):
        self.outputs.append(output)

    def hook_last(self, module, input, output):
        self.outputs.append(output)
        assert False

    def register_hooks(self, selected_layers):#, selected_layer_pos):

        self.features_per_layer = []
        for ind in range(len(selected_layers)):

            self.features_per_layer.append(
                self.module_dict[selected_layers[ind]].out_channels)
            
            if ind == len(selected_layers)-1:
                self.module_dict[selected_layers[ind]].register_forward_hook(self.hook_last)
            else:
                self.module_dict[selected_layers[ind]].register_forward_hook(self.hook_normal)


    def get_layer_dict(self):
        """Create a flat list of all modules as well as a dictionary of modules with 
        keys describing the layers."""
        
        named_modules = list(self.model.named_modules())
        self.named_modules = [n for n in named_modules if len(list(n[1].named_modules())) == 1]
        self.module_id_dict = dict([(x[0] + ' ' + x[1].__str__(), x[0]) for x in self.named_modules])
        self.module_dict = dict([(x[0] + ' ' + x[1].__str__(), x[1]) for x in self.named_modules])