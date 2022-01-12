from collections import OrderedDict

import torch
import numpy as np
import skimage.transform
import pandas as pd

import torchvision.models as models
from torch import nn

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

def predict_image(image, model, classifier):
    """
    Given a filter model and a classifier, predict the class of 
    each pixel in an image.

    Parameters
    ----------
    image: 2d array
        image to segment
    model: pytorch model
        first layer of VGG16
    classifier: sklearn classifier
        classifier to use for prediction

    Returns
    -------
    predicted_image: 2d array
        predicted image with classes

    """
    
    if model is None:
        model = load_nn_model()
    n_filters = 64

    scalings = [2**x for x in range(0, int(classifier.n_features_in_/n_filters))]
    all_scales = filter_image(image, model, scalings)
    all_pixels = pd.DataFrame(np.reshape(np.concatenate(all_scales, axis=1), (len(scalings)*n_filters, image.shape[0]*image.shape[1])).T)
    predictions = classifier.predict(all_pixels)

    predicted_image = np.reshape(predictions, image.shape)
    return predicted_image

def load_nn_model():
    """Load VGG16 model from torchvision, keep first layer only
    
    Parameters
    ----------

    Returns
    -------
    model: torch model
        model for pixel feature extraction
    
    """

    vgg16 = models.vgg16(pretrained=True)
    #model = nn.Sequential(vgg16.features[0])

    model = nn.Sequential(OrderedDict([('conv1', nn.Conv2d(1,64,3,1,1))]))

    pretrained_dict = vgg16.state_dict()
    reduced_dict = model.state_dict()

    reduced_dict['conv1.weight'][:,0,:,:] = pretrained_dict['features.0.weight'][:,:,:,:].sum(axis=1)
    reduced_dict['conv1.bias'] = pretrained_dict['features.0.bias']

    model.load_state_dict(reduced_dict)
    
    return model