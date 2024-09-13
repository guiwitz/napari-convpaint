from collections import OrderedDict

import numpy as np
import pandas as pd
import skimage
import torch
from torch import nn
from torch.nn.functional import interpolate as torch_interpolate
from torchvision import models
from .conv_paint_utils import get_device
from .conv_paint_feature_extractor import FeatureExtractor


AVAILABLE_MODELS = ['vgg16', 'efficient_netb0', 'single_layer_vgg16']

class Hookmodel(FeatureExtractor):
    """Class to extract features from a pytorch model using hooks on chosen layers.

    Parameters
    ----------
    model_name : str
        Name of model to use. Currently only 'vgg16' and 'single_layer_vgg16',
         'single_layer_vgg16_rgb' are supported.
    model : torch model, optional
        Model to extract features from, by default None
    use_cuda : bool, optional
        Use cuda, by default False
    param : Param, optional
        Parameters for model, by default None
        
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

    def __init__(self, model_name='vgg16', model=None, use_cuda=None, param=None):

        if model is not None:
            assert model_name is None, 'model_name must be None if model is not None'
            self.model = model
        else:
            if param is not None:
                model_name = param.model_name

            if model_name == 'vgg16':
                self.model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
                # self.transform =  models.VGG16_Weights.IMAGENET1K_V1.transforms()
            elif model_name == 'efficient_netb0':
                self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            elif model_name == 'single_layer_vgg16':
                self.model = self.load_single_layer_vgg16(keep_rgb=False)
            elif model_name == 'single_layer_vgg16_rgb':
                self.model = self.load_single_layer_vgg16(keep_rgb=True)
            elif model_name == 'dino_vits16':
                self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')

        self.device = get_device(use_cuda)
        self.model = self.model.to(self.device)
    
        self.outputs = []
        self.features_per_layer = []
        self.selected_layers = []
        self.selectable_layer_keys = []

        self.get_layer_dict()
        if model_name == 'single_layer_vgg16':
            self.register_hooks(list(self.module_dict.keys()))

        if (param is not None) and (model_name != 'single_layer_vgg16'):
            self.register_hooks(param.model_layers)

    def __call__(self, tensor_image):
        tensor_image_dev = tensor_image.to(self.device)
        return self.model(tensor_image_dev)

    def hook_normal(self, module, input, output):
        self.outputs.append(output)

    def hook_last(self, module, input, output):
        self.outputs.append(output)
        assert False

    def register_hooks(self, selected_layers):  # , selected_layer_pos):

        self.features_per_layer = []
        self.selected_layers = selected_layers.copy()
        for ind in range(len(selected_layers)):
            self.features_per_layer.append(
                self.module_dict[selected_layers[ind]].out_channels)

            if ind == len(selected_layers) - 1:
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
        
        self.selectable_layer_keys = dict([(x[0] + ' ' + x[1].__str__(), x[1]) for x in self.named_modules if isinstance(x[1], nn.Conv2d)])


    def load_single_layer_vgg16(self, keep_rgb=False):
        """Load VGG16 model from torchvision, keep first layer only
        
        Parameters
        ----------

        Returns
        -------
        model: torch model
            model for pixel feature extraction
        keep_rgb: bool
            if True, keep model with three input channels, otherwise convert to single channel
        
        """

        vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        pretrained_dict = vgg16.state_dict()
        
        if keep_rgb:
            model = nn.Sequential(OrderedDict([('conv1', nn.Conv2d(3, 64, 3, 1, 1))]))
        else:
            model = nn.Sequential(OrderedDict([('conv1', nn.Conv2d(1, 64, 3, 1, 1))]))

        reduced_dict = model.state_dict()

        if keep_rgb:
            reduced_dict['conv1.weight'] = pretrained_dict['features.0.weight']
            reduced_dict['conv1.bias'] = pretrained_dict['features.0.bias']
        else:
            reduced_dict['conv1.weight'][:, 0, :, :] = pretrained_dict['features.0.weight'][:, :, :, :].sum(axis=1)
            reduced_dict['conv1.bias'] = pretrained_dict['features.0.bias']

        model.load_state_dict(reduced_dict)

        return model

    def get_features_scaled(self, image, scalings=[1], order=0,
                     use_min_features=True, image_downsample=1):
        """Given an image and a set of annotations, extract multiscale features
        
        Parameters
        ----------
        model : Hookmodel
            Model to extract features from
        image : np.ndarray
            Image to extract features from (currently 2D only or 2D multichannel (C,H,W))
        annotations : np.ndarray
            Annotations (1,2) to extract features from (currently 2D only)
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
        extracted_features : np.ndarray
            Extracted features. Dimensions (nfeatures * nbscales) x W x H 
        """

        if use_min_features:
            max_features = np.min(self.features_per_layer)
        else:
            max_features = np.max(self.features_per_layer)
        # test with minimal number of features i.e. taking only n first features
        rows = np.ceil(image.shape[-2] / image_downsample).astype(int)
        cols = np.ceil(image.shape[-1] / image_downsample).astype(int)

        all_scales = self.filter_image_multichannels(
            image, scalings, order=order,
            image_downsample=image_downsample)
        if use_min_features:
            all_scales = [a[:, 0:max_features, :, :] for a in all_scales]
        all_values_scales = []

        for ind, a in enumerate(all_scales):
            n_features = a.shape[1]
            extract = a[0]
            all_values_scales.append(extract)
        extracted_features = np.concatenate(all_values_scales, axis=0)
        return extracted_features
    
    def predict_image(self, image, classifier, scalings=[1], order=0,
                  use_min_features=True, image_downsample=1,
                  ):
        """
        Given a filter model and a classifier, predict the class of 
        each pixel in an image.

        Parameters
        ----------
        image: 2d array
            image to segment
        classifier: sklearn classifier
            classifier to use for prediction
        scalings: list of ints
            downsampling factors
        order: int
            interpolation order for low scale resizing
        use_min_features: bool
            if True, use the minimum number of features per layer
        image_downsample: int, optional
            downsample image by this factor before extracting features, by default 1

        Returns
        -------
        predicted_image: 2d array
            predicted image with classes

        """
        
        padding = self.get_padding() * np.max(scalings)
        if image.ndim == 2:
            image = np.pad(image, ((padding, padding), (padding, padding)), mode='reflect')
        elif image.ndim == 3:
            image = np.pad(image, ((0, 0), (padding, padding), (padding, padding)), mode='reflect')

        if use_min_features:
            max_features = np.min(self.features_per_layer)
            all_scales = self.filter_image_multichannels(
                image, scalings=scalings, order=order,
                image_downsample=image_downsample)
            all_scales = [a[:, 0:max_features, :, :] for a in all_scales]
            tot_filters = max_features * len(all_scales)

        else:
            # max_features = np.max(model.features_per_layer)
            all_scales = self.filter_image_multichannels(
                image, scalings=scalings, order=order,
                image_downsample=image_downsample)
            tot_filters = np.sum(a.shape[1] for a in all_scales)
        
        tot_filters = int(tot_filters)
        rows = np.ceil(image.shape[-2] / image_downsample).astype(int)
        cols = np.ceil(image.shape[-1] / image_downsample).astype(int)
        all_pixels = pd.DataFrame(
            np.reshape(np.concatenate(all_scales, axis=1), newshape=(tot_filters, rows * cols)).T)

        predictions = classifier.predict(all_pixels)

        predicted_image = np.reshape(predictions, [rows, cols])
        if image_downsample > 1:
            predicted_image = skimage.transform.resize(
                image=predicted_image,
                output_shape=(image.shape[-2], image.shape[-1]),
                preserve_range=True, order=1).astype(np.uint8)
            
        predicted_image = predicted_image[padding:-padding, padding:-padding]
        

        return predicted_image


    def filter_image_multichannels(self, image, scalings=[1], order=0,
                                image_downsample=1):
        """Recover the outputs of chosen layers of a pytorch model. Layers and model are
        specified in the hookmodel object. If image has multiple channels, each channel
        is processed separately.
        
        Parameters
        ----------
        image : np.ndarray
            2d Image to filter
        scalings : list of ints, optional
            Downsampling factors, by default None
        order : int, optional
            Interpolation order for low scale resizing,
            by default 0
        image_downsample : int, optional
            Downsample image by this factor before extracting features, by default 1

        Returns
        -------
        all_scales : list of np.ndarray
            List of filtered images. The number of images is C x Sum_i(F_i x S) where C is the number of channels,
            F_i is the number of filters of the ith layer and S the number of scaling factors.
            
        """
        input_channels = self.named_modules[0][1].in_channels
        image = np.asarray(image, dtype=np.float32)
        
        if image.ndim == 2:
            image = image[::image_downsample, ::image_downsample]
            image = np.ones((input_channels, image.shape[0], image.shape[1]), dtype=np.float32) * image
            image_series = [image]
        elif image.ndim == 3:
            image = image[:, ::image_downsample, ::image_downsample]
            image_series = [np.ones((input_channels, im.shape[0], im.shape[1]), dtype=np.float32) * im for im in image]

        int_mode = 'bilinear' if order > 0 else 'nearest'
        align_corners = False if order > 0 else None

        all_scales = []
        with torch.no_grad():
            for image in image_series:
                for s in scalings:
                    im_tot = image[:, ::s, ::s]
                    im_torch = torch.tensor(im_tot[np.newaxis, ::])
                    self.outputs = []
                    try:
                        _ = self(im_torch)
                    except AssertionError as ea:
                        pass
                    except Exception as ex:
                        raise ex

                    for im in self.outputs:
                        if image.shape[1:3] != im.shape[2:4]:
                            im = torch_interpolate(im, size=image.shape[1:3], mode=int_mode, align_corners=align_corners)
                            '''out_np = skimage.transform.resize(
                                image=out_np,
                                output_shape=(1, out_np.shape[1], image.shape[1], image.shape[2]),
                                preserve_range=True, order=order)'''

                        out_np = im.cpu().detach().numpy()
                        all_scales.append(out_np)
        return all_scales
    
    def get_padding(self):
        return self.get_max_kernel_size() // 2
    
    def get_max_kernel_size(self):
        """
        Given a hookmodel, find the maximum kernel size needed for the deepest layer.
        
        Parameters
        ----------
        
        Returns
        -------
        max_kernel_size: int
            maximum kernel size needed for the deepest layer
        """
        # Initialize variables
        max_kernel_size = 1
        current_total_pool = 1
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