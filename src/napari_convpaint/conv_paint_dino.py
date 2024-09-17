import torch
import numpy as np
from .conv_paint_utils import get_device
from .conv_paint_feature_extractor import FeatureExtractor
import skimage
import pandas as pd

AVAILABLE_MODELS = ['dinov2_vits14_reg']

class DinoFeatures(FeatureExtractor):
    def __init__(self, model_name='dinov2_vits14_reg', use_cuda=False):
        self.model_name = model_name
        self.use_cuda = use_cuda
        # prevent rate limit error on GitHub Actions: https://github.com/pytorch/pytorch/issues/61755
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        try:
            self.model = torch.hub.load('facebookresearch/dinov2', self.model_name, pretrained=True, verbose=False)
        except(RuntimeError):
            self.model = torch.hub.load('facebookresearch/dinov2', self.model_name, pretrained=True, verbose=False, force_reload=True)
        self.patchsize = 14
        self.padding = int(self.patchsize/2)
        if self.use_cuda:
            self.device = get_device()
            self.model.to(self.device)
        else:
            self.device = 'cpu'
        self.model.eval()

    def _preprocess_image(self, image):
        '''Normalizes input image to image net stats, return to 1x3xHxW tensor.
        Expects image to be 3xHxW'''

        assert len(image.shape) == 3
        assert image.shape[0] == 3

        # for uint8 or uint16 images, get divide by max value
        if image.dtype == np.uint8:
            image = image.astype(np.float32)
            image = image / 255
        elif image.dtype == np.uint16:
            image = image.astype(np.float32)
            image = image / 65535
        # else just min max normalize to 0-1.
        else:
            image = image.astype(np.float32)
            image = (image - np.min(image)) / (np.max(image) - np.min(image))

        # normalize to imagenet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean[:, None, None]) / std[:, None, None]

#       # make sure image is divisible by patch size
        h, w = image.shape[-2:]
        new_h = h - (h % self.patchsize)
        new_w = w - (w % self.patchsize)
        if new_h != h or new_w != w:
            image = image[:, :new_h, :new_w]

        # add batch dimension
        image = np.expand_dims(image, axis=0)
    
        # convert to tensor
        image_tensor = torch.tensor(image, dtype=torch.float32,device=self.device)
        return image_tensor

    def _extract_features_rgb(self, image):
        '''Extract features from image, return features as np.array with dimensions  H x W x nfeatures.
        Input image has to be multiple of patch size'''
        assert image.shape[-2] % self.patchsize == 0
        assert image.shape[-1] % self.patchsize == 0
        assert image.shape[0] == 3

        image_tensor = self._preprocess_image(image)
        with torch.no_grad():
            features_dict = self.model.forward_features(image_tensor)
        features = features_dict['x_norm_patchtokens']
        if self.use_cuda:
            features = features.cpu()
        features = features.numpy()[0]
        features_shape = (int(image.shape[-2] / self.patchsize), int(image.shape[-1] / self.patchsize), features.shape[-1])
        features = np.reshape(features, features_shape)

        assert features.shape[0] == image.shape[-2] / self.patchsize
        assert features.shape[1] == image.shape[-1] / self.patchsize
        return features
    
    def _extract_features(self, image):
        '''Helper function to extract features from image with arbitrary number of color channels, 
        return features as np.array with dimensions  H x W x nfeatures'''
        assert len(image.shape) == 3
        if image.shape[0] == 3:
            features = self._extract_features_rgb(image)
        else:
            features = []
            for channel_nb in range(image.shape[0]):
                channel = np.expand_dims(image[channel_nb], axis=0)
                channel = np.repeat(channel, 3, axis=0)
                features_rgb = self._extract_features_rgb(channel)
                features.append(features_rgb)
            features = np.concatenate(features, axis=-1)
        return features

    def get_padding(self):
        return self.padding
    
    def get_features_scaled(self, image, order=0, image_downsample=1, return_patches = False, **kwargs):
        """
        Overwrite the get_features_scaled function, as we don't want to extract features at different scales for DINO.

        Parameters
        ----------
        image: 2d array
            image to segment
        order: int
            interpolation order for low scale resizing
        image_downsample: int, optional
            downsample image by this factor before extracting features, by default 1

        Returns
        -------
        features: [nb_features x width x height]
            return extracted features

        """

        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)

        if image_downsample > 1:
            image = image[:, ::image_downsample, ::image_downsample]
        
        features = self.get_features(image, order=order, return_patches= return_patches, **kwargs) #features have shape [nb_features, width, height]
        nb_features = features.shape[0]

        if not return_patches:
            features = skimage.transform.resize(
                                image=features,
                                output_shape=(nb_features, image.shape[-2], image.shape[-1]),
                                preserve_range=True,
                                order=order)            
        return features
    
    def predict_image(self, image, classifier, scalings = [1], order=0, image_downsample=1, **kwargs):
        features = self.get_features_scaled(image=image,
                                            scalings=scalings,
                                            order=order,
                                            image_downsample=image_downsample,
                                            return_patches=True)
        nb_features = features.shape[0] #[nb_features, width, height]

        #move features to last dimension
        features = np.moveaxis(features, 0, -1)
        features = np.reshape(features, (-1, nb_features))

        rows = np.ceil(image.shape[-2] / image_downsample).astype(int)
        cols = np.ceil(image.shape[-1] / image_downsample).astype(int)

        all_pixels = pd.DataFrame(features)
        predictions = classifier.predict(all_pixels)

        predicted_image = np.reshape(predictions, [rows, cols])
        if image_downsample > 1:
            predicted_image = skimage.transform.resize(
                image=predicted_image,
                output_shape=(image.shape[-2], image.shape[-1]),
                preserve_range=True, order=order).astype(np.uint8)
        predicted_image = predicted_image +1#for XGBoost
        return predicted_image

    def get_features(self, image, return_patches=False, **kwargs):
        '''Given an CxWxH image, extract features.
        Returns features with dimensions nb_features x H x W'''

        #make sure image is divisible by patch size
        h, w = image.shape[-2:]
        new_h = (h // self.patchsize) * self.patchsize
        new_w = (w // self.patchsize) * self.patchsize
        h_pad_top = (h - new_h)//2
        w_pad_left = (w - new_w)//2
        h_pad_bottom = h - new_h - h_pad_top
        w_pad_right = w - new_w - w_pad_left

        if h_pad_top > 0 or w_pad_left > 0 or h_pad_bottom > 0 or w_pad_right > 0:
            image = image[:, h_pad_top:-h_pad_bottom if h_pad_bottom != 0 else None, 
                             w_pad_left:-w_pad_right if w_pad_right != 0 else None]
#

        features = self._extract_features(image) #[H, W, nfeatures]

        if not return_patches:
            #upsample features to original size
            features = np.repeat(features, self.patchsize, axis=0)
            features = np.repeat(features, self.patchsize, axis=1)

            #replace with padding where there are no annotations
            pad_width = ((h_pad_top, h_pad_bottom), (w_pad_left, w_pad_right), (0,0))

            features = np.pad(features, pad_width=pad_width, mode= 'edge')
            features = np.moveaxis(features, -1, 0) #[nb_features, H, W]

            assert features.shape[1] == h and features.shape[2] == w
            return features
        
        else:
            #return patches
            features = np.moveaxis(features, -1, 0) #[nb_features, H, W]
            assert features.shape[1] == (h // self.patchsize) and features.shape[2] == (w // self.patchsize)
            return features


    def predict_image(self, image, classifier, image_downsample=1, order=0, **kwargs):
        '''Special predict function for feature extraction models that output patchwise features.
        1. Extract patch wise features
        2. Predict each patch with classifier
        3. Scale up the predictions'''

        #add padding
        padding = self.get_padding()
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)
        image = np.pad(image, ((0, 0), (padding, padding), (padding, padding)), mode='reflect')

        features = self.get_features_scaled(image, return_patches=True, **kwargs)
        w_patch = np.ceil(features.shape[-2] / image_downsample).astype(int)
        h_path = np.ceil(features.shape[-1] / image_downsample).astype(int)

        nb_features = features.shape[0] #[nb_features, width, height]

        #move features to last dimension
        features = np.moveaxis(features, 0, -1) #[width, height, nb_features]
        features = np.reshape(features, (-1, nb_features)) #[width*height, nb_features] (1d-array for RF)


        all_pixels = pd.DataFrame(features)
        predictions = classifier.predict(all_pixels)

        predicted_image = np.reshape(predictions, [w_patch, h_path])
        predicted_image = skimage.transform.resize(
            image=predicted_image,
            output_shape=(image.shape[-2], image.shape[-1]),
            preserve_range=True, order=order).astype(np.uint8)
        
        #remove padding
        predicted_image = predicted_image[padding:-padding, padding:-padding]
        
        return predicted_image
