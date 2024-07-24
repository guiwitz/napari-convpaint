import torch
import numpy as np
from .conv_paint_utils import get_device
from .conv_paint_feature_extractor import FeatureExtractor

AVAILABLE_MODELS = ['dinov2_vits14_reg']

class DinoFeatures(FeatureExtractor):
    def __init__(self, model_name='dinov2_vits14_reg', use_cuda=False):
        self.model_name = model_name
        self.use_cuda = use_cuda
        try:
            self.model = torch.hub.load('facebookresearch/dinov2', self.model_name, pretrained=True, verbose=False)
        except(RuntimeError):
            self.model = torch.hub.load('facebookresearch/dinov2', self.model_name, pretrained=True, verbose=False, force_reload=True)
        self.patchsize = 14
        self.padding = self.patchsize
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

        #for uint8 or uint16 images, get divide by max value
        if image.dtype == np.uint8:
            image = image.astype(np.float32)
            image = image / 255
        elif image.dtype == np.uint16:
            image = image.astype(np.float32)
            image = image / 65535
        #else just min max normalize to 0-1
        else:
            image = (image - np.min(image)) / (np.max(image) - np.min(image))
        #normalize to imagenet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean[:, None, None]) / std[:, None, None]
#       # make sure image is divisible by patch size
        h, w = image.shape[-2:]
        new_h = (h // self.patchsize) * self.patchsize
        new_w = (w // self.patchsize) * self.patchsize
        image = image[:, :new_h, :new_w]    

        #add batch dimension
        image = np.expand_dims(image, axis=0)
    
        #convert to tensor
        image = torch.tensor(image, dtype=torch.float32,device=self.device)
        return image

    def get_padding(self):
        return self.padding
    
    def _extract_features_rgb(self, image):
        '''Extract features from image, return features as np.array with dimensions  H x W x nfeatures.
        Input image has to be multiple of patch size'''
        assert image.shape[-2] % self.patchsize == 0
        assert image.shape[-1] % self.patchsize == 0
        assert image.shape[0] == 3

        image_preprocessed = self._preprocess_image(image)
        with torch.no_grad():
            features_dict = self.model.forward_features(image_preprocessed)
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
        '''Extract features from image with arbitrary number of color channels, 
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

    def get_features(self, image, **kwargs):
        '''Given an image, extract features.
        Returns features with dimensions nb_features x H x W'''

        #make sure image is divisible by patch size
        h, w = image.shape[-2:]
        new_h = (h // self.patchsize) * self.patchsize
        new_w = (w // self.patchsize) * self.patchsize
        h_pad_top = (h - new_h)//2
        w_pad_left = (w - new_w)//2
        h_pad_bottom = h - new_h - h_pad_top
        w_pad_right = w - new_w - w_pad_left
        image = image[:, h_pad_top:-h_pad_bottom, w_pad_left:-w_pad_right]

        features = self._extract_features(image) #[H, W, nfeatures]
        features = np.repeat(features, self.patchsize, axis=0)
        features = np.repeat(features, self.patchsize, axis=1)

        #replace with padding where there are no annotations
        pad_width = ((h_pad_top, h_pad_bottom), (w_pad_left, w_pad_right), (0,0))
        features = np.pad(features, pad_width=pad_width, mode= 'edge')
        features = np.moveaxis(features, -1, 0)
        #print(features.shape) --> e.g. (384, 150, 95) 
        return features
