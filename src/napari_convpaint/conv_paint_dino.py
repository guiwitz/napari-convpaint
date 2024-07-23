import torch
import numpy as np
from .conv_paint_utils import get_device

AVAILABLE_MODELS = ['dinov2_vits14_reg']

class DinoFeatures():
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
        '''Input image is 3xHxW, normalize to image net stats,c return to 1xCxHxW tensor'''
        #for uint8 or uint16 images, get divide by max value
        print(image.dtype)
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
        '''Extract features from image with arbitrary number of color channels, return features as np.array with dimensions  H x W x nfeatures'''
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)
            image = np.repeat(image, 3, axis=0)
            features = self._extract_features_rgb(image)
        elif image.shape[0] == 3:
            features = self._extract_features_rgb(image)
        else:
            features = []
            for i in range(image.shape[0]):
                features_rgb = self._extract_features_rgb(image[i])
                features.append(features_rgb)
            features = np.concatenate(features, axis=-1)
        return features


    def get_features(self, image, annotations, **kwargs):
        #make sure image is divisible by patch size
        h, w = image.shape[-2:]
        h, w = image.shape[-2:]
        new_h = (h // self.patchsize) * self.patchsize
        new_w = (w // self.patchsize) * self.patchsize
        h_pad_top = (h - new_h)//2
        w_pad_left = (w - new_w)//2
        h_pad_bottom = h - new_h - h_pad_top
        w_pad_right = w - new_w - w_pad_left
        image = image[:, h_pad_top:-h_pad_bottom, w_pad_left:-w_pad_right]

        features = self._extract_features(image)
        features = np.repeat(features, self.patchsize, axis=0)
        features = np.repeat(features, self.patchsize, axis=1)

        #add 0s padding to match original image size, with the dimension of features 
        # (since we cropped to be divisible by patch size
        features_padded = np.zeros((h, w, features.shape[-1]), dtype=np.float32)
        features_padded[h_pad_top:-h_pad_bottom, w_pad_left:-w_pad_right,:] = features
        features = features_padded

        #extract features where there are annotations, return np.array with npixels x nfeatures
        mask = annotations > 0
        features = features[mask == 1]
        features = np.reshape(features, (features.shape[0], -1))
        return features

    def predict_image(self, image, classifier, **kwargs):
        features = self.get_features(image, np.ones(image.shape[-2::], dtype=np.uint8))
        predictions = classifier.predict(features)
        predictions = np.reshape(predictions, image.shape[-2::])
        return predictions
    
