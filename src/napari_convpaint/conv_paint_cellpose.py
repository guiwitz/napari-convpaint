import torch
import numpy as np
from .conv_paint_utils import get_device
from .conv_paint_feature_extractor import FeatureExtractor
import skimage
import pandas as pd
from cellpose import models
import warnings

AVAILABLE_MODELS = ['cellpose_backbone']


class CellposeFeatures(FeatureExtractor):

    def __init__(self, model_name='cellpose_backbone', model=None, use_cuda=False):

        super().__init__(model_name=model_name, model=model, use_cuda=use_cuda)
        self.patch_size = 8
        self.padding = self.patch_size

    @staticmethod
    def create_model(model_name):
        # Load the cellpose model
        model_cellpose = models.CellposeModel(model_type='tissuenet_cp3')
        return model_cellpose.net
    
    def get_description(self):
        return "Model specialized in cell segmentation."

    def get_default_params(self, param=None):
        param = super().get_default_params(param=param)
        param.fe_name = self.model_name
        param.fe_use_cuda = self.use_cuda
        param.fe_layers = []
        param.fe_scalings = [1]
        param.fe_order = 0
        param.tile_annotations = False
        return param

    def get_features(self, img, **kwargs):
        """
        Extracts features from the image using the Cellpose model.
        """
        if img.ndim > 2:
            features = self.get_features_multichannel(img, **kwargs)
        else:
            features = self.extract_features(img, **kwargs)

        return features
        
    
    def extract_features(self, image, **kwargs):
        """
        Extracts features from the image using the Cellpose model.
        Assumes that the image is single or two channels.
        """
        img_expanded = CellposeFeatures.expand_img(image)

        # make sure image is divisible by patch size
        h, w = img_expanded.shape[-2:]
        new_h = (h // self.patch_size) * self.patch_size
        new_w = (w // self.patch_size) * self.patch_size
        h_pad_top = (h - new_h)//2
        w_pad_left = (w - new_w)//2
        h_pad_bottom = h - new_h - h_pad_top
        w_pad_right = w - new_w - w_pad_left

        if h_pad_top > 0 or w_pad_left > 0 or h_pad_bottom > 0 or w_pad_right > 0:
            img_expanded = img_expanded[:, h_pad_top:-h_pad_bottom if h_pad_bottom != 0 else None,
                                           w_pad_left:-w_pad_right if w_pad_right != 0 else None]

        # convert to tensor
        img_expanded = np.expand_dims(img_expanded, axis=0)
        tensor = torch.from_numpy(img_expanded).float()

        if self.model.mkldnn:
            tensor = tensor.to_mkldnn()
        T0 = self.model.downsample(tensor)
        if self.model.mkldnn:
            style = self.model.make_style(T0[-1].to_dense())
        else:
            style = self.model.make_style(T0[-1])
        style0 = style
        if not self.model.style_on:
            style = style * 0
        T1 = self.model.upsample(style, T0, self.model.mkldnn)
        T1 = self.model.output(T1)
        if self.model.mkldnn:
            T0 = [t0.to_dense() for t0 in T0]
            T1 = T1.to_dense()

        w_img,h_img = image.shape[-2:]

        out_t = []
        #append the output tensors from T0
        for t in T0[:3]:
            t = t.detach().numpy()[0]
            f,w,h = t.shape[-3:]
            if (w,h != w_img,h_img):
                t = skimage.transform.resize(
                        image=t,
                        output_shape=(f, w_img, h_img),
                        preserve_range=True, order=0)
            out_t.append(t)

        #append the output tensor from T1 (gradients and cell probability)
        t =T1.detach().numpy()[0]
        f,w,h = t.shape[-3:]
        if (w,h != w_img,h_img):
            t = skimage.transform.resize(
                    image=t,
                    output_shape=(f, w_img, h_img),
                    preserve_range=True, order=0)
        out_t.append(t)

        #append the original image
        if image.ndim == 2:
            out_t.append(np.expand_dims(image, axis=0))
        elif image.ndim == 3:
            out_t.append(image)
        out_t = np.concatenate(out_t,axis=0)
        return out_t
    
    def get_features_multichannel(self, image, **kwargs):
        """
        Feature Extraction with Cellpose for multichannel images.
        Assumes that the image is in the shape (C, H, W).
        Concatenates the features of each channel.
        """
        # Loop over channels, extract features and concatenate them
        for ch_idx in range(image.shape[0]):
            channel_features = self.extract_features(image[ch_idx], **kwargs)
            if ch_idx == 0:
                features = channel_features
            else:
                features = np.concatenate((features, channel_features), axis=0)

        return features

    @staticmethod
    def expand_img(img):
        """
        Expands the image to have 2 channels.
        Expects the image to be in the shape (H, W) or (C, H, W) with C=1 or C=2.
        Returns the image in the shape (2, H, W).
        """
        if img.ndim == 2:
            img_expanded = np.stack([img, img], axis=0)
        elif img.ndim == 3:
            if img.shape[0] == 1:
                img_expanded = np.concatenate([img, img], axis=0)
            if img.shape[0] == 2:
                img_expanded = img
        return img_expanded
