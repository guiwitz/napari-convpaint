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
        return param

    def get_features(self, img, **kwargs):
        """
        Gets the features of an image.

        Parameters:
        - image: The input image. Dimensions are [nb_channels, width, height]

        Returns:
        - features: The extracted features of the image. [nb_features, width, height]
        """

        if img.ndim == 2:
            img_expanded = np.stack([img, img], axis=0)
        elif img.ndim == 3:
            if img.shape[0] == 1:
                img_expanded = np.concatenate([img, img], axis=0)
            if img.shape[0] == 2:
                img_expanded = img

            elif img.shape[0] > 2:
                warnings.warn('Cellpose backbone expects < 3 channels. Using only the first two channels.') 
                img_expanded = img[:2]

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

        w_img,h_img = img.shape[-2:]

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
        if img.ndim == 2:
            out_t.append(np.expand_dims(img, axis=0))
        elif img.ndim == 3:
            out_t.append(img)
        out_t = np.concatenate(out_t,axis=0)
        return out_t
        