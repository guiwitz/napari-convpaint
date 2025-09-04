import torch
import numpy as np
from .conv_paint_utils import get_device, normalize_image_percentile
from .conv_paint_feature_extractor import FeatureExtractor
import skimage
from cellpose import models
import warnings

AVAILABLE_MODELS = ['cellpose_backbone']


class CellposeFeatures(FeatureExtractor):

    def __init__(self, model_name='cellpose_backbone', model=None, use_gpu=False):

        super().__init__(model_name=model_name, model=model, use_gpu=use_gpu)
        self.patch_size = 8

        self.device = self.model.device if self.model is not None else None

    @staticmethod
    def create_model(model_name, use_gpu=False):
        # Load the cellpose model
        model_cellpose = models.CellposeModel(model_type='tissuenet_cp3',
                                              gpu=use_gpu)
        return model_cellpose.net
    
    def get_description(self):
        return "Model specialized in cell segmentation."
    
    def gives_patched_features(self) -> bool:
        # Requires image divisible by 8x8 patches as input, but returns non-patched features
        return False

    def get_default_params(self, param=None):
        param = super().get_default_params(param=param)
        param.fe_name = self.model_name
        param.fe_use_gpu = self.use_gpu
        param.fe_layers = []
        param.fe_scalings = [1]
        param.fe_order = 0
        param.tile_annotations = False
        return param
    
    def get_num_input_channels(self):
        return [2]

    def get_features_from_plane(self, image):
        image = normalize_image_percentile(image)

        image_expanded = np.expand_dims(image, axis=0)
        tensor = torch.from_numpy(image_expanded).float()
        tensor = tensor.to(self.device)

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
            # Put to cpu, detach, and convert to numpy
            t = t.detach().cpu().numpy()[0]
            # Resize if necessary
            f,w,h = t.shape[-3:]
            if (w,h != w_img,h_img):
                t = skimage.transform.resize(
                        image=t,
                        output_shape=(f, w_img, h_img),
                        preserve_range=True, order=0)
            out_t.append(t)

        #append the output tensor from T1 (gradients and cell probability)
        t = T1.detach().cpu().numpy()[0]
        f,w,h = t.shape[-3:]
        if (w,h != w_img,h_img):
            t = skimage.transform.resize(
                    image=t,
                    output_shape=(f, w_img, h_img),
                    preserve_range=True, order=0)
        out_t.append(t)

        #append the original image
        out_t.append(image)
        
        #combine the tensors
        out_t = np.concatenate(out_t, axis=0)

        return out_t
