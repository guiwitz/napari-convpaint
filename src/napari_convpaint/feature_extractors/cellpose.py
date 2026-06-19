import warnings
import torch
import numpy as np
import skimage
import importlib.util
from ..utils import get_device_from_torch_model

def import_models():
    try:
        from cellpose import models
    except ImportError:
        models = None
    return models

# Check availability and provide infos for ConvpaintModel

def cellpose_available():
    available = importlib.util.find_spec("cellpose") is not None
    # if not available:
    #     warnings.warn(
    #         "Cellpose is not installed and is not available as feature extractor.\n"
    #         "Run 'pip install napari-convpaint[cellpose]' to install it."
    #     )
    return available


AVAILABLE_MODELS = ['cellpose_backbone'] if cellpose_available() else []

STD_MODELS = {
    "cellpose": {"fe_name": "cellpose_backbone"},
}

IMPORT_ERROR_MESSAGE = (
            "Cellpose is not installed and is not available as feature extractor.\n"
            "Run 'pip install napari-convpaint[cellpose]' to install it."
        )

# Actual feature extractor implementation

from ..feature_extractor import FeatureExtractor

class CellposeFeatures(FeatureExtractor):
    """Feature extractor using the Cellpose model."""
    def __init__(self, model_name='cellpose_backbone', model=None, **kwargs):

        super().__init__(model_name=model_name, model=model)
        self.patch_size = 8
        self.has_global_context = True
        self.num_input_channels = [2]
        self.norm_mode = "percentile"

        self.device = get_device_from_torch_model(self.model.net) if self.model is not None else torch.device("cpu")

    @staticmethod
    def create_model(model_name):

        models = import_models()
        if models is None:
            raise ImportError(
            "Cellpose could not be imported. If called through ConvpaintModel, this should not happen as the availability of Cellpose is checked before. " +
            "Make sure to have cellpose installed and available in your environment."
        )

        # Load the cellpose model
        model_cellpose = models.CellposeModel(model_type='tissuenet_cp3',
                                              gpu=False) # We will move the model to the appropriate device at feature extraction time
        return model_cellpose

    def get_description(self):
        return "Model specialized in cell segmentation."
    
    def gives_patched_features(self) -> bool:
        # Requires image divisible by 8x8 patches as input, but returns non-patched features
        return False

    def get_default_params(self, param=None):
        param = super().get_default_params(param=param)
        param.fe_name = self.model_name
        param.fe_layers = None
        param.fe_scalings = [1]
        param.fe_order = 0
        param.tile_annotations = False
        return param
    
    def move_model_to_device(self, device=torch.device("cpu")):
        """
        Move cellpose feature extractor model to the runtime device.
        """
        if isinstance(device, torch.device):
            device = device
        elif device is None:
            device = torch.device("cpu")
        elif isinstance(device, str):
            warnings.warn(f"Resolving device from string '{device}' is deprecated. Please provide a torch.device object instead.")
            # Legacy fallback for direct FE usage outside ConvpaintModel.
            if device == "cpu":
                device = torch.device("cpu")
            elif torch.cuda.is_available():
                device = torch.device("cuda:0")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            raise ValueError(f"Invalid device: {device}. Please provide a torch.device object or a string ('cpu', 'cuda', 'mps').")

        if self.model is None:
            self.device = device
            return device

        current_device = next(self.model.net.parameters()).device
        if current_device != device:
            self.model.net = self.model.net.to(device)
            self.model.device = device
            self.model.gpu = device.type in ("cuda", "mps")
            self.model.net.eval()

        self.device = device
        return device
    
    def supported_devices(self):
        
        if (self.model is not None and
                hasattr(self.model, "net")
        ):
            return [torch.device("cuda"), torch.device("mps"), torch.device("cpu")]
        else:
            return [torch.device("cpu")]

    def extract_features_from_plane(self, image, device=torch.device("cpu")):

        self.move_model_to_device(device)
        net = getattr(self.model, "net", self.model)
        image_expanded = np.expand_dims(image, axis=0)
        tensor = torch.from_numpy(image_expanded).float()
        tensor = tensor.to(self.device)
        use_mkldnn = getattr(net, "mkldnn", False) and self.device.type == "cpu"

        with torch.no_grad():
            if use_mkldnn:
                tensor = tensor.to_mkldnn()
            T0 = net.downsample(tensor)
            if use_mkldnn:
                style = net.make_style(T0[-1].to_dense())
            else:
                style = net.make_style(T0[-1])
            if not net.style_on:
                style = style * 0
            T1 = net.upsample(style, T0, use_mkldnn)
            T1 = net.output(T1)
            if use_mkldnn:
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
            if (w,h) != (w_img,h_img):
                t = skimage.transform.resize(
                        image=t,
                        output_shape=(f, w_img, h_img),
                        preserve_range=True, order=0)
            out_t.append(t)

        #append the output tensor from T1 (gradients and cell probability)
        t = T1.detach().cpu().numpy()[0]
        f,w,h = t.shape[-3:]
        if (w,h) != (w_img,h_img):
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
