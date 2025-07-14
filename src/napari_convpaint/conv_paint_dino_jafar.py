import skimage
import numpy as np
from .conv_paint_feature_extractor import FeatureExtractor
from pathlib import Path
import torch
import torchvision.transforms as T
import warnings


from pathlib import Path
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate           # (single import is enough)
import napari_convpaint.jafar.hydra_plugins.resolvers  # ← registers the resolver

AVAILABLE_MODELS = ["vit_small_patch14_reg4_dinov2"]

def load_model(backbone_name: str, project_root: str, device):
    # ------------------------------------------------------------------
    # Absolute path to the vendored config directory
    # ------------------------------------------------------------------
    CONFIG_DIR = Path(__file__).resolve().parent / "jafar" / "config"
    if not CONFIG_DIR.is_dir():
        raise FileNotFoundError(f"Config dir not found: {CONFIG_DIR}")

    # ------------------------------------------------------------------
    # Initialise Hydra exactly once (relative paths not allowed here)
    # ------------------------------------------------------------------
    if not GlobalHydra.instance().is_initialized():
        initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None)

    # ------------------------------------------------------------------
    # Compose config with your overrides
    # ------------------------------------------------------------------
    overrides = [
        "val_dataloader.batch_size=1",
        f"project_root={project_root}",
        f"backbone.name={backbone_name}",
    ]
    cfg = compose(config_name="base", overrides=overrides)

    # ------------------------------------------------------------------
    # Instantiate backbone and model
    # ------------------------------------------------------------------
    backbone = instantiate(cfg.backbone).to(device)
    model    = instantiate(cfg.model).to(device)
    model.eval()

    return model, backbone

   # backbone = instantiate(cfg.backbone)
   # backbone.to(device)

    # Load Model
   #model = instantiate(cfg.model)
   # model.to(device)
   # model.eval()

    # Load checkpoint
   # try:
   #     checkpoint = torch.load(f"./output/jafar/{backbone.name}/model.pth", weights_only=True)
   # except Exception as e:
   #     print(f"Failed with weights_only=True, trying with weights_only=False: {e}")
   #     checkpoint = torch.load(f"./output/jafar/{backbone.name}/model.pth", weights_only=False)
  #  model.load_state_dict(checkpoint["jafar"])
 #   return model, backbone


@torch.inference_mode()
def extract_hr_from_patches_seq_overlap(
    image_batch: torch.Tensor,
    backbone,
    hr_head,
    *,
    patch_px: int = 448,
    overlap_tokens: int = 2,
):
    """
    High-resolution feature extraction with sliding-window overlap and smooth
    feather-blending. Works for images both larger **and** smaller than `patch_px`.
    """
    # ---------- basics ----------
    B, _, H, W = image_batch.shape
    overlap_px = overlap_tokens * 14          # ViT token = 14 px
    stride     = patch_px - overlap_px
    assert stride > 0, "overlap too large (stride ≤ 0)"

    # ---------- fallback for tiny images ----------
    if H <= patch_px and W <= patch_px:
        if H < patch_px or W < patch_px:
            warnings.warn(
                f"Image ({H}×{W}) smaller than patch size ({patch_px}): "
                "using single cropped patch."
            )
        lr_feats_full, _ = backbone(image_batch)
        hr_feats_full    = hr_head(image_batch, lr_feats_full, (H, W))
        return hr_feats_full, lr_feats_full

    # ---------- LR features of full image ----------
    lr_feats_full, _ = backbone(image_batch)

    # ---------- 2-D feather window ----------
    if overlap_px == 0:
        weight_2d = torch.ones(patch_px, patch_px, device=image_batch.device)
    else:
        ramp = torch.linspace(0, 1, overlap_px + 1, device=image_batch.device)[1:]
        flat = torch.ones(patch_px - 2 * overlap_px, device=image_batch.device)
        w1d  = torch.cat([ramp, flat, ramp.flip(0)])
        weight_2d = w1d[:, None] * w1d[None, :]   # (patch_px, patch_px)

    # ---------- helpers ----------
    def tok_slice(start_px: int) -> slice:
        """Convert pixel start→token slice along one axis."""
        return slice(start_px // 14, (start_px + patch_px) // 14)

    # ---------- accumulators ----------
    hr_sum       = None                       # (B, C_hr, H, W)
    weight_accum = torch.zeros(
        1, 1, H, W, dtype=image_batch.dtype, device=image_batch.device
    )

    # ---------- slide patch-by-patch ----------
    for b in range(B):

        # vertical starts
        if H <= patch_px:
            starts_h = [0]
        else:
            starts_h = list(range(0, H - patch_px + 1, stride))
            if starts_h[-1] != H - patch_px:      # cover bottom edge
                starts_h.append(H - patch_px)

        # horizontal starts
        if W <= patch_px:
            starts_w = [0]
        else:
            starts_w = list(range(0, W - patch_px + 1, stride))
            if starts_w[-1] != W - patch_px:      # cover right edge
                starts_w.append(W - patch_px)

        for top in starts_h:
            for left in starts_w:
                img_patch = image_batch[b:b+1, :, top:top+patch_px, left:left+patch_px]
                lr_patch  = lr_feats_full[
                    b:b+1, :, tok_slice(top), tok_slice(left)
                ]

                hr_patch = hr_head(img_patch, lr_patch, (patch_px, patch_px))  # (1,C,H,W)

                # lazy init accumulators once we know C_hr
                if hr_sum is None:
                    C_hr = hr_patch.size(1)
                    hr_sum = torch.zeros(
                        B, C_hr, H, W, dtype=hr_patch.dtype, device=hr_patch.device
                    )

                # -------- blended add (safe for edge crops) --------
                h_end = min(top + patch_px, H)
                w_end = min(left + patch_px, W)
                patch_h = h_end - top
                patch_w = w_end - left

                cropped_patch   = hr_patch[0, :, :patch_h, :patch_w]
                cropped_weight2 = weight_2d[:patch_h, :patch_w]

                hr_sum[b, :, top:h_end, left:w_end]         += cropped_patch * cropped_weight2
                weight_accum[:, :, top:h_end, left:w_end]   += cropped_weight2[None, None]

    # ---------- normalise ----------
    eps = 1e-8
    hr_feats_full = hr_sum / weight_accum.clamp(min=eps)

    return hr_feats_full, lr_feats_full

# 2) DEFINE THE INITIALIZATION, DESCRIPTION AND DEFAULT PARAMETERS
class DinoJafarFeatures(FeatureExtractor):
    # Define the initiation method
    def __init__(self, model_name="vit_small_patch14_reg4_dinov2", use_cuda=False):
        # Sets some default parameters and chooses the itialization method

        if use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                print("No GPU available, using CPU for feature extraction.")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        #super().__init__(model_name=model_name, use_cuda=use_cuda)

        # Define specifications for the feature extractor model, if necessary
        self.padding = 0
        self.patch_size = 14
        self.num_input_channels = [3]
        #self.device = torch.device("cpu")
        self.model = None
        self.backbone = None
        self.model_name = model_name
        self.use_cuda = use_cuda
        self.create_model(model_name, use_cuda)


    def get_description(self):
        return "" # Briefly describe the feature extractor here

    def get_default_params(self, param=None):
        param = super().get_default_params(param=param)
        param.fe_name = self.model_name
        param.fe_use_cuda = self.use_cuda
        param.fe_layers = []
        param.fe_scalings = [1]
        param.fe_order = 0
        # param.image_downsample = 1
        param.tile_image = False
        param.tile_annotations = False

        # param.normalize = 1
        return param
    
    def get_enforced_params(self, param=None):
        param = super().get_enforced_params(param=param)
        param.fe_scalings = [1]
        return param
    
    def gives_patched_features(self):
        """
        Returns True if the feature extractor returns features that are patched, i.e. the
        features are extracted from patches of the image and then stitched together.
        """
        return False


# 3) OPTIONAL METHODS:

# FOR MODELS THAT NEED TO CREATE A SEPARATE FE MODEL, DEFINE THE FOLLOWING METHOD
    def create_model(self, model_name, use_cuda=False):
        backbone = "vit_small_patch14_reg4_dinov2.lvd142m"
        project_root = str(Path().absolute())
        model, backbone = load_model(backbone, project_root, self.device)    
        self.model = model
        self.backbone = backbone
        
# IF THE MODEL ABSOLUTELY REQUIRES SOME PARAMETERS TO BE SET, DEFINE THE FOLLOWING METHOD
    
        # def get_enforced_params(self, param=None):
        #     param = super().get_enforced_params(param=param)
            
        #     # Define here, which parameters shall be used as enforced parameters
        #     # param.fe_layers = []
        #     # param.fe_scalings = [1]
        #     # etc.
    
        #     return param


# 4) CHOSE BETWEEN THE FOLLOWING METHODS HOW TO IMPLEMENT THE FEATURE EXTRACTION
#    WITH INCREASING CONTROL AND COMPLEXITY:


    def get_features_from_plane(self, image):
        """
        Extract HR ViT features from a single 2D image plane (C, H, W), using patch-wise overlap blending.
        Returns a numpy array of shape (nb_features, H, W).
        """
        # ---------------------- Prepare image ------------------------
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        if isinstance(image, torch.Tensor):
            if image.max() > 1.0:
                image = image / 255.0  # Assume uint8 image
        else:
            raise TypeError("Expected image as numpy array or torch.Tensor")

        if image.ndim == 2:
            image = image.unsqueeze(0)  # (H, W) -> (1, H, W)
        assert image.ndim == 3 and image.shape[0] == 3, \
            f"Expected 3-channel image, got shape {image.shape}"

        image_batch = image.unsqueeze(0).to(self.device, dtype=torch.float32)  # (1, 3, H, W)

        # ---------------------- Normalize ----------------------------
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[:, None, None]
        std  = torch.tensor([0.229, 0.224, 0.225], device=self.device)[:, None, None]
        image_batch = (image_batch - mean) / std

        # ---------------------- Extract features ---------------------
        hr_feats, _ = extract_hr_from_patches_seq_overlap(
            image_batch=image_batch,
            backbone=self.backbone,
            hr_head=self.model,
            patch_px=448,
            overlap_tokens=1,
        )

        return hr_feats[0].detach().cpu().numpy()  # (C_feat, H, W)
    
        # Define how to extract features from a single plane of the image, assuming the number of
        # channels is compatible with the model (self.num_input_channels). Input = [C, H ,W]
        # Important: Output needs to be (a list of) 3D: [nb_features, H, W]
        #

#    If this is given, stacks are simply treated as independent planes and their features
#    stacked back along the z axis. Also, the channels are automatically repeated to suit the
#    required input channels, and the output of the channels are treated as different features.
#    Finally, the image is automatically scaled according to the scaling factors in the
#    parameters, reduced to patch size. The output is then scaled back to the original size.

# b) get_features(self, image):
#
#    Define how to extract features from an image stack (4D), assuming the number of input channels
#    is compatible with the model (self.num_input_channels). Input = [C, Z, H ,W]
#    Important: Output needs to be (a list of) 4D: [nb_features, Z, H, W]
#    
#    If this is given, the channels are automatically repeated to suit the required input channels,
#    and the output of the channels are treated as different features.
#    Finally, the image is automatically scaled according to the scaling factors in the
#    parameters, reduced to patch size. The output is then scaled back to the original size.

# c) get_features_from_channels(self, image):
#
#    Define how to extract features from an image stack (4D) with an arbitrary number of channels.
#    Input = [C, Z, H ,W]
#    Important: Output needs to be (a list of) 4D: [nb_features, Z, H, W]
#
#    If this is given, the image is automatically scaled according to the scaling factors in the
#    parameters, reduced to patch size. The output is then scaled back to the original size.
    
# d) get_feature_pyramid(self, data, param, patched=True):
#
#    Define the full feature extraction process, including the feature pyramid. Input = [C, Z, H ,W]
#    Important: Output needs to be 4D: [nb_features, Z, H, W]


# 5) TO USE THE MODEL INSIDE THE CONVPAINT MODEL, YOU NEED TO FOLLOW THE FOLLOWING STEPS:

# - Import the AVAILABLE_MODELS and the class of the feature extractor
# - Implement to add the model to the FE_MODELS_TYPES_DICT in the _init_fe_models_dict method
# - Optional: Add it to the dictionary std_models, to enable loading it with an alias
