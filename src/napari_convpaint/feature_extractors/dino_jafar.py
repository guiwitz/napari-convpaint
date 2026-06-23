import warnings
import numpy as np
import torch
import torch.nn.functional as F
from ..utils import get_device_from_torch_model, guided_model_download
from typing import List
import copy

def import_vitwrapper_jafar():
    try:
        from ..jafar.layers import PretrainedViTWrapper, JAFAR
    except ImportError:
        return None

    return {
        "PretrainedViTWrapper": PretrainedViTWrapper,
        "JAFAR": JAFAR,
    }

AVAILABLE_MODELS = ["dinov2_small-reg_jafar", "dinov3_small-plus_jafar"]

STD_MODELS = {
    "dinov2-jafar": {"fe_name": "dinov2_small-reg_jafar"},
    "dinov3-jafar": {"fe_name": "dinov3_small-plus_jafar"},
}

# Per-FE-name spec: which timm backbone, which weights URLs/filenames, and the
# JAFAR head dims (must match the embed_dim of the backbone). For DINOv2 the
# backbone weights live on Meta's CDN and timm finds them in torch hub cache;
# for DINOv3 the weights come from the timm-mirrored HuggingFace repo and we
# hand the file to timm explicitly via checkpoint_path.
JAFAR_BACKBONES = {
    "dinov2_small-reg_jafar": {
        "internal_name": "vit_small_patch14_reg4_dinov2",
        "backbone_file": "dinov2_vits14_reg4_pretrain.pth",
        "backbone_url": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth",
        "backbone_via_checkpoint_path": False,
        "jafar_file": "vit_small_patch14_reg4_dinov2.pth",
        "jafar_url": "https://github.com/PaulCouairon/JAFAR/releases/download/Weights/vit_small_patch14_reg4_dinov2.pth",
        "patch_size": 14,
        "embed_dim": 384,
    },
    "dinov3_small-plus_jafar": {
        "internal_name": "vit_small_plus_patch16_dinov3.lvd1689m",
        "backbone_file": "vit_small_plus_patch16_dinov3_lvd1689m.safetensors",
        "backbone_url": "https://huggingface.co/timm/vit_small_plus_patch16_dinov3.lvd1689m/resolve/main/model.safetensors",
        "backbone_via_checkpoint_path": True,
        "jafar_file": "vit_small_plus_patch16_dinov3.lvd1689m.pth",
        "jafar_url": "https://github.com/PaulCouairon/JAFAR/releases/download/Weights/vit_small_plus_patch16_dinov3.lvd1689m.pth",
        "patch_size": 16,
        "embed_dim": 384,
    },
}

from ..feature_extractor import FeatureExtractor

class DinoJafarFeatures(FeatureExtractor):
    """
    DINO + JAFAR upsampler feature extractor integrated with ConvPaint.
    Expects that ConvPaint already padded/cropped images so H,W are multiples
    of self.patch_size. Provides dynamic patch size: large images use sliding
    patches with overlap; smaller images shrink patch size to the largest
    multiple of the backbone patch size that fits within min(H,W).

    This code is adapted from the JAFAR implementation:
    Paul Couairon, Loick Chambon, Louis Serrano, Jean-Emmanuel Haugeard, Matthieu Cord, Nicolas Thome
    *JAFAR: Jack up Any Feature at Any Resolution*
    {https://arxiv.org/abs/2506.11136}
    {github.com/PaulCouairon/JAFAR/}
    """

    def __init__(self, model_name="dinov2_small-reg_jafar", **kwargs):
        super().__init__(model_name=model_name)

        spec = JAFAR_BACKBONES[model_name]
        self.patch_size = spec["patch_size"]    # ViT token size
        self.padding    = 0                     # model-internal extra pad (none)
        self.num_input_channels = [3]           # RGB
        self.norm_mode = "imagenet"
        self.rgb_input = True
        # The largest scale equals the backbone patch size — at that scale
        # JAFAR is asked for native patch-resolution output (no upsampling).
        self.proposed_scalings = [[1],
                                  [1, 8],
                                  [1, 8, self.patch_size],
                                  ]

        # Parent .create_model() saves tuple (hr_head, backbone) in self.model
        self.model, self.backbone = self.model
        self.device = get_device_from_torch_model(self.model)

    # ------------------------------------------------------------------ #
    # Load JAFAR upscaler and ViT backbone
    # ------------------------------------------------------------------ #

    @staticmethod
    def create_model(model_name="dinov2_small-reg_jafar"):
        """
        Load ViT backbone and JAFAR head from remote .pth checkpoints using guided download.
        """
        device = torch.device("cpu") # Will move device at feature extraction time

        spec = JAFAR_BACKBONES[model_name]

        # Pre-fetch both checkpoints with guided progress.
        backbone_path = guided_model_download(spec["backbone_file"], spec["backbone_url"])
        jafar_ckpt    = guided_model_download(spec["jafar_file"],    spec["jafar_url"])

        vitwrapper_jafar = import_vitwrapper_jafar()
        if vitwrapper_jafar is None:
            raise ImportError(
                "JAFAR backbone could not be imported. If called through ConvpaintModel, this should not happen as the availability of JAFAR is checked before. " +
                "Make sure to have the jafar module installed and available in your environment."
            )
        wrapper, head = vitwrapper_jafar["PretrainedViTWrapper"], vitwrapper_jafar["JAFAR"]

        # DINOv3 weights come from HuggingFace, not torch hub cache, so we hand
        # the local file to timm explicitly. DINOv2's torch-hub URL aligns with
        # what timm fetches automatically, so the pre-download is enough.
        if spec["backbone_via_checkpoint_path"]:
            backbone = wrapper(name=spec["internal_name"], checkpoint_path=backbone_path).to(device)
        else:
            backbone = wrapper(name=spec["internal_name"]).to(device)

        # --- Instantiate JAFAR head ---
        model = head(
            input_dim=3,
            qk_dim=128,
            v_dim=spec["embed_dim"],
            feature_dim=spec["embed_dim"],
            kernel_size=1,
            num_heads=4,
            name="jafar"
        ).to(device)

        # --- Load JAFAR weights ---
        state = torch.load(jafar_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state.get("jafar", state))

        return model.eval(), backbone.eval()

    # ------------------------------------------------------------------ #
    # Metadata / enforced params
    # ------------------------------------------------------------------ #

    def gives_patched_features(self):
        # JAFAR returns per-pixel (upsampled) features already.
        return False

    def get_description(self):
        spec = JAFAR_BACKBONES[self.model_name]
        backbone_label = "DINOv3 ViT-S+" if "dinov3" in self.model_name.lower() else "DINOv2"
        return (f"{backbone_label} + JAFAR upsampler feature extractor\n"
                f"Patch size {spec['patch_size']}, overlap blending, CPU-accumulated HR features.")

    def get_default_params(self, param=None):
        param = super().get_default_params(param)
        param.fe_scalings = [1] # [1,8,14]
        param.fe_order = 0
        param.tile_image = False
        param.tile_annotations = False
        return param

    def get_enforced_params(self, param=None):
        # Use the given FE scalings internally as JAFAR upscales, but override the ConvpaintModel Pyramid FE scalings to [1]
        self.jafar_scalings = param.fe_scalings
        param = super().get_enforced_params(param)
        param.fe_scalings = [1]
        #if not param.fe_scalings:
            #param.fe_scalings = [4]
        return param

    # ------------------------------------------------------------------ #
    # Public extraction entry points
    # ------------------------------------------------------------------ #

    def move_model_to_device(self, device=torch.device("cpu")):
        """
        Move jafar feature extractor model to the runtime device.
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

        current_device = get_device_from_torch_model(self.model)
        if current_device != device:
            self.model = self.model.to(device).eval()
            self.backbone = self.backbone.to(device).eval()

        self.device = device
        return device

    def extract_features_from_plane(self, img: np.ndarray, device=torch.device("cpu")):
        """
        Extract features for a single Z plane (RGB). Shape: [3,H,W] -> [F,H,W].
        Dynamic patch size with single (implicit) scale = 1.
        """
        self.move_model_to_device(device)

        assert img.ndim == 3 and img.shape[0] == 3, "Expected image [3,H,W]"
        H, W = img.shape[1:]
        self._assert_multiples(H, W)
        img = torch.tensor(img[None], dtype=torch.float32, device=self.device)  # [1,3,H,W]

        tile_px, overlap_tokens = self._choose_tile_params(H, W, desired_tile_px=self.patch_size * 32, overlap_tokens=2)

        hr_cat = self._extract_tiled_multiscale(
                    image_batch=img,
                    backbone=self.backbone,
                    hr_head=self.model,
                    tile_px=tile_px,
                    overlap_tokens=overlap_tokens,
                    scales=self.jafar_scalings,
                )  # (1, C_total, H, W) CPU
        return hr_cat[0].cpu().numpy()  # [C,H,W]

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _assert_multiples(self, H: int, W: int):
        ps = self.patch_size
        if (H % ps) != 0 or (W % ps) != 0:
            raise ValueError(
                f"DinoJafarFeatures expects H,W multiples of patch size {ps}; got H={H}, W={W}. "
                "ConvpaintModel should pad beforehand."
            )

    def _choose_tile_params(self, H: int, W: int, desired_tile_px: int, overlap_tokens: int):
        """Decide tiling parameters for patch-based extraction.

        Behavior:
        - `tile_px` is chosen as the largest multiple of `self.patch_size`
          that does not exceed `min(H, W, desired_tile_px)`.
        - `overlap_tokens` is clamped so that the computed stride
          (tile_px - overlap_tokens * patch_size) is strictly positive.

        Returns:
        - tile_px: tile size in pixels (multiple of `self.patch_size`)
        - overlap_tokens: adjusted token overlap (non-negative int)
        """
        ps = self.patch_size
        max_fit = (min(H, W) // ps) * ps
        if max_fit == 0:
            # Extremely small edge case
            tile_px = ps
        else:
            tile_px = min(desired_tile_px, max_fit)

        # Ensure overlap_tokens yields positive stride
        # stride = tile_px - overlap_tokens*ps
        max_overlap_tokens = (tile_px // ps) - 1  # need at least one stride
        if max_overlap_tokens < 0:
            max_overlap_tokens = 0
        if overlap_tokens > max_overlap_tokens:
            overlap_tokens = max_overlap_tokens

        return tile_px, overlap_tokens

    # ------------------------------------------------------------------ #
    # Multi-scale extraction
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def _extract_tiled_multiscale(self, image_batch: torch.Tensor, backbone, hr_head, *,
                                  tile_px: int, overlap_tokens: int, scales: List[int]):
        """Extract high-resolution features by tiling the input and blending tiles.

        This method performs these high-level steps:
        1. Compute overlapping tile grid based on `tile_px` and `overlap_tokens`.
        2. For each tile, call the JAFAR `hr_head` at each requested `scale`
           to produce per-scale feature maps for that tile.
        3. Upsample per-scale outputs to the tile pixel resolution and
           accumulate them into CPU buffers using a 2D blending window.
        4. Normalize by the accumulated blending weights and concatenate
           per-scale results into a single multi-scale tensor.

        Notes:
        - Computation for `hr_head` is done on the same device as the
          input `image_batch`; a CPU fallback copy (`hr_head_cpu`) is
          prepared to handle device-specific runtime errors (e.g., MPS).
        - Accumulation is performed on CPU to avoid GPU memory pressure
          while still performing head inference on the runtime device.

        Returns:
        - Tensor of shape (B, C_total, H, W) on CPU.
        """
        B, _, H, W = image_batch.shape
        dev = image_batch.device
        ps = self.patch_size

        # Overlap and stride in pixels
        ov_px = overlap_tokens * ps
        stride = tile_px - ov_px
        assert stride > 0, "Invalid stride (overlap too large)."

        # A CPU copy of the hr_head is kept for device-incompatible runtime errors
        hr_head_cpu = copy.deepcopy(hr_head).to("cpu").eval()

        # Get low-resolution backbone features for the full image once
        lr_full, _ = backbone(image_batch)

        # Build 2D blending window (w2d) so overlapping tiles blend smoothly.
        # If no overlap, the window is uniform ones.
        if ov_px == 0:
            w2d = torch.ones(tile_px, tile_px, device=dev, dtype=torch.float32)
        else:
            # 1D ramp from 0->1 over overlap region, flat center, mirrored
            ramp = torch.linspace(0, 1, ov_px + 1, device=dev, dtype=torch.float32)[1:]
            flat = torch.ones(tile_px - 2 * ov_px, device=dev, dtype=torch.float32)
            w1d = torch.cat([ramp, flat, ramp.flip(0)])
            w2d = w1d[:, None] * w1d[None, :]

        # Helper to convert pixel coordinates to token indices for lr feature map
        def tok_slice(px):
            return slice(px // ps, (px + tile_px) // ps)

        # Build grid positions (top,left) that cover the image; include the
        # final tile aligned to the image border to ensure full coverage.
        if H <= tile_px:
            grid_h = [0]
        else:
            grid_h = list(range(0, H - tile_px + 1, stride))
            if grid_h[-1] != H - tile_px:
                grid_h.append(H - tile_px)
        if W <= tile_px:
            grid_w = [0]
        else:
            grid_w = list(range(0, W - tile_px + 1, stride))
            if grid_w[-1] != W - tile_px:
                grid_w.append(W - tile_px)

        # Prepare accumulation buffers per-scale (created lazily once we know
        # the channel counts produced by the hr_head).
        hr_sums = [None] * len(scales)
        weight_cpu = torch.zeros((1, 1, H, W), dtype=torch.float32, device="cpu")

        for b in range(B):
            for top in grid_h:
                for left in grid_w:
                    # Crop tile from original image and corresponding LR tokens
                    img_tile = image_batch[b:b + 1, :, top:top + tile_px, left:left + tile_px]
                    lr_tile = lr_full[b:b + 1, :, tok_slice(top), tok_slice(left)]

                    per_scale_feats = []
                    for s in scales:
                        # Requested output spatial size (in pixels) per scale
                        out_h = max(1, tile_px // s)
                        try:
                            feat = hr_head(img_tile, lr_tile, (out_h, out_h))
                        except RuntimeError as e:
                            # Some devices (MPS) or dtype mismatches may fail;
                            # retry on the CPU copy and move back to runtime device.
                            if "MPS" in str(e) or "weight type" in str(e):
                                feat = hr_head_cpu(img_tile.cpu(), lr_tile.cpu(), (out_h, out_h)).to(dev)
                            else:
                                raise
                        # If the head returned a lower-resolution map, upsample
                        if out_h != tile_px:
                            feat = F.interpolate(feat, size=(tile_px, tile_px), mode="nearest")
                        per_scale_feats.append(feat)  # (1, C_s, tile_px, tile_px)

                    # Concatenate features from all scales along channel dim
                    feat_cat = torch.cat(per_scale_feats, dim=1)  # (1, C_total, tile_px, tile_px)

                    # Initialize hr_sums buffers when we know channel counts
                    if hr_sums[0] is None:
                        total_C = feat_cat.shape[1]
                        assert total_C % len(scales) == 0, "Channel count not divisible by #scales."
                        per_scale_C = total_C // len(scales)
                        for i in range(len(scales)):
                            hr_sums[i] = torch.zeros((B, per_scale_C, H, W), dtype=torch.float32, device="cpu")

                    # Split concatenated channels back to per-scale chunks and accumulate
                    chunks = feat_cat[0].chunk(len(scales), dim=0)
                    for i, chunk in enumerate(chunks):
                        weighted = chunk * w2d
                        h_eff, w_eff = weighted.shape[-2:]
                        # Move to CPU accumulation buffer to reduce GPU memory pressure
                        hr_sums[i][b, :, top:top + h_eff, left:left + w_eff] += weighted.cpu()

                    # Track the blending weights used for normalization
                    weight_cpu[:, :, top:top + tile_px, left:left + tile_px] += w2d.cpu()

                    # Free temporary locals and clear MPS cache if available
                    del img_tile, lr_tile, per_scale_feats, feat_cat, chunks, weighted
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()  # avoid MPS fragmentation issues

        # Normalize accumulated sums by weights and concatenate per-scale results
        eps = 1e-8
        normalized = [s_sum / weight_cpu.clamp(min=eps) for s_sum in hr_sums]
        return torch.cat(normalized, dim=1)  # (B, C_total, H, W)