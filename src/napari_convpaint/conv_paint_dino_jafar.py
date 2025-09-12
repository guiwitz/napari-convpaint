import sys
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from .conv_paint_utils import get_device, get_device_from_torch_model, guided_model_download
from .conv_paint_feature_extractor import FeatureExtractor
from typing import List, Tuple
import copy
from napari_convpaint.jafar.layers import PretrainedViTWrapper, JAFAR

AVAILABLE_MODELS = ["dino_jafar_small"]

class DinoJafarFeatures(FeatureExtractor):
    """
    DINOv2 + JAFAR upsampler feature extractor integrated with ConvPaint.
    Expects that ConvPaint already padded/cropped images so H,W are multiples
    of self.patch_size (14). Provides dynamic patch size: large images use
    448px (14*32) sliding patches with overlap; smaller images shrink patch
    size to the largest multiple of 14 that fits within min(H,W).

    This code is adapted from the JAFAR implementation:
    Paul Couairon, Loick Chambon, Louis Serrano, Jean-Emmanuel Haugeard, Matthieu Cord, Nicolas Thome
    *JAFAR: Jack up Any Feature at Any Resolution*
    {https://arxiv.org/abs/2506.11136}
    {github.com/PaulCouairon/JAFAR/}
    """

    def __init__(self, model_name="dino_jafar_small", use_gpu=False):
        super().__init__(model_name=model_name, use_gpu=use_gpu)
        
        self.patch_size = 14          # token size of ViT
        self.padding    = 0           # model-internal extra pad (none)
        self.num_input_channels = [3] # RGB
        self.norm_imagenet = True

        # Parent .create_model() saves tuple (hr_head, backbone) in self.model
        self.model, self.backbone = self.model
        self.device = get_device_from_torch_model(self.model)

    # ------------------------------------------------------------------ #
    # Load JAFAR upscaler and DINOv2 backbone
    # ------------------------------------------------------------------ #
    @staticmethod
    def create_model(model_name="dino_jafar_small", use_gpu=False):
        """
        Load DINOv2 backbone and JAFAR model head from remote .pth checkpoints using guided download.
        """

        device = get_device(use_gpu)

        # Define filenames
        backbone_file = "dinov2_vits14_reg4_pretrain.pth"
        backbone_url = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/" + backbone_file

        jafar_file = "vit_small_patch14_reg4_dinov2.pth"
        jafar_url = "https://github.com/PaulCouairon/JAFAR/releases/download/Weights/" + jafar_file

        # Download checkpoints
        _ = guided_model_download(backbone_file, backbone_url) # Just make sure the backbone is downloaded
        jafar_ckpt = guided_model_download(jafar_file, jafar_url)

        # --- Load backbone ---
        internal_names = {"dino_jafar_small": "vit_small_patch14_reg4_dinov2"}
        backbone = PretrainedViTWrapper(name=internal_names[model_name]).to(device)

        # --- Instantiate JAFAR head ---
        model = JAFAR(
            input_dim=3,
            qk_dim=128,
            v_dim=384,
            feature_dim=384,
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
    def gives_patched_features(self) -> bool:
        # JAFAR returns per-pixel (upsampled) features already.
        return False

    def get_description(self):
        return ("DINOv2 + JAFAR upsampler feature extractor\n"
                "Patch size 14, overlap blending, CPU-accumulated HR features.")

    def get_default_params(self, param=None):
        param = super().get_default_params(param)
        param.fe_scalings = [1,8,14]
        param.fe_order = 0
        param.tile_image = False
        param.tile_annotations = False
        return param

    def get_enforced_params(self, param=None):
        self.jafar_scalings = param.fe_scalings
        param = super().get_enforced_params(param)
        param.fe_scalings = [1]
        #if not param.fe_scalings:
            #param.fe_scalings = [4]
        return param

    # ------------------------------------------------------------------ #
    # Public extraction entry points
    # ------------------------------------------------------------------ #
    def get_features_from_plane(self, img: np.ndarray):
        """
        Extract features for a single Z plane (RGB). Shape: [3,H,W] -> [F,H,W].
        Dynamic patch size with single (implicit) scale = 1.
        """
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

    def _choose_tile_params(
        self,
        H: int,
        W: int,
        desired_tile_px: int,
        overlap_tokens: int,
    ) -> Tuple[int, int]:
        """
        Decide actual tile_px (multiple of self.patch_size) <= min(H,W, desired_tile_px).
        Adjust overlap_tokens if necessary so stride > 0.
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
    def _extract_tiled_multiscale(
        self,
        image_batch: torch.Tensor,
        backbone,
        hr_head,
        *,
        tile_px: int,
        overlap_tokens: int,
        scales: List[int],
    ):
        B, _, H, W = image_batch.shape
        dev = image_batch.device
        ps = self.patch_size

        ov_px = overlap_tokens * ps
        stride = tile_px - ov_px
        assert stride > 0, "Invalid stride (overlap too large)."

        hr_head_cpu = copy.deepcopy(hr_head).to("cpu").eval()

        lr_full, _ = backbone(image_batch)

        if ov_px == 0:
            w2d = torch.ones(tile_px, tile_px, device=dev, dtype=torch.float32)
        else:
            ramp = torch.linspace(0, 1, ov_px + 1, device=dev, dtype=torch.float32)[1:]
            flat = torch.ones(tile_px - 2 * ov_px, device=dev, dtype=torch.float32)
            w1d = torch.cat([ramp, flat, ramp.flip(0)])
            w2d = w1d[:, None] * w1d[None, :]

        def tok_slice(px):
            return slice(px // ps, (px + tile_px) // ps)

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

        hr_sums = [None] * len(scales)
        weight_cpu = torch.zeros((1, 1, H, W), dtype=torch.float32, device="cpu")

        for b in range(B):
            for top in grid_h:
                for left in grid_w:
                    img_tile = image_batch[b:b+1, :, top:top+tile_px, left:left+tile_px]
                    lr_tile = lr_full[b:b+1, :, tok_slice(top), tok_slice(left)]

                    per_scale_feats = []
                    for s in scales:
                        out_h = max(1, tile_px // s)
                        try:
                            feat = hr_head(img_tile, lr_tile, (out_h, out_h))
                        except RuntimeError as e:
                            if "MPS" in str(e) or "weight type" in str(e):
                                feat = hr_head_cpu(img_tile.cpu(), lr_tile.cpu(), (out_h, out_h)).to(dev)
                            else:
                                raise
                        if out_h != tile_px:
                            feat = F.interpolate(feat, size=(tile_px, tile_px), mode="nearest")
                        per_scale_feats.append(feat)  # (1,Cs,tile_px,tile_px)

                    feat_cat = torch.cat(per_scale_feats, dim=1)  # (1, ΣC, tile_px, tile_px)
                    if hr_sums[0] is None:
                        total_C = feat_cat.shape[1]
                        assert total_C % len(scales) == 0, "Channel count not divisible by #scales."
                        per_scale_C = total_C // len(scales)
                        for i in range(len(scales)):
                            hr_sums[i] = torch.zeros((B, per_scale_C, H, W),
                                                     dtype=torch.float32, device="cpu")

                    chunks = feat_cat[0].chunk(len(scales), dim=0)
                    for i, chunk in enumerate(chunks):
                        weighted = chunk * w2d
                        h_eff, w_eff = weighted.shape[-2:]
                        hr_sums[i][b, :, top:top+h_eff, left:left+w_eff] += weighted.cpu()
                    weight_cpu[:, :, top:top+tile_px, left:left+tile_px] += w2d.cpu()

                    del img_tile, lr_tile, per_scale_feats, feat_cat, chunks, weighted
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()

        eps = 1e-8
        normalized = [s_sum / weight_cpu.clamp(min=eps) for s_sum in hr_sums]
        return torch.cat(normalized, dim=1)  # (B, C_total, H, W)