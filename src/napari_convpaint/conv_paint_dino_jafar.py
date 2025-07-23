import sys
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from IPython.display import clear_output
from .conv_paint_utils import get_device
from .conv_paint_feature_extractor import FeatureExtractor
from typing import List, Tuple
import copy
from pathlib import Path
from napari_convpaint.jafar.layers import PretrainedViTWrapper, JAFAR



AVAILABLE_MODELS = ["vit_small_patch14_reg4_dinov2"]


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

    def __init__(self, model_name="vit_small_patch14_reg4_dinov2", use_cuda=False):
        super().__init__(model_name=model_name, use_cuda=use_cuda)
        self.patch_size = 14          # token size of ViT
        self.padding    = 0           # model-internal extra pad (none)
        self.num_input_channels = [3] # RGB

        self.device = get_device(use_cuda)
        # Parent .create_model() returned tuple (hr_head, backbone) in self.model
        self.model, self.backbone = self.model
        self.backbone = self.backbone.to(self.device).eval()
        self.model    = self.model.to(self.device).eval()


    # ------------------------------------------------------------------ #
    # Load JAFAR upscaler and DINOv2 backbone
    # ------------------------------------------------------------------ #
    @staticmethod
    def create_model(model_name="vit_small_patch14_reg4_dinov2", use_cuda=False):
        """
        Load backbone and JAFAR model head without Hydra, from .pth checkpoints.
        """

        device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        head_file = "vit_small_patch14_reg4_dinov2.pth"

        project_root = Path().resolve().parents[0]
        print(f"Project root: {project_root}")
        ckpt_root = project_root / "napari-convpaint" / "src" / "napari_convpaint" / "jafar" / "checkpoints"

        # 1. Load backbone
        backbone = PretrainedViTWrapper(name=model_name).to(device)

        # 2. Instantiate JAFAR head
        model = JAFAR(
            input_dim=3,
            qk_dim=128,
            v_dim=384,
            feature_dim=384,
            kernel_size=1,
            num_heads=4,
            name="jafar"
        ).to(device)

        # 3. Load JAFAR head weights
        model_ckpt = ckpt_root / head_file
        state = torch.load(model_ckpt, map_location=device,weights_only=False)
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
        p = super().get_default_params(param)
        p.fe_scalings = [1,8,14]
        p.fe_order = 0
        p.tile_image = False
        p.tile_annotations = False
        return p

    def get_enforced_params(self, param=None):
        p = super().get_enforced_params(param)
        #if not p.fe_scalings:
            #p.fe_scalings = [4]
        return p

    # ------------------------------------------------------------------ #
    # Public extraction entry points
    # ------------------------------------------------------------------ #
    def get_features_from_plane(self, image: np.ndarray):
        """
        Extract features for a single Z plane (RGB). Shape: [3,H,W] -> [F,H,W].
        Dynamic patch size with single (implicit) scale = 1.
        """
        assert image.ndim == 3 and image.shape[0] == 3, "Expected image [3,H,W]"
        H, W = image.shape[1:]
        self._assert_multiples(H, W)

        img = torch.tensor(image[None], dtype=torch.float32, device=self.device)  # [1,3,H,W]
        img = self._imagenet_normalize(img)

        patch_px, overlap_tokens = self._choose_patch_params(H, W, desired_patch_px=self.patch_size * 32, overlap_tokens=2)

        hr, _ = self._extract_patched(
            image_batch=img,
            backbone=self.backbone,
            hr_head=self.model,
            patch_px=patch_px,
            overlap_tokens=overlap_tokens,
        )
        return hr[0].cpu().numpy()  # [C,H,W]

    def get_feature_pyramid(self, stack, param, patched=True):
        """
        Multi-scale pyramid without downsampling input; we instead vary the
        *requested output resolution per patch* inside JAFAR and then
        upsample coarse outputs back to full pixel resolution so all scales
        align and can be concatenated channel-wise.

        data: np.ndarray [3,Z,H,W]
        returns: np.ndarray [F_total, Z, H, W]
        """
        # Get the number of channels the model expects (e.g., RGB = 3)
        input_channels = self.get_num_input_channels()

        # If the image has one the number of channels the model expects, use it directly
        if stack.shape[0] in input_channels:
            channel_series = [stack]
        else:
            # For each channel, create a replicate with the needed number of input channels
            input_channels = min(input_channels)
            channel_series = [np.tile(ch, (input_channels, 1, 1, 1)) for ch in stack]

        # Get outputs for each channel_series
        features_all_channels = []


        for data in channel_series:
            assert data.ndim == 4 and data.shape[0] == 3, "Expected data [3,Z,H,W]"
            _, Z, H, W = data.shape
            self._assert_multiples(H, W)
            scales = param.fe_scalings if (param and param.fe_scalings) else [1]
            scales = sorted({int(s) for s in scales if int(s) >= 1}) or [1]

            patch_px, overlap_tokens = self._choose_patch_params(H, W, desired_patch_px=self.patch_size * 32, overlap_tokens=2)

            feats_per_z = []
            for z in range(Z):
                plane = torch.tensor(data[:, z][None], dtype=torch.float32, device=self.device)  # [1,3,H,W]
                plane = self._imagenet_normalize(plane)

                hr_cat = self._extract_patched_multiscale(
                    image_batch=plane,
                    backbone=self.backbone,
                    hr_head=self.model,
                    patch_px=patch_px,
                    overlap_tokens=overlap_tokens,
                    scales=scales,
                )  # (1, C_total, H, W) CPU
                feats_per_z.append(hr_cat[0].cpu().numpy())
            features = np.stack(feats_per_z, axis=1)  # [F_total, Z, H, W]

            features_all_channels.append(features)
        #concat along first axis 
        features_all_channels = np.concatenate(features_all_channels, axis=0)  # [F_total, C, Z, H, W]
        print(f"Extracted features shape: {features_all_channels.shape}")
        return features_all_channels

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _imagenet_normalize(self, img: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device, dtype=img.dtype)[None, :, None, None]
        std  = torch.tensor([0.229, 0.224, 0.225], device=img.device, dtype=img.dtype)[None, :, None, None]
        return (img - mean) / std
    
    def _imagenet_normalize(self, img: torch.Tensor) -> torch.Tensor:
        """
        Normalize a torch tensor image to ImageNet stats.
        1) Brings values into [0,1] by:
        - dividing uint8 by 255
        - dividing uint16 by 65535
        - otherwise min/max scaling floats
        2) Applies per-channel ImageNet mean/std.
        
        Args:
            img: Tensor of shape [C, H, W] or [C, Z, H, W], dtype uint8, uint16, or float.
        
        Returns:
            Tensor, same shape & device, dtype float32, normalized.
        """
        device = img.device
        # Step 1: cast to float32
        x = img.float()

        # Bring into [0,1]
        if img.dtype == torch.uint8:
            x = x / 255.0
        elif img.dtype == torch.uint16:
            x = x / 65535.0
        else:
            # float input: min–max scale
            mn = x.amin()
            mx = x.amax()
            # avoid zero division
            denom = (mx - mn).clamp_min(1e-6)
            x = (x - mn) / denom

        # Step 2: ImageNet normalization
        # shape: (3, 1, 1) for [C,H,W] or (3,1,1,1) for [C,Z,H,W]
        spatial_dims = x.ndim - 1
        shape = (3,) + (1,) * spatial_dims
        mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32).reshape(shape)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32).reshape(shape)

        return (x - mean) / std

    def _assert_multiples(self, H: int, W: int):
        ps = self.patch_size
        if (H % ps) != 0 or (W % ps) != 0:
            raise ValueError(
                f"DinoJafarFeatures expects H,W multiples of patch size {ps}; got H={H}, W={W}. "
                "ConvpaintModel should pad beforehand."
            )

    def _choose_patch_params(
        self,
        H: int,
        W: int,
        desired_patch_px: int,
        overlap_tokens: int,
    ) -> Tuple[int, int]:
        """
        Decide actual patch_px (multiple of self.patch_size) <= min(H,W, desired_patch_px).
        Adjust overlap_tokens if necessary so stride > 0.
        """
        ps = self.patch_size
        max_fit = (min(H, W) // ps) * ps
        if max_fit == 0:
            # Extremely small edge case
            patch_px = ps
        else:
            patch_px = min(desired_patch_px, max_fit)

        # Ensure overlap_tokens yields positive stride
        # stride = patch_px - overlap_tokens*ps
        max_overlap_tokens = (patch_px // ps) - 1  # need at least one stride
        if max_overlap_tokens < 0:
            max_overlap_tokens = 0
        if overlap_tokens > max_overlap_tokens:
            overlap_tokens = max_overlap_tokens

        return patch_px, overlap_tokens

    # ------------------------------------------------------------------ #
    # Core patch extraction (single scale)
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def _extract_patched(
        self,
        image_batch: torch.Tensor,
        backbone,
        hr_head,
        *,
        patch_px: int,
        overlap_tokens: int = 1,
    ):
        B, _, H, W = image_batch.shape
        dev = image_batch.device
        ps = self.patch_size
        ov_px = overlap_tokens * ps
        stride = patch_px - ov_px
        assert stride > 0, "Invalid stride (overlap too large for patch size)."

        lr_full, _ = backbone(image_batch)

        dtype = image_batch.dtype
        if ov_px == 0:
            w2d = torch.ones(patch_px, patch_px, device=dev, dtype=dtype)
        else:
            ramp = torch.linspace(0, 1, ov_px + 1, device=dev, dtype=dtype)[1:]
            flat = torch.ones(patch_px - 2 * ov_px, device=dev, dtype=dtype)
            w1d = torch.cat([ramp, flat, ramp.flip(0)])
            w2d = w1d[:, None] * w1d[None, :]

        def tok_slice(px):
            return slice(px // ps, (px + patch_px) // ps)

        hr_sum = None
        w_acc = torch.zeros((1, 1, H, W), device="cpu", dtype=dtype)

        # Grids (allow partial coverage along big dimension if other is small)
        if H <= patch_px:
            grid_h = [0]
        else:
            grid_h = list(range(0, H - patch_px + 1, stride))
            if grid_h[-1] != H - patch_px:
                grid_h.append(H - patch_px)
        if W <= patch_px:
            grid_w = [0]
        else:
            grid_w = list(range(0, W - patch_px + 1, stride))
            if grid_w[-1] != W - patch_px:
                grid_w.append(W - patch_px)

        for b in range(B):
            for top in grid_h:
                for left in grid_w:
                    img_patch = image_batch[b:b+1, :, top:top+patch_px, left:left+patch_px]
                    lr_patch = lr_full[b:b+1, :, tok_slice(top), tok_slice(left)]

                    hr_patch = hr_head(img_patch, lr_patch, (patch_px, patch_px))  # (1,C,patch_px,patch_px)

                    if hr_sum is None:
                        C_hr = hr_patch.shape[1]
                        hr_sum = torch.zeros((B, C_hr, H, W), device="cpu", dtype=dtype)

                    weighted = (hr_patch[0] * w2d[:hr_patch.shape[-2], :hr_patch.shape[-1]]).cpu()
                    hr_sum[b, :, top:top+hr_patch.shape[-2], left:left+hr_patch.shape[-1]] += weighted
                    w_acc[:, :, top:top+hr_patch.shape[-2], left:left+hr_patch.shape[-1]] += w2d[:hr_patch.shape[-2], :hr_patch.shape[-1]].cpu()

                    del img_patch, lr_patch, hr_patch, weighted
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()

        eps = 1e-8
        hr_out = hr_sum / w_acc.clamp(min=eps)
        return hr_out, lr_full.cpu()

    # ------------------------------------------------------------------ #
    # Multi-scale extraction
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def _extract_patched_multiscale(
        self,
        image_batch: torch.Tensor,
        backbone,
        hr_head,
        *,
        patch_px: int,
        overlap_tokens: int,
        scales: List[int],
    ):
        B, _, H, W = image_batch.shape
        dev = image_batch.device
        ps = self.patch_size

        ov_px = overlap_tokens * ps
        stride = patch_px - ov_px
        assert stride > 0, "Invalid stride (overlap too large)."

        hr_head_cpu = copy.deepcopy(hr_head).to("cpu").eval()

        lr_full, _ = backbone(image_batch)

        if ov_px == 0:
            w2d = torch.ones(patch_px, patch_px, device=dev, dtype=torch.float32)
        else:
            ramp = torch.linspace(0, 1, ov_px + 1, device=dev, dtype=torch.float32)[1:]
            flat = torch.ones(patch_px - 2 * ov_px, device=dev, dtype=torch.float32)
            w1d = torch.cat([ramp, flat, ramp.flip(0)])
            w2d = w1d[:, None] * w1d[None, :]

        def tok_slice(px):
            return slice(px // ps, (px + patch_px) // ps)

        if H <= patch_px:
            grid_h = [0]
        else:
            grid_h = list(range(0, H - patch_px + 1, stride))
            if grid_h[-1] != H - patch_px:
                grid_h.append(H - patch_px)
        if W <= patch_px:
            grid_w = [0]
        else:
            grid_w = list(range(0, W - patch_px + 1, stride))
            if grid_w[-1] != W - patch_px:
                grid_w.append(W - patch_px)

        hr_sums = [None] * len(scales)
        weight_cpu = torch.zeros((1, 1, H, W), dtype=torch.float32, device="cpu")

        for b in range(B):
            for top in grid_h:
                for left in grid_w:
                    img_patch = image_batch[b:b+1, :, top:top+patch_px, left:left+patch_px]
                    lr_patch = lr_full[b:b+1, :, tok_slice(top), tok_slice(left)]

                    per_scale_feats = []
                    for s in scales:
                        out_h = max(1, patch_px // s)
                        try:
                            feat = hr_head(img_patch, lr_patch, (out_h, out_h))
                        except RuntimeError as e:
                            if "MPS" in str(e) or "weight type" in str(e):
                                feat = hr_head_cpu(img_patch.cpu(), lr_patch.cpu(), (out_h, out_h)).to(dev)
                            else:
                                raise
                        if out_h != patch_px:
                            feat = F.interpolate(feat, size=(patch_px, patch_px), mode="nearest")
                        per_scale_feats.append(feat)  # (1,Cs,patch_px,patch_px)

                    feat_cat = torch.cat(per_scale_feats, dim=1)  # (1, ΣC, patch_px, patch_px)
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
                    weight_cpu[:, :, top:top+patch_px, left:left+patch_px] += w2d.cpu()

                    del img_patch, lr_patch, per_scale_feats, feat_cat, chunks, weighted
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()

        eps = 1e-8
        normalized = [s_sum / weight_cpu.clamp(min=eps) for s_sum in hr_sums]
        return torch.cat(normalized, dim=1)  # (B, C_total, H, W)