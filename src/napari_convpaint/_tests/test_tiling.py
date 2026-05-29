"""Hypothesis: Tiling must not change the segmentation.

The padding applied before tiling is derived from the feature extractor's
receptive field (kernel size, conv depth / pooling) and binning (fe_scalings).
With that padding AND alignment of tile dims to the FE's downsampling grid in
place, feature values at annotated pixels match whether tiling is on or off —
so, with a deterministic classifier trained on a canonical sample order,
training produces the same model and segmentation is pixel-identical.

These tests pin both tiling paths:

1. `utils.tile_annot` — tiles must be aligned to `patch_size * total_stride *
   lcm(fe_scalings)`. Deeper feature maps are at coarser resolution (e.g. 1/2
   for VGG-m, 1/4 for VGG-l); if the tile dim is not a multiple of that, the
   upsample ratio differs from the whole-padded-image case and align_corners=
   False samples the feature map at shifted sub-pixel positions.

2. `_parallel_predict_image` — the per-tile margin must match the FE's actual
   required padding (`get_padding() * max(fe_scalings)`), and both margin and
   maxblock must be aligned to the same downsampling grid.
"""

import torch  # noqa: F401  — must import before catboost/napari to avoid segfault on macOS
import numpy as np
import pytest

from napari_convpaint.convpaint_model import ConvpaintModel
from napari_convpaint.testing_data import (
    generate_synthetic_square,
    generate_synthetic_circle_annotation,
)


def _make_data(im_dims, square_dims, circle_offset, seed=0):
    np.random.seed(seed)  # generate_synthetic_square uses the global RNG
    im_hwc, _ = generate_synthetic_square(im_dims=im_dims, square_dims=square_dims, rgb=True)
    # channel_mode='rgb' expects (C, H, W); generate_synthetic_square returns (H, W, C).
    im = np.moveaxis(im_hwc, -1, 0)
    im_annot = generate_synthetic_circle_annotation(
        im_dims=im_dims,
        circle1_xy=(im_dims[0] // 2, circle_offset),
        circle2_xy=(im_dims[0] // 2, im_dims[1] // 2),
    )
    return im, im_annot


def _build_model(alias, fe_scalings, tile_annotations, tile_image):
    cp_model = ConvpaintModel(alias=alias)
    cp_model.set_params(
        fe_scalings=fe_scalings,
        tile_annotations=tile_annotations,
        tile_image=tile_image,
        channel_mode="rgb",
        normalize=1,
        image_downsample=1,
    )
    return cp_model


def _assert_equal_segmentations(seg_a, seg_b, label_a, label_b):
    assert seg_a.shape == seg_b.shape, (
        f"shape mismatch: {label_a}={seg_a.shape}, {label_b}={seg_b.shape}"
    )
    diff_mask = seg_a != seg_b
    n_diff = int(diff_mask.sum())
    n_total = int(seg_a.size)
    info = ""
    if n_diff > 0:
        m = diff_mask.reshape(diff_mask.shape[-2:]) if diff_mask.ndim > 2 else diff_mask
        ys, xs = np.where(m)
        info = f" diff bbox rows=[{ys.min()},{ys.max()}] cols=[{xs.min()},{xs.max()}]"
    assert n_diff == 0, (
        f"{label_a} vs {label_b}: {n_diff}/{n_total} pixels differ "
        f"({100 * n_diff / n_total:.3f}%).{info}"
    )


# --------------------------------------------------------------------------- #
# tile_annotations: annotation tiling during training                          #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "alias,fe_scalings",
    [
        ("vgg-m", [1]),  # 3 layers, total_stride=2 — misalignment by 1 (odd tile)
        ("vgg-l", [1]),  # 5 layers, total_stride=4 — misalignment up to 3 (bigger hit)
        ("ilastik", [1]),  # ilastik filter set; FE-reported kernel_size=41 -> pad=20
        ("gaussian", [1]),  # skimage.filters.gaussian, sigma=3; FE reports pad=2*sigma=6
    ],
)
@pytest.mark.parametrize("use_rf", [True, False], ids=["rf", "catboost"])
def test_tile_annotations_matches_no_tile(alias, fe_scalings, use_rf):
    """Annotation tiling should not change the segmentation.

    Train two models with identical seeds / params, differing only in
    tile_annotations. If the FE-derived padding around each tile covers the
    receptive field *and* the tile dims are aligned to the network's total
    stride, the feature vectors at annotated pixels match, the (seeded)
    classifier fits the same data, and segmentations are pixel-identical.

    Both `use_rf=True` (RandomForest, random_state=0) and `use_rf=False`
    (CatBoost, random_seed=0) are exercised — CatBoost is the UI default and
    is the more sensitive bar (its bootstrap is order-aware on top of being
    seeded). If features and row-order are bit-identical, both are bit-identical.
    """
    im, im_annot = _make_data(im_dims=(300, 300), square_dims=(70, 70), circle_offset=70)

    def seg(tile_flag):
        cp = _build_model(alias, fe_scalings, tile_annotations=tile_flag, tile_image=False)
        cp.train(im, im_annot, fe_use_device="cpu", use_rf=use_rf)
        return cp.segment(im, fe_use_device="cpu")

    _assert_equal_segmentations(
        seg(False), seg(True), "tile_annotations=False", "tile_annotations=True"
    )


# --------------------------------------------------------------------------- #
# tile_image: image tiling during prediction                                   #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "alias,fe_scalings,im_dims,square_dims,circle_offset",
    [
        # vgg-m no binning: required padding = 12 * 1 = 12 < 50 -> should pass.
        ("vgg-m", [1], (1100, 1100), (220, 220), 300),
        # vgg-l with binning: required padding = 36 * 4 = 144 > 50 -> should fail.
        ("vgg-l", [1, 2, 4], (1100, 1100), (220, 220), 300),
    ],
)
def test_tile_image_matches_no_tile(alias, fe_scalings, im_dims, square_dims, circle_offset):
    """Image tiling at predict time should not change the segmentation.

    Train one model with tile_annotations=False (so training is identical),
    then predict with tile_image on and off. If `_parallel_predict_image`'s
    per-tile margin is large enough for the FE's receptive field AND aligned
    to the total stride, segmentations should be pixel-identical.
    """
    im, im_annot = _make_data(im_dims=im_dims, square_dims=square_dims, circle_offset=circle_offset)

    cp = _build_model(alias, fe_scalings, tile_annotations=False, tile_image=False)
    cp.train(im, im_annot, fe_use_device="cpu", use_rf=True)

    cp._param.tile_image = False
    seg_no_tile = cp.segment(im, fe_use_device="cpu")

    cp._param.tile_image = True
    seg_tiled = cp.segment(im, fe_use_device="cpu")

    _assert_equal_segmentations(
        seg_no_tile, seg_tiled, "tile_image=False", "tile_image=True"
    )


# --------------------------------------------------------------------------- #
# Gaussian-reach padding: high-frequency texture exposes blur boundary leak   #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("use_rf", [True, False], ids=["rf", "catboost"])
def test_tile_annotations_textured_image_default_vgg(use_rf):
    """Default vgg16 (1 layer, fe_pad=1) with scalings [1,2,4] on textured input.

    Pins tile/no-tile equivalence on high-frequency content where any
    boundary-mode leakage in the downsample step would surface as feature
    drift. Smooth synthetic shapes don't trigger it; white-noise + sinusoids
    do. With block-mean downsampling (`scale_img` via `block_reduce`), each
    output pixel reads exactly `scaling_factor` input pixels strictly inside
    its block, so an aligned tile produces bit-identical features. Run for
    both classifiers — CatBoost is the UI default and the more sensitive bar.
    """
    rng = np.random.default_rng(0)
    h = w = 256
    # White noise + sinusoidal stripes: every pixel has neighbours that differ
    # strongly, so any boundary-mode blur produces visible feature drift.
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    stripes = 0.5 * np.sin(xx / 3.0) + 0.5 * np.sin(yy / 5.0)
    im_2d = (rng.standard_normal((h, w)) + stripes).astype(np.float32)
    im = np.stack([im_2d, im_2d, im_2d], axis=0)  # (3, H, W) for rgb mode

    annot = np.zeros((h, w), dtype=np.uint16)
    annot[40:55, 40:55] = 1
    annot[180:195, 180:195] = 2
    annot[100:103, 200:203] = 1  # tiny patch near boundary of its own tile

    def seg(tile_flag):
        cp = ConvpaintModel(alias='vgg')  # default vgg16, scalings [1,2,4]
        cp.set_params(
            tile_annotations=tile_flag,
            tile_image=False,
            channel_mode='rgb',
            normalize=1,
            image_downsample=1,
        )
        cp.train(im, annot, fe_use_device='cpu', use_rf=use_rf)
        return cp.segment(im, fe_use_device='cpu')

    _assert_equal_segmentations(
        seg(False), seg(True),
        'tile_annotations=False', 'tile_annotations=True',
    )


# --------------------------------------------------------------------------- #
# image_downsample × pipeline alignment: pins that train+segment succeed and  #
# return original-shape output at each `image_downsample` value, including    #
# odd factors that previously crashed get_features_targets (image/labels      #
# shape drift before scale_img was made symmetric via pre-pad).               #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("image_downsample", [-2, 0, 2, 3, 4])
def test_train_and_segment_across_image_downsample(image_downsample):
    """End-to-end train+segment at boundary `image_downsample` values. Verifies
    (a) no crash through scale_img + _get_overall_paddings + FE + restore, and
    (b) segmentation comes back at the original input shape.
    """
    rng = np.random.default_rng(0)
    h = w = 256  # 256 % 3 = 1 → exercises the odd-factor pad path
    im_2d = rng.standard_normal((h, w)).astype(np.float32)
    im = np.stack([im_2d, im_2d, im_2d], axis=0)
    annot = np.zeros((h, w), dtype=np.uint16)
    annot[40:55, 40:55] = 1
    annot[180:195, 180:195] = 2

    cp = ConvpaintModel(alias='vgg')
    cp.set_params(
        tile_annotations=False, tile_image=False,
        channel_mode='rgb', normalize=1,
        image_downsample=image_downsample,
    )
    cp.train(im, annot, fe_use_device='cpu', use_rf=True)
    seg = cp.segment(im, fe_use_device='cpu')

    assert seg.shape[-2:] == (h, w), (
        f"image_downsample={image_downsample}: expected segmentation shape "
        f"to restore to ({h},{w}), got {seg.shape}"
    )