import warnings
import numpy as np
import pytest

from napari_convpaint import convpaint_model
from napari_convpaint.feature_extractors import Hookmodel
from napari_convpaint.param import Param
from napari_convpaint.utils import scale_img, create_instances_from_semantic
import skimage

def test_hook_model():
    
    model = Hookmodel(model_name='vgg16')
    layers = ['features.0 Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))',
          'features.12 Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))'
         ]
    model.register_hooks(selected_layers=layers)
    
    model.features_per_layer
    assert isinstance(model.features_per_layer, list), "Expect list of number of features"
    assert model.features_per_layer[0] == 64, f'Number of features expected from first layer is 64 but got {model.features_per_layer[0]}'
    assert model.features_per_layer[1] == 256, f'Number of features expected from first layer is 256 but got {model.features_per_layer[1]}'

def test_filter_image():

    param = Param()
    param.fe_name = 'vgg16'
    param.fe_layers = ['features.0 Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))',
            'features.12 Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))'
            ]
    param.fe_use_min_features = False
    param.fe_order = 0
    param.fe_scalings = [1]
    param.tile_annotations = True
    param.image_downsample = 1

    model = convpaint_model.ConvpaintModel(param=param)

    image = skimage.data.cells3d()
    image = image[30,1]
    image = image[60:188, 0:128]

    features = model.get_feature_image(data=image)
    assert features.shape[0] == 320, f'Expecting 320 features but got {features.shape[1]}'
    assert features.shape[1] == 128, f'Expecting 128 annotated pixels but got {features.shape[0]}'

    #disable annotations tiling should lead to the same results
    model.set_param("tile_annotations", False)
    features = model.get_feature_image(data=image)
    assert features.shape[0] == 320, f'Expecting 320 features but got {features.shape[1]}'
    assert features.shape[1] == 128, f'Expecting 128 annotated pixels but got {features.shape[0]}'


@pytest.mark.parametrize("upscale", [False, True], ids=["down", "up"])
@pytest.mark.parametrize("factor", [2, 3, 5, 7])
@pytest.mark.parametrize("H,W", [(256, 256), (255, 257), (100, 101)])
def test_scale_img_image_and_labels_shape_match(factor, H, W, upscale):
    """Image and labels paths must produce the same spatial shape after scale_img.

    Regression: previously the downsample image path centre-cropped (floor(H/f))
    while the labels path padded (ceil(H/f)), so non-aligned `image_downsample`
    crashed `get_features_targets` (boolean-mask shape mismatch). After the fix,
    both paths pre-pad to a multiple of f and reduce, so shapes always agree.
    """
    rng = np.random.default_rng(0)
    img = rng.standard_normal((1, H, W)).astype(np.float32)
    lbl = rng.integers(0, 3, size=(1, H, W), dtype=np.uint16)

    img_out = scale_img(img, factor, upscale=upscale, input_type="img")
    lbl_out = scale_img(lbl, factor, upscale=upscale, input_type="labels")

    assert img_out.shape[-2:] == lbl_out.shape[-2:], (
        f"shape mismatch at factor={factor}, upscale={upscale}, (H,W)=({H},{W}): "
        f"img={img_out.shape[-2:]}  lbl={lbl_out.shape[-2:]}"
    )

def _two_discs_seg():
    """Semantic segmentation (uint8, background = 1) with two touching discs of class 2
    (~700 px each, with a small hole in the first one) and a separate 16 px blob of class 2."""
    yy, xx = np.mgrid[:100, :100]
    seg = np.ones((100, 100), dtype=np.uint8)
    seg[(yy - 50)**2 + (xx - 35)**2 < 15**2] = 2
    seg[(yy - 50)**2 + (xx - 62)**2 < 15**2] = 2
    seg[52:54, 34:36] = 1 # 4 px hole in the first disc
    seg[10:14, 80:84] = 2 # 16 px blob
    return seg

def test_create_instances_from_semantic():
    seg = _two_discs_seg()

    # min_size=0: plain connected components (touching discs merged), nothing removed or filled
    inst = create_instances_from_semantic(seg, min_size=0)
    assert inst.shape == seg.shape and inst.dtype == np.int32
    assert np.array_equal(np.unique(inst), [0, 1, 2]) # merged discs + blob
    assert inst[52, 34] == 0, "Hole must not be filled with min_size=0"
    assert (inst > 0).sum() == (seg == 2).sum(), "Instances must cover exactly the class-2 pixels"

    # min_size=100: touching discs separated by watershed, blob removed, hole filled
    inst = create_instances_from_semantic(seg, min_size=100)
    assert np.array_equal(np.unique(inst), [0, 1, 2])
    assert inst[11, 81] == 0, "16 px blob must be removed"
    assert inst[52, 34] != 0, "4 px hole must be filled"
    assert create_instances_from_semantic(seg, min_size=12)[52, 34] == 0, "Hole as big as a quarter of min_size stays"
    assert inst[50, 35] != inst[50, 62], "Touching discs must get different instance IDs"

    # Filtering and splitting are independent: keep touching objects merged, but still remove the blob
    inst = create_instances_from_semantic(seg, min_size=100, split_touching=False)
    assert np.array_equal(np.unique(inst), [0, 1]) # discs merged
    assert inst[11, 81] == 0, "16 px blob must still be removed"

    # An object with exactly min_size pixels is kept, one pixel less is removed
    assert create_instances_from_semantic(seg, min_size=16)[11, 81] != 0
    assert create_instances_from_semantic(seg, min_size=17)[11, 81] == 0

    # Explicitly given classes still skip 1 (background); missing classes yield nothing
    assert np.array_equal(np.unique(create_instances_from_semantic(seg, min_size=100, classes=[1, 2])), [0, 1, 2])
    assert np.array_equal(np.unique(create_instances_from_semantic(seg, min_size=100, classes=[3])), [0])

def test_create_instances_from_semantic_3d_and_list():
    seg = _two_discs_seg()
    seg3d = np.stack([seg, seg])

    # 3D as a whole: objects connected across planes are one instance
    inst = create_instances_from_semantic(seg3d, min_size=100)
    assert inst.shape == seg3d.shape
    assert np.array_equal(np.unique(inst), [0, 1, 2])

    # 3D per plane: unique IDs across planes
    inst = create_instances_from_semantic(seg3d, min_size=100, per_plane=True)
    assert np.array_equal(np.unique(inst), [0, 1, 2, 3, 4])
    assert set(np.unique(inst[0])).isdisjoint(set(np.unique(inst[1])) - {0})

    # List input: list output with unique IDs across the segmentations
    insts = create_instances_from_semantic([seg, seg], min_size=100)
    assert isinstance(insts, list) and len(insts) == 2
    assert np.array_equal(np.unique(insts[0]), [0, 1, 2])
    assert np.array_equal(np.unique(insts[1]), [0, 3, 4])

    # An image without objects of a class must not reset the running ID (unique IDs across the list)
    other = np.ones_like(seg); other[20:40, 20:40] = 3
    insts = create_instances_from_semantic([seg, other], min_size=100, warn=False)
    assert set(np.unique(insts[0])).isdisjoint(set(np.unique(insts[1])) - {0})

    # per_plane filters per plane: a column too small in every plane is removed, although it is big in 3D
    column = np.ones((5, 20, 20), dtype=np.uint8); column[:, 5:13, 5:13] = 2 # 64 px per plane, 320 in 3D
    assert (create_instances_from_semantic(column, min_size=100, per_plane=True) > 0).sum() == 0
    assert (create_instances_from_semantic(column, min_size=100) > 0).sum() == 320

    # Class 0 (unlabelled) is skipped like the background, and the inputs are checked
    unlabelled = np.where(seg == 1, 0, seg) # background 1 -> 0, objects stay class 2
    assert np.array_equal(np.unique(create_instances_from_semantic(unlabelled, min_size=100, warn=False)), [0, 1, 2])
    with pytest.raises(ValueError):
        create_instances_from_semantic(seg, min_size=-1)
    assert create_instances_from_semantic([], warn=False) == []

    # No background class 1 present -> warning (unless warn=False)
    with pytest.warns(UserWarning, match="No class 1"):
        create_instances_from_semantic(seg + 1, min_size=100)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        create_instances_from_semantic(seg + 1, min_size=100, warn=False)

def test_instance_size_filter_keywords():
    """The scikit-image < 0.26 fallback must ask for the same sizes as the current API."""
    from napari_convpaint import utils

    calls = []
    def fake(mask, **kwargs):
        calls.append(kwargs)
        return mask
    installed = utils._SKIMAGE_MAX_SIZE
    try:
        for use_max_size, expected in ((True, [{'max_size': 24}, {'max_size': 99}]),
                                       (False, [{'area_threshold': 25}, {'min_size': 100}])):
            calls.clear()
            utils._SKIMAGE_MAX_SIZE = use_max_size
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(utils.morph, 'remove_small_holes', fake)
                mp.setattr(utils.morph, 'remove_small_objects', fake)
                utils._fill_holes_up_to(np.zeros((4, 4), dtype=bool), 24) # Holes of at most 24 px
                utils._remove_objects_below(np.zeros((4, 4), dtype=bool), 100) # Objects below 100 px
            assert calls == expected
    finally:
        utils._SKIMAGE_MAX_SIZE = installed
