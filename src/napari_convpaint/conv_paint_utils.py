import warnings
import torch
import numpy as np
from scipy.ndimage import gaussian_filter
# from scipy.ndimage import median_filter
import skimage.transform
import skimage.morphology as morph
from joblib import Parallel, delayed
import einops as ein
#import xgboost as xgb
from torch.nn.functional import interpolate as torch_interpolate
from matplotlib import pyplot as plt
import os
import requests


### MODEL DOWNLOAD

def guided_model_download(model_file: str, model_url: str, model_dir: str = None) -> str:
    """
    Downloads a model file with progress indication.
    Uses Napari UI if available, otherwise falls back to CLI.
    
    Parameters:
    ----------
    model_file : str
        Filename to save (e.g. "dinov2_vits14_reg.pth").
    model_url : str
        Direct URL to the model.
    model_dir : str
        Directory to save the model file (default: torch hub cache).

    Returns:
    ----------
    model_path : str
        Full path to the downloaded model file.
    """
    if model_dir is None:
        model_dir = os.path.expanduser('~/.cache/torch/hub/checkpoints')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_file)

    if os.path.exists(model_path):
        return model_path  # Already downloaded

    # Try importing Napari tools
    use_napari = False
    try:
        from napari.utils.progress import progress as napari_progress
        from napari.utils.notifications import show_info, show_error
        import napari
        if napari.current_viewer():
            use_napari = True
    except ImportError:
        pass  # Fall back to CLI mode

    try:
        if use_napari:
            show_info(f"🔄 Downloading model weights: {model_file}")

        with requests.get(model_url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            chunk_size = 8192
            num_chunks = total // chunk_size + 1

            if use_napari:
                progress_bar = napari_progress(total=num_chunks, desc=f"Downloading {model_file}")
            else:
                print(f"Downloading {model_file} ({total / 1e6:.2f} MB) from {model_url}...")
                progress_bar = range(num_chunks)

            with open(model_path, 'wb') as f:
                for i, chunk in enumerate(r.iter_content(chunk_size=chunk_size)):
                    if chunk:
                        f.write(chunk)
                    if use_napari:
                        progress_bar.update(1)
                    elif i % 50 == 0 or i == num_chunks - 1:
                        print(f"  {min((i + 1) * chunk_size / total * 100, 100):.1f}% done", end="\r")

        # Confirm completion
        if use_napari:
            show_info(f"✅ Download completed: {model_file}")

    except Exception as e:
        if use_napari:
            show_error(
                f"❌ Download failed: {e}\n\n"
                f"Please manually download {model_file} from:\n{model_url}\n"
                f"And place it in:\n{model_dir}"
            )
        else:
            print(f"\n❌ Download failed: {e}")
            print(f"Please manually download from:\n{model_url}")
            print(f"And place it in:\n{model_dir}")
        raise RuntimeError(f"Model download failed: {e}")

    return model_path


### SCALING AND RESCALING

def scale_img(image, scaling_factor, upscale=False, input_type="img"):
    """
    Downscale an image by averaging over non-overlapping blocks of the specified size.
    OR Upscale by repeating the pixels.
    
    Parameters:
    ----------
    image : np.ndarray
        Input array to be downscaled. Must have spatial dimensions as the last two dimensions.
    scaling_factor : int
        Factor by which to downscale the image.
    upscale : bool
        If True, upscale the image by repeating the pixels.
    input_type : str ("img", "labels", "coords")
        Type of the input image. Determines how to scale the image:
        If "img", use median, if "labels", use mode, if "coords", use max.

    Returns:
    ----------
    downscaled_img / upscaled_img : np.ndarray
        Downscaled array or upscaled array.
    """
    # If the scaling factor is such that de facto no scaling is needed, return the image as is
    if scaling_factor in (-1, 0, 1):
        return image

    # If option to UPSCALE is True, return the image upscaled

    # Upscale is the same for all input types
    if upscale:
        # Upscale by duplicating elements
        upscaled_img = np.repeat(
            np.repeat(image, scaling_factor, axis=-2), # Repeat along the height
            scaling_factor, axis=-1                    # Repeat along the width
            )
        return upscaled_img
    
    # Else DOWNSCALE the image

    # IMAGES by gaussian filter and striding
    if input_type == 'img':
        # Apply a small Gaussian blur to avoid aliasing
        sigma = 0.4 * scaling_factor
        sigma = [0] * (image.ndim - 2) + [sigma, sigma]  # Add zeros for batch and channel dimensions
        blurred_img = gaussian_filter(image, sigma=sigma)  # assuming shape (..., H, W)
        # Downsample by striding
        H, W = image.shape[-2:]
        start_h = (H % scaling_factor) // 2 # Move the start such that the image is centered
        start_w = (W % scaling_factor) // 2 # Move the start such that the image is centered
        scaled_img = blurred_img[..., start_h::scaling_factor, start_w::scaling_factor]
        # from matplotlib import pyplot as plt
        # fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        # ax[0].imshow(image[0,0,...], cmap='gray')
        # ax[1].imshow(blurred_img[0,0,...], cmap='gray')
        # ax[2].imshow(scaled_img[0,0,...], cmap='gray')
        # plt.show()
        return scaled_img

    # For LABELS and COORDINATES, we slice/stack the image to apply the downscaling along new axes
    # First, we pad to the next multiple of the scaling factor, distributing on each side (with 1 pixel more on bottom/right if uneven)
    if image.shape[-2] % scaling_factor != 0 or image.shape[-1] % scaling_factor != 0:
        # Calculate padding sizes
        pad_h = (scaling_factor - (image.shape[-2] % scaling_factor)) % scaling_factor
        pad_w = (scaling_factor - (image.shape[-1] % scaling_factor)) % scaling_factor
        pad_top, pad_left = pad_h // 2, pad_w // 2
        pad_bot, pad_right = pad_h - pad_top, pad_w - pad_left
        # Pad the image
        image = pad(image, (pad_top, pad_bot, pad_left, pad_right), input_type=input_type)
    # Slice the last two dimensions
    slice_start = (0, 0) #((image.shape[-2] % scaling_factor) // 2,
                    #  (image.shape[-1] % scaling_factor) // 2)
    slice_size = (image.shape[-2] // scaling_factor * scaling_factor, # These should now be equal to image shape, since we padded ...
                image.shape[-1] // scaling_factor * scaling_factor) # ... to the next multiple of the scaling factor

    if input_type == 'labels':
        classes_before = np.unique(image)

        output_shape = (
            image.shape[0],
            *image.shape[1:-2],
            slice_size[0] // scaling_factor,
            slice_size[1] // scaling_factor
        )
        scaled_img = np.zeros(output_shape, dtype=image.dtype)

        for i in range(image.shape[0]):
            plane = image[i]
            if np.all(plane == 0):
                continue  # skip empty plane
            sliced_plane = plane[
                ..., 
                slice_start[0]:slice_start[0] + slice_size[0],
                slice_start[1]:slice_start[1] + slice_size[1]
            ]
            blocks = sliced_plane.reshape(
                *plane.shape[:-2],
                slice_size[0] // scaling_factor, scaling_factor,
                slice_size[1] // scaling_factor, scaling_factor
            )
            scaled_img[i] = fast_mode(blocks, axis=(-3, -1))

        classes_after = np.unique(scaled_img)
        if len(classes_before) != len(classes_after):
            warnings.warn(f"Classes have changed after downscaling from {classes_before} to {classes_after}.")

        # from matplotlib import pyplot as plt
        # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # ax[0].imshow(image[0,...], cmap='gray')
        # ax[1].imshow(scaled_img[0,...], cmap='gray')
        # plt.show()
        return scaled_img

    elif input_type == 'coords':
        sliced_img = image[
            ..., 
            slice_start[0]:slice_start[0] + slice_size[0],
            slice_start[1]:slice_start[1] + slice_size[1]
        ]
        # Reshape to group elements for downscaling
        stacked_blocks = sliced_img.reshape(
            *image.shape[:-2],  # Preserve all leading dimensions
            slice_size[0] // scaling_factor, scaling_factor,
            slice_size[1] // scaling_factor, scaling_factor
        )
        scaled_img = np.max(stacked_blocks, axis=(-3, -1)) # Take max so we ignore the padding with -1 values
        return scaled_img

    else:
        raise ValueError(f"Unknown input type: {input_type}. Supported types are 'img', 'labels', and 'coords'.")
    
def fast_mode(arr, axis):
    """
    Fast mode pooling assuming integer values >= 0
    """
    if isinstance(axis, int):
        axis = (axis,)
    axis = tuple(a + arr.ndim if a < 0 else a for a in axis)  # Transform negative indices to positive indices
    transpose_axes = [a for a in range(arr.ndim) if a not in axis] + list(axis)
    transposed = arr.transpose(transpose_axes)
    flat_main = transposed.reshape(-1, np.prod([arr.shape[a] for a in axis]))
    result = np.apply_along_axis(lambda x: np.bincount(x).argmax(), 1, flat_main)
    out_shape = [arr.shape[a] for a in range(arr.ndim) if a not in axis]
    result = result.reshape(out_shape)
    return result

def rescale_features(feature_img, target_shape, order=1):
    """
    Rescale an image to the specified target size. target_shape is a tuple of (H, W).

    Parameters:
    ----------
    feature_img : np.ndarray or torch.Tensor
        Input image to be rescaled. Must have spatial dimensions as the last two dimensions.
    target_shape : tuple of int
        Target size to rescale the image to. Must be a tuple of (H, W).
    order : int
        Order of the spline interpolation. Default is 1 (bilinear interpolation).

    Returns:
    ----------
    rescaled_img : np.ndarray or torch.Tensor
        Rescaled image.
    """
    output_shape = (feature_img.shape[0], target_shape[1], target_shape[2], target_shape[3])

    if feature_img.shape == output_shape:
        return feature_img
    
    if isinstance(feature_img, torch.Tensor):
        # If the input is a PyTorch tensor, use the faster torch interpolation
        int_mode = 'bilinear' if order > 0 else 'nearest'
        align_corners = False if order > 0 else None
        return torch_interpolate(feature_img, size=output_shape[2:],
                                  mode=int_mode, align_corners=align_corners)
    else:
        return skimage.transform.resize(feature_img, output_shape, order=order, mode='reflect', preserve_range=True)

def rescale_class_labels(label_img, output_shape):
    """
    Rescale a class label image to the specified output size.

    Parameters:
    ----------
    label_img : np.ndarray
        Input label image to be rescaled. Must have spatial dimensions as the last two dimensions.
    output_shape : tuple of int
        Target size to rescale the image to. Must be a tuple of (H, W).
    
    Returns:
    ----------
    rescaled_label_img : np.ndarray
        Rescaled label image.
    """
    rescaled_label_img = skimage.transform.resize(label_img, output_shape, order=0, mode='reflect', preserve_range=True).astype(np.uint8)
    return rescaled_label_img

def rescale_outputs(output_img, output_shape, order=0):
    """
    Rescale a class probability or feature image to the specified output size.

    Parameters:
    ----------
    output_img : np.ndarray
        Class probabilities or feature image to be rescaled.
        Must have spatial dimensions as the last two dimensions.
    output_shape : tuple of int
        Target size to rescale the image to. Must be a tuple of (H, W).

    Returns:
    ----------
    rescaled_output : np.ndarray
        Rescaled class probability or feature image.
    """
    rescaled_output = skimage.transform.resize(output_img, output_shape, order=order, mode='reflect', preserve_range=True)
    return rescaled_output


### CROPPING, PADDING, TILING & ANNOTATION EXTRACTION

def reduce_to_patch_multiple(input, patch_size):
    """
    Reduces the input to the patch size multiple.
    Assumes that the input has spatial dimensions in last two dimensions.
    In cases of uneven patch size, the bottom and right crops will be 1 pixel larger than top and left, respectively.

    Parameters:
    ----------
    input : np.ndarray
        Input array to be reduced. Must have spatial dimensions as the last two dimensions.
    patch_size : int
        Patch size to reduce the input to.

    Returns:
    ----------
    reduced_input : np.ndarray
        Reduced input array.
    """
    if patch_size == 1:
        return input
    # Get spatial dimensions
    h, w = input.shape[-2:]
    # What is the next smaller multiple of the patch size?
    new_h = (h // patch_size) * patch_size
    new_w = (w // patch_size) * patch_size
    # If the input is already a multiple of the patch size, return it
    if new_h == h and new_w == w:
        return input
    # Otherwise, reduce the input to the patch size multiple
    h_crop_top = (h - new_h)//2
    w_crop_left = (w - new_w)//2
    h_crop_bottom = h - new_h - h_crop_top # in uneven cases, bottom crop will be 1 larger than top crop
    w_crop_right = w - new_w - w_crop_left # in uneven cases, right crop will be 1 larger than left crop
    # Transform bottom and right crop for slicing; also make sure we can handle crops of size 0
    h_idx_bottom = -h_crop_bottom if h_crop_bottom > 0 else None
    w_idx_right = -w_crop_right if w_crop_right > 0 else None
    input = input[..., h_crop_top:h_idx_bottom, w_crop_left:w_idx_right]
    return input

def pad(img, padding, input_type='img'):
    """
    Pads the image with the padding size given.
    The padding is done with the 'reflect' mode for images and 'constant' for annotations.
    Dimensions are assumed [C, Z, H, W] and [Z, H, W] for images and annotations, respectively.

    Parameters:
    ----------
    img : np.ndarray
        Input image to be padded. Must have spatial dimensions as the last two dimensions.
    padding : int or tuple
        Padding size to be added to the image.
        If an int is provided, it is applied to all sides.
        If a tuple is provided, it should be of the form (pad_top, pad_bottom, pad_left, pad_right).
    input_type : str
        Type of the input image. Determines how to pad the image:
        If "img", use 'reflect' padding, if "labels", use 'constant' padding with 0, if "coords", use 'constant' padding with -1.

    Returns:
    ----------
    padded_img : np.ndarray
        Padded image.
    """
    if isinstance(padding, int):
        padding = ((padding, padding), (padding, padding))
    elif isinstance(padding, tuple) and len(padding) == 2 and isinstance(padding[0], int) and isinstance(padding[1], int):
        # If padding is a tuple of two ints, apply it to both height and width
        padding = ((padding[0], padding[0]), (padding[1], padding[1]))
    elif isinstance(padding, tuple) and len(padding) == 4 and all(isinstance(p, int) for p in padding):
        # If padding is a tuple of four ints, apply it to all sides
        padding = ((padding[0], padding[1]), (padding[2], padding[3]))
    elif isinstance(padding, tuple) and len(padding) == 2 and isinstance(padding[0], tuple) and isinstance(padding[1], tuple):
        # If padding is a tuple of two tuples, apply it to height and width separately
        padding = padding

    # Adjust the padding to the input
    padding = ((0, 0),) * (img.ndim - 2) + padding  # Add batch and channel dimensions if necessary
    # Pad the image with the padding size
    if input_type == 'img':
        # For images, use 'reflect' padding
        return np.pad(img, padding, mode='reflect')
    elif input_type == 'labels':
        # For labels, use 'constant' padding with 0
        return np.pad(img, padding, mode='constant')
    elif input_type == 'coords':
        # For coordinates, use 'constant' padding with -1
        return np.pad(img, padding, mode='constant', constant_values=-1)
    
def pad_to_shape(feat, target_shape):
    """
    Pads the spatial dimensions (last two) of `feat` symmetrically to match `target_shape`.
    In case of an odd number of pixels to add, bottom and right receive 1 more pixel than top and left.

    This convention ensures alignment with reductions like `reduce_to_patch_multiple()`,
    which crop more from bottom and right. Using this padding ensures spatial consistency
    for later reconstruction or upsampling steps.

    Parameters
    ----------
    feat : np.ndarray
        Input array with spatial dimensions in the last two axes.
    target_shape : tuple
        Desired shape for the last two dimensions: (..., H_target, W_target)

    Returns
    -------
    np.ndarray
        Padded array matching the target shape.
    """
    pad = [(0, 0), (0, 0)]  # batch and channel dims
    for i in range(2):  # spatial dims (height, width)
        current = feat.shape[2 + i]
        target = target_shape[i]
        diff = max(target - current, 0)
        pad_before = diff // 2
        pad_after = diff - pad_before  # ensures bottom/right get the extra pixel if diff is odd
        pad.append((pad_before, pad_after))
    return np.pad(feat, pad, mode='constant')


def crop_to_shape(arr, target_shape):
    """
    Crops the spatial dimensions (last two) of `arr` symmetrically to match `target_shape`.
    In case of odd number of pixels to remove, top and left crops are 1 larger than bottom and right
    to reverse a prior reduction that removed more from bottom/right (from reduce_to_patch_multiple).

    Parameters
    ----------
    arr : np.ndarray
        Input array with spatial dimensions in the last two axes.
    target_shape : tuple
        Desired shape for the last two dimensions: (..., H_target, W_target)

    Returns
    -------
    np.ndarray
        Cropped array matching the target shape.
    """
    current_h, current_w = arr.shape[-2:]
    target_h, target_w = target_shape[-2:]

    assert target_h <= current_h and target_w <= current_w, \
        f"Target shape {target_shape[-2:]} must be <= current shape {arr.shape[-2:]}"

    crop_h = current_h - target_h
    crop_w = current_w - target_w

    crop_top = (crop_h + 1) // 2  # top gets more when odd
    crop_bottom = crop_h - crop_top
    crop_left = (crop_w + 1) // 2  # left gets more when odd
    crop_right = crop_w - crop_left

    h_end = -crop_bottom if crop_bottom > 0 else None
    w_end = -crop_right if crop_right > 0 else None

    return arr[..., crop_top:h_end, crop_left:w_end]


def get_annot_planes(img, annot=None, coords=None):
    """
    Extracts the planes (of the z dimension) of the padded data, annotations and cordinates
    where there is at least one annotation.
    Dimensions are assumed [C, Z, H, W] and [Z, H, W] for images and annotations, respectively.

    Parameters:
    ----------
    img : np.ndarray
        Input image to extract planes from. Must have spatial dimensions as the last two dimensions.
    annot : np.ndarray, optional
        Input annotation to extract planes from. Must have spatial dimensions as the last two dimensions.
        If None, all image planes and no annotation planes are returned.

    Returns:
    ----------
    img_planes : list of np.ndarray
        List of image planes that contain annotations. If annot is None, all image planes are returned.
    annot_planes : list of np.ndarray or None
        List of annotation planes from the input annotation. If annot is None, None is returned.
    coords : np.ndarray or None
        List of coordinates planes from the input coordinates. If coords is None, None is returned.
    """
    if annot is None:
        img_planes = [img[:, z:z+1] for z in range(img.shape[1])]
        return img_planes, None, coords
    
    non_empty = np.unique(np.where(annot > 0)[0])
    if len(non_empty) == 0:
        warnings.warn('No annotations found')
        return None, None, None
    
    # Extract the planes of the image and annotations (and coordinates if provided)
    img_planes = [img[:,z:z+1] for z in non_empty] # images have a channels dimension
    annot_planes = [annot[z:z+1] for z in non_empty] # annotations do not have a channels dimension
    if coords is not None:
        coords_planes = [coords[:,z:z+1] for z in non_empty]
    else:
        coords_planes = None
    return img_planes, annot_planes, coords_planes

def tile_annot(img, annot, coords, padding, plot_tiles=False):
    """
    Tile the image and annotations into patches of the size of the annotations.
    Takes a number of pixels equal to 'padding' more than the bounding box of the annotation.
    Dimensions are assumed [C, Z, H, W] and [Z, H, W] for images and annotations, respectively.

    Parameters:
    ----------
    img : np.ndarray
        Input image to be tiled. Must have spatial dimensions as the last two dimensions.
    annot : np.ndarray
        Input annotation to use for tiling. Must have spatial dimensions as the last two dimensions.
    padding : int or tuple
        Padding size to be added to the bounding box of the annotation.
        If an int is provided, it is applied to all sides.
        If a tuple is provided, it should be of the form (pad_top, pad_bottom, pad_left, pad_right).

    Returns:
    ----------
    img_tiles : list of np.ndarray
        List of image tiles that contain the annotations.
    annot_tiles : list of np.ndarray
        List of annotation tiles that contain the annotations.
    """
    if isinstance(padding, int):
        pad_top, pad_bottom, pad_left, pad_right = (padding, padding, padding, padding)
    elif isinstance(padding, tuple) and len(padding) == 2 and len(padding[0]) == 2:
        pad_top, pad_bottom, pad_left, pad_right = (side for dim in padding for side in dim)
    elif isinstance(padding, tuple) and len(padding) == 4:
        pad_top, pad_bottom, pad_left, pad_right = padding
    elif isinstance(padding, tuple) and len(padding) == 2 and isinstance(padding[0], int):
        pad_top, pad_left = padding # Use first number for vertical padding, second for horizontal
        pad_bottom, pad_right = padding # And pad the same on both sides
    else:
        raise ValueError(f"Padding must be an int or a tuple of 2 or 4 ints, got {padding}.")
    
    # Find the bounding boxes of the annotations
    annot_regions = skimage.morphology.label(annot > 0)
    regions = skimage.measure.regionprops(annot_regions)

    # Create lists to store the tiles
    img_tiles = []
    annot_tiles = []
    coord_tiles = []

    if plot_tiles:
        im_to_show = img[0,0,...].copy()
        im_to_show[annot[0]>0] = 0
    
    for region in regions:
        # Get the bounding box of the annotation
        z_min, y_min, x_min, z_max, y_max, x_max = region.bbox

        if plot_tiles:
            # Draw the bounding box on the image
            im_to_show[y_min:y_max, x_min] = 1
            im_to_show[y_min:y_max, x_max-1] = 1
            im_to_show[y_min, x_min:x_max] = 1
            im_to_show[y_max-1, x_min:x_max] = 1

        y_min -= pad_top
        y_max += pad_bottom
        x_min -= pad_left
        x_max += pad_right

        if plot_tiles:
            # Draw the bounding box WITH PADDING on the image
            im_to_show[y_min:y_max, x_min] = 2
            im_to_show[y_min:y_max, x_max-1] = 2
            im_to_show[y_min, x_min:x_max] = 2
            im_to_show[y_max-1, x_min:x_max] = 2

        img_tile = img[..., y_min:y_max, x_min:x_max]
        annot_tile = annot[..., y_min:y_max, x_min:x_max]

        # Zero-out all labels except this region
        mask = annot_regions[..., y_min:y_max, x_min:x_max] == region.label
        annot_tile = annot_tile * mask

        if coords is not None:
            coords_tile = coords[..., y_min:y_max, x_min:x_max]
        else:
            coords_tile = None

        img_tiles.append(img_tile)
        annot_tiles.append(annot_tile)
        coord_tiles.append(coords_tile)
    
    if plot_tiles:
            plt.imshow(im_to_show, cmap='gray')
            plt.show()

    return img_tiles, annot_tiles, coord_tiles

def get_coordinates_image(img):
    """
    Get the coordinates of the pixels in the image.
    The coordinates are returned as a 3D array of shape (Z, H, W) where each element is a tuple
    of the form (z, h, w) representing the coordinates of the pixel in the image.
    Dimensions are assumed [C, Z, H, W] for images.
    
    Parameters:
    ----------
    img : np.ndarray
        Input image to extract coordinates from. Must have spatial dimensions as the last two dimensions.
    Returns:
    ----------
    coords : np.ndarray
        3D array of shape (Z, H, W) where each element is a tuple of the form (z, h, w).
    """
    # Check if the image has the correct number of dimensions
    if img.ndim != 4:
        raise ValueError("Input image must have 4 dimensions (C, Z, H, W).")
    # Get spatial dimensions (remember, channels are first)
    Z, H, W = img.shape[1:4]

    # Create a 3D array of coordinates
    # Each element is a tuple of the form (z, h, w)
    # coords = np.empty((Z, H, W), dtype=object)
    # for z in range(Z):
    #     for h in range(H):
    #         for w in range(W):
    #             coords[z, h, w] = (z, h, w)

    # Generate the coordinate grid
    z_coords, h_coords, w_coords = np.meshgrid(
        np.arange(Z), np.arange(H), np.arange(W), indexing='ij'
    )
    # Stack them into a coordinate array of shape (3, Z, H, W)
    coords = np.stack((z_coords, h_coords, w_coords), axis=0)

    return coords

def get_features_targets(features, annot):
    """
    Given a set of features and targets, extract the annotated pixels and targets,
    and concatenate each.

    Parameters:
    ----------
    features : np.ndarray
        Input features to extract. Must have spatial dimensions as the last two dimensions.
    annot : np.ndarray
        Input annotation to use for extraction. Must have spatial dimensions as the last two dimensions.
    
    Returns:
    ----------
    features : np.ndarray
        Extracted features from the input features. Shape is (num_pixels, num_features).
    annot : np.ndarray
        Extracted targets from the input annotation. Shape is (num_pixels,).
    """
    # Get the annotated pixels and targets
    mask = annot > 0
    features = np.moveaxis(features, 0, -1) #move [z, h, w, features]
    # Select only the pixels that are annotated, linearizing them
    features = features[mask] # Get the features
    annot = annot[mask] # Get the targets

    return features, annot


### DEVICE & NORMALIZATION

def get_device(use_gpu=None):
    """
    Get the device to use for PyTorch operations.
    If CUDA is available, use the first available GPU. If MPS is available, use it.

    Parameters:
    ----------
    use_gpu : bool, optional
        If True, use GPU if available. If False, use CPU. Default is None.

    Returns:
    ----------
    device : torch.device
        The device to use for PyTorch operations.
    """
    if use_gpu:
        if torch.cuda.is_available():
            return torch.device('cuda:0')  # use first available GPU
        elif torch.backends.mps.is_available(): #check if mps is available
            return torch.device('mps')
        else:
            warnings.warn('CUDA or MPS is not available. Setting device to CPU.')
    return torch.device('cpu')

def get_device_from_torch_model(model):
    """
    Get the device from a PyTorch model.
    
    Parameters:
    ----------
    model : torch.nn.Module
        The PyTorch model to get the device from.

    Returns:
        ----------
        device : torch.device
            The device the model is on.
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        try:
            return next(model.buffers()).device
        except StopIteration:
            return torch.device("unknown")

def get_catboost_device(use_gpu=None):
    """
    Get the device to use for CatBoost operations.
    If CUDA is available, use the first available GPU. If MPS is available, use it.

    Parameters:
    ----------
    use_gpu : bool, optional
        If True, use CUDA if available. If False, use CPU. Default is None.

    Returns:
    ----------
    device : str
        The device to use for CatBoost operations.
    """
    if use_gpu:
        if torch.cuda.is_available():
            return 'GPU'
        else:
            warnings.warn('CUDA is not available. Using CPU for CatBoost.')
    return 'CPU'

def normalize_image_percentile(image: np.ndarray) -> np.ndarray:
    """
    Linearly rescale image intensities so that, per-channel,
    the 1st percentile → 0.0 and the 99th percentile → 1.0.

    Any values below the 1st percentile become 0.0; above the 99th
    become 1.0.

    Parameters:
    ----------
    image: np.ndarray with 2-4 dims.
        - 2D: treated as single-channel.
        - 3D or 4D: first axis is channels.

    Returns:
    ----------
    np.ndarray, dtype float32, same shape as input.

    Raises:
    ----------
    ValueError: if input.ndim not in [2,4].
    """
    if image.ndim < 2 or image.ndim > 4:
        raise ValueError(f"Image must have 2-4 dimensions (got {image.ndim})")

    # Prepare output
    out = np.empty_like(image, dtype=np.float32)

    if image.ndim == 2:
        # single-channel case
        p1, p99 = np.percentile(image, [1, 99])
        span = max(p99 - p1, 1e-6)
        out = (image.astype(np.float32) - p1) / span
        return np.clip(out, 0.0, 1.0)

    # multi-channel case
    # image.shape[0] = number of channels
    C = image.shape[0]
    # for each channel, compute percentiles over all other dims
    for c in range(C):
        arr = image[c].astype(np.float32)
        p1, p99 = np.percentile(arr, [1, 99])
        span = max(p99 - p1, 1e-6)
        channel_norm = (arr - p1) / span
        out[c] = np.clip(channel_norm, 0.0, 1.0)

    return out

def normalize_image(image, image_mean, image_std):
    """
    Normalize a numpy array with 2-4 dimensions.

    If the array has multiple channels (determined by the multi_channel_img flag), 
    each channel is normalized independently. If there are multiple time or z frames, 
    they are normalized together.

    Parameters:
    ----------
    image : np.ndarray
        Input array to be normalized. Must have 2-4 dimensions.
    image_mean : float or nd.array
        Mean of the input array along non-channel dimensions.
    image_std : float or nd.array
        Standard deviation of the input array along non-channel dimensions.
        
    Returns:
    ----------
    arr_norm : np.ndarray
        Normalized array.

    Raises
    ------
    ValueError
        If the input array does not have 2-4 dimensions.
    """
    
    if image.ndim < 2 or image.ndim > 4:
        raise ValueError("Array must have 2-4 dimensions")
    
    # Avoid division by zero
    image_std = np.maximum(image_std, 1e-6)

    img_norm = (image - image_mean) / image_std

    return img_norm


def normalize_image_imagenet(image):
    """
    Normalize a numpy array or torch tensor image to ImageNet stats.

    For numpy (np.ndarray):
      1) Cast to float32.
      2) Bring values into [0,1] by:
         - dividing uint8 by 255
         - dividing uint16 by 65535
         - otherwise min-max scaling floats
      3) Applies per-channel ImageNet mean/std.

    For torch (torch.Tensor):
      1) Cast to float32.
      2) Bring values into [0,1] by:
         - dividing uint8 by 255
         - dividing uint16 by 65535
         - otherwise min-max scaling floats (with clamp_min)
      3) Applies per-channel ImageNet mean/std on the same device.

    Parameters:
    ----------
    image: np.ndarray or torch.Tensor of shape [3, H, W] or [3, Z, H, W] or the same with 1 channel,
            dtype uint8, uint16, or float.

    Returns:
    ----------
    np.ndarray or torch.Tensor: Same type as input (np.ndarray or torch.Tensor), same shape,
        unless if single channel image is given, it is repeated on first axis and normalized as RGB
        dtype float32, normalized so each channel has ImageNet mean [0.485,0.456,0.406] and
        std [0.229,0.224,0.225].

    Raises:
    ----------
    ValueError: if input is not np.ndarray or torch.Tensor, if ndim not in (3,4),
                or if channel dimension != 3 or 1.
    """
    # dispatch on type
    if isinstance(image, np.ndarray):
        # ---------- numpy path ----------
        x = image.astype(np.float32)
        if image.ndim not in (3, 4):
            raise ValueError(f"Image must have 3 or 4 dimensions (got ndim={image.ndim})")
        if image.shape[0] not in [3, 1]:
            raise ValueError(f"First dimension must be 3 or 1 channels (got {image.shape[0]})")

        # bring into [0,1]
        if image.dtype == np.uint8:
            x = x / 255.0
        elif image.dtype == np.uint16:
            x = x / 65535.0
        else:
            mn, mx = x.min(), x.max()
            denom = max(mx - mn, 1e-6)
            x = (x - mn) / denom

        # Repeat single channel if necessary
        if image.shape[0] == 1:
            x = np.repeat(x, 3, axis=0)

        # ImageNet stats
        spatial_dims = x.ndim - 1
        shape = (3,) + (1,) * spatial_dims
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(shape)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(shape)

        return (x - mean) / std

    elif isinstance(image, torch.Tensor):
        # ---------- torch path ----------
        if image.ndim not in (3, 4):
            raise ValueError(f"Image must have 3 or 4 dimensions (got ndim={image.ndim})")
        if image.shape[0] not in [3, 1]:
            raise ValueError(f"First dimension must be 3 or 1 channels (got {image.shape[0]})")

        device = image.device
        x = image.float()

        if image.dtype == torch.uint8:
            x = x / 255.0
        elif image.dtype == torch.uint16:
            x = x / 65535.0
        else:
            mn = x.amin()
            mx = x.amax()
            denom = (mx - mn).clamp_min(1e-6)
            x = (x - mn) / denom
        
        if image.shape[0] == 1:
            x = x.repeat(3, *[1] * (x.ndim - 1))

        spatial_dims = x.ndim - 1
        shape = (3,) + (1,) * spatial_dims
        mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float32).reshape(shape)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float32).reshape(shape)

        return (x - mean) / std

    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    
def compute_image_stats(image, ignore_n_first_dims=None):
    """
    Compute mean and standard deviation of a numpy array with 2-4 dimensions.

    If the array has multiple channels (determined by the first_dim_is_channel option), 
    each channel is normalized independently. Channels can only be the first dimension of the array.

    Parameters:
    ----------
    image : np.ndarray
        Input array to be normalized. Must have 2-4 dimensions.
    ignore_n_first_dims : int
        Number of first dimensions to ignore when computing the mean and standard deviation.
        For example if the input array has 3 dimensions CHW and ignore_n_first_dims=1, the mean
        and standard deviation will be computed for each channel C.
        
    Returns:
    ----------
    image_mean : float or nd.array
        Mean of the input array along non-channel dimensions
    image_std : float or nd.array
        Standard deviation of the input array along non-channel dimensions

    Raises
    ------
    ValueError
        If the input array does not have 2-4 dimensions.
    """
    
    if image.ndim < 2 or image.ndim > 4:
        raise ValueError("Array must have 2-4 dimensions")
    
    if ignore_n_first_dims is None:
        image_mean = image.mean()
        image_std = image.std()
    else:
        image_mean = image.mean(axis=tuple(range(ignore_n_first_dims, image.ndim)), keepdims=True)
        image_std = image.std(axis=tuple(range(ignore_n_first_dims, image.ndim)), keepdims=True)

    return image_mean, image_std


### ANNOTATION HANDLING (optional)

def get_balanced_mask(ref_mask, tgt_mask):
    """
    Given two masks, subsample the second one to have the same number of pixels as the first one.
    Parameters:
    ----------
    ref_mask: np.ndarray
        2d boolean reference mask
    tgt_mask: np.ndarray
        2d boolean target mask

    Returns:
    ----------
    tgt_mask_balanced: 2d boolean np.ndarray
        target mask sub-sampled to have at most the same number of pixels as the reference mask
    """

    tgt_mask_balanced = np.zeros_like(tgt_mask).astype(np.bool_)
    tgt_idx = np.argwhere(tgt_mask)
    np.random.shuffle(tgt_idx)
    tgt_idx = tgt_idx[0:ref_mask.sum()]  # subsample
    tgt_mask_balanced[tgt_idx[:, 0], tgt_idx[:, 1]] = True
    return tgt_mask_balanced

def get_annotation_regions(annotation, d_edge=1):
    """
    Given the annotation returns list of pixel masks for each of the following regions:
    1. all foreground pixels
    2. all not foreground pixels
    3. at most as many randomly selected not-foreground pixels as foreground pixels

    4. signal: union of foreground pixels sharnk by 1 pixel for each of the foreground classes
    5. edge obtained by dilation of 4. by 1+d_edge pixels and than subtracting 4.
    6. background: complement of 4.+5.,
    7. 6. subsampled to at most as many pixels as in 4.

    Parameters:
    ----------
    annotation: 2d np.ndarray
        annotation image, 1 is background, 2,3,... are foreground classes
    d_edge: int
        dilation factor for edge, by default 1

    Returns:
    ----------
    masks: dictionary of 2d boolean np.ndarray
        masks for each of the regions described above ('fg', 'not_fg', 'not_fg_balanced',
         'signal', 'edge', 'bg', 'bg_balanced')
    """

    # 1. all foreground pixels
    fg = annotation > 1

    # 2. all not foreground pixels
    not_fg = annotation == 1

    # 3. at most as many randomly selected not-foreground pixels as foreground pixels
    not_fg_balanced = get_balanced_mask(ref_mask=fg, tgt_mask=not_fg)

    # 4. signal: union of foreground pixels shrank by 1 pixel for each of the foreground classes
    # signal = np.zeros_like(fg).astype(np.bool_)
    # for c in range(2, annotation.max() + 1):
    #    signal = signal | morph.binary_erosion(annotation == c, morph.cross(1))
    # parallel implementation with joblib of the three lines above:
    def erode(c):
        return morph.binary_erosion(annotation == c)

    signal = np.any(Parallel(n_jobs=-1)(delayed(erode)(c) for c in range(2, annotation.max() + 1)), axis=0)

    # 5. edge obtained by dilation of 4. by 1+d_edge pixels and than subtracting 4.
    edge_signal = morph.binary_dilation(signal, morph.disk(1 + d_edge))
    edge = edge_signal & ~signal

    # 6. background: complement of 4.+5.,
    bg = ~edge_signal

    # 7. 6. subsampled to at most as many pixels as in 4.
    bg_balanced = get_balanced_mask(ref_mask=signal, tgt_mask=bg)

    masks = {
        'fg': fg,
        'not_fg': not_fg,
        'not_fg_balanced': not_fg_balanced,
        'signal': signal,
        'edge': edge,
        'bg': bg,
        'bg_balanced': bg_balanced
    }
    return masks

def extract_annotated_pixels(features, annotation, full_annotation=True):
    """
    Given a set of features and an annotation
    if not full_annotation, select all pixels in annotations (values 1,2,...), record same class_id as annotation
    otherwise, select all pixels in annotations (values 2,3...; assumes 1 added to the original [0-bg, 1,2...signal
    annotation values]). targets set for signal: class_id=2, the same number of background pixels class_id=1,
    and the edge pixels class_id=3
    Parameters:
    ----------
    features: 3D np.ndarray (CHW)
        features extracted from image
    annotation:
        annotated image, 1 is background, 2,3,... are foreground classes
    full_annotation: bool
        if True, use all pixels in annotations, otherwise balance background and foreground
        if True - assumes annotations are 1,2,..., where 1 is bg. otherwise assumes 0 is not annotated,
            and 1 is background, 2 is first class, etc.
            Default True


    Returns:
    ----------
    features: 2D np.ndarray (pixel, features)
        features extracted from image
    annotation: 1D np.ndarray (pixel)
        annotation class_id for each pixel
    """

    # select pixels in annotations
    if not full_annotation:
        # select pixels in annotations
        annot_flat = annotation.flatten()
        annot_mask = annot_flat > 0  # something annotated
        features_flat = ein.rearrange(features, 'c h w -> (h w) c')

        features_sel = features_flat[annot_mask]
        targets_sel = annot_flat[annot_mask]
    else:
        # get masks for annotation regions
        masks = get_annotation_regions(annotation)

        # get signal, edge, and balanced background pixels
        signal = masks['signal']
        edge = masks['edge']
        bg_balanced = masks['bg_balanced']

        # select pixels in features
        features_flat = ein.rearrange(features, 'c h w -> (h w) c')
        signal_flat = signal.flatten()
        edge_flat = edge.flatten()
        bg_balanced_flat = bg_balanced.flatten()

        features_signal = features_flat[signal_flat]
        features_edge = features_flat[edge_flat]
        features_bg_balanced = features_flat[bg_balanced_flat]

        # fill selected features and targets
        features_sel = np.concatenate([features_signal, features_edge, features_bg_balanced], axis=0)
        n_pix_signal = features_signal.shape[0]
        n_pix_edge = features_edge.shape[0]
        n_pix_bg_balanced = features_bg_balanced.shape[0]
        targets_sel = np.concatenate([
            2 * np.ones(n_pix_signal),
            3 * np.ones(n_pix_edge),
            1 * np.ones(n_pix_bg_balanced)
        ], axis=0)

    return features_sel, targets_sel