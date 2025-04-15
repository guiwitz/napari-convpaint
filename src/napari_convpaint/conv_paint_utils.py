import warnings
import torch
import numpy as np
from scipy.stats import mode
# from scipy.ndimage import median_filter
import skimage.transform
import skimage.morphology as morph
from joblib import Parallel, delayed
import einops as ein
#import xgboost as xgb
from torch.nn.functional import interpolate as torch_interpolate


### SCALING AND RESCALING

def scale_img(image, scaling_factor, upscale=False, use_labels=False):
    """
    Downscale an image by averaging over non-overlapping blocks of the specified size.
    OR Upscale by repeating the pixels.
    
    Parameters
    ----------
    image : np.ndarray
        Input array to be downscaled. Must have spatial dimensions as the last two dimensions.
    scaling_factor : int
        Factor by which to downscale the image.
    upscale : bool
        If True, upscale the image by repeating the pixels.
    use_labels : bool
        If True, use the mode (majority) instead of the median for downscaling labels.

    Returns
    -------
    downscaled_img / upscaled_img : np.ndarray
        Downscaled array or upscaled array.
    """

    if scaling_factor == 1:
        return image
    
    # If option to UPSCALE is True, return the image upscaled
    if upscale:
        # Upscale by duplicating elements
        upscaled_img = np.repeat(
            np.repeat(image, scaling_factor, axis=-2), # Repeat along the height
            scaling_factor, axis=-1                    # Repeat along the width
            )
        return upscaled_img
    
    # Else DOWNSCALE the image
    # Slice the last two dimensions
    slice_start = ((image.shape[-2] % scaling_factor) // 2,
                     (image.shape[-1] % scaling_factor) // 2)
    slice_size = (image.shape[-2] // scaling_factor * scaling_factor,
                image.shape[-1] // scaling_factor * scaling_factor)
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
    # print(stacked_blocks[0,:,0,:])
    if not use_labels:
        # Take the median along the scaling dimensions
        scaled_img = np.median(stacked_blocks, axis=(-3, -1))
        # scaled_img = median_filter(stacked_blocks, size=(1, scaling_factor, 1, scaling_factor)).reshape(
        #     *image.shape[:-2], slice_size[0] // scaling_factor, slice_size[1] // scaling_factor)
    else:
        classes_before = np.unique(image)
        # For labels, take the majority (max count) along the scaling dimensions
        scaled_img = mode(stacked_blocks, axis=(-3, -1)).mode
        classes_after = np.unique(scaled_img)
        # Check if the classes have changed after downscaling
        if len(classes_before) != len(classes_after):
            warnings.warn(f"Classes have changed after downscaling from {classes_before} to {classes_after}.")
    return scaled_img

def rescale_features(feature_img, target_shape, order=1):
    """
    Rescale an image to the specified target size. target_shape is a tuple of (H, W).
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
    """
    return skimage.transform.resize(label_img, output_shape, order=0, mode='reflect', preserve_range=True).astype(np.uint8)

def rescale_outputs(output_img, output_shape, order=0):
    """
    Rescale a class probability or feature image to the specified output size.
    """
    return skimage.transform.resize(output_img, output_shape, order=order, mode='reflect', preserve_range=True)

def reduce_to_patch_multiple(input, patch_size):
    """
    Reduces the input to the patch size multiple.
    Assumes that the input has spatial dimensions in last two dimensions.
    """
    if patch_size == 1:
        return input
    # Get the patch size from the feature extractor model
    patch_size = patch_size
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
    h_crop_bottom = h - new_h - h_crop_top
    # Make sure we can also handle crops of size 0
    h_idx_bottom = -h_crop_bottom if h_crop_bottom > 0 else None
    w_crop_right = w - new_w - w_crop_left
    w_idx_right = -w_crop_right if w_crop_right > 0 else None
    input = input[..., h_crop_top:h_idx_bottom, w_crop_left:w_idx_right]
    return input


### DEVICE & NORMALIZATION

def get_device(use_cuda=None):
    if torch.cuda.is_available() and (use_cuda==True):
        device = torch.device('cuda:0')  # use first available GPU
    elif torch.backends.mps.is_available() and (use_cuda==True): #check if mps is available
        device = torch.device('mps')
    else:
        if use_cuda:
            warnings.warn('CUDA or MPS is not available. Using CPU')
        device = torch.device('cpu')
    return device

def normalize_image(image, image_mean, image_std):
    """
    Normalize a numpy array with 2-4 dimensions.

    If the array has multiple channels (determined by the multi_channel_img flag), 
    each channel is normalized independently. If there are multiple time or z frames, 
    they are normalized together.

    Parameters
    ----------
    image : np.ndarray
        Input array to be normalized. Must have 2-4 dimensions.
    image_mean : float or nd.array
        Mean of the input array along non-channel dimensions.
    image_std : float or nd.array
        Standard deviation of the input array along non-channel dimensions.
        
    Returns
    -------
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

    arr_norm = (image - image_mean) / image_std

    return arr_norm
    
def compute_image_stats(image, ignore_n_first_dims=None):
    """
    Compute mean and standard deviation of a numpy array with 2-4 dimensions.

    If the array has multiple channels (determined by the first_dim_is_channel option), 
    each channel is normalized independently. Channels can only be the first dimension of the array.

    Parameters
    ----------
    image : np.ndarray
        Input array to be normalized. Must have 2-4 dimensions.
    ignore_n_first_dims : int
        Number of first dimensions to ignore when computing the mean and standard deviation.
        For example if the input array has 3 dimensions CHW and ignore_n_first_dims=1, the mean
        and standard deviation will be computed for each channel C.
        
    Returns
    -------
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
        image_mean = image.mean(axis=tuple(range(ignore_n_first_dims,image.ndim)), keepdims=True)
        image_std = image.std(axis=tuple(range(ignore_n_first_dims,image.ndim)), keepdims=True)

    return image_mean, image_std


### ANNOTATION HANDLING (optional)

def get_balanced_mask(ref_mask, tgt_mask):
    """
    Given two masks, subsample the second one to have the same number of pixels as the first one.
    Parameters
    ----------
    ref_mask: np.ndarray
        2d boolean reference mask
    tgt_mask: np.ndarray
        2d boolean target mask

    Returns
    -------
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

    Parameters
    ----------
    annotation: 2d np.ndarray
        annotation image, 1 is background, 2,3,... are foreground classes
    d_edge: int
        dilation factor for edge, by default 1

    Returns
    -------
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
    Parameters
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


    Returns
    -------
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