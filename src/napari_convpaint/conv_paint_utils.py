from collections import OrderedDict
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


from torch import nn
from sklearn.model_selection import train_test_split

from torch.nn.functional import interpolate as torch_interpolate

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
        # For labels, take the majority (max count) along the scaling dimensions
        scaled_img = mode(stacked_blocks, axis=(-3, -1)).mode
    return scaled_img

def rescale_features(image, output_shape, order=1):
    """
    Rescale an image to the specified output size.
    """
    return skimage.transform.resize(image, output_shape, order=order, mode='reflect', preserve_range=True)

def rescale_class_labels(label_img, output_shape):
    """
    Rescale a class label image to the specified output size.
    """
    return skimage.transform.resize(label_img, output_shape, order=0, mode='reflect', preserve_range=True).astype(np.uint8)

def rescale_probs(label_prob_img, output_shape, order=0):
    """
    Rescale a class probability image to the specified output size.
    """
    return skimage.transform.resize(label_prob_img, output_shape, order=order, mode='reflect', preserve_range=True)

def pad_for_kernel(data_stack, effective_kernel_padding):
    """
    Pad a data stack (4D image or 3D annotation) to the kernel size.
    Adds half the kernel size to each side of the data stack.
    Takes into account the scale factor (for later pyramid scaling) for padding.
    If kernel[0] is not None, also pad along the Z axis.

    Parameters
    ----------
    data_stack : np.ndarray
        Input data stack to be padded. Shape:
        - [C, Z, H, W] for images (e.g., with multiple channels)
        - [Z, H, W] for annotations
    effective_kernel_padding : tuple of int
        Effective padding for the kernel size, to be added to each side of the data stack.
        Takes into account the scale factor for later pyramid scaling.

    Returns
    -------
    padded_data : np.ndarray
        Padded data stack with the same shape as the input plus padding.
    """
    if not isinstance(effective_kernel_padding, tuple) or len(effective_kernel_padding) != 3:
        raise ValueError("kernel_size must be a tuple of three integers (Z, H, W), " +
                         "whereas Z can be None.")
    
    # Get padding for each dimension
    z_pad, h_pad, w_pad = effective_kernel_padding

    # Create padding tuples
    if data_stack.ndim == 4:  # [C, Z, H, W] for images
        padding = ((0, 0), (z_pad, z_pad), (h_pad, h_pad), (w_pad, w_pad))
        mode = 'reflect'
    elif data_stack.ndim == 3:  # [Z, H, W] for annotations
        padding = ((z_pad, z_pad), (h_pad, h_pad), (w_pad, w_pad))
        mode = 'constant'
    else:
        raise ValueError("Invalid data_stack dimensions. Expected 3D or 4D array.")

    # Apply padding
    padded_data = np.pad(data_stack, padding, mode=mode)

    return padded_data

def pad_to_patch(data_stack, effective_patch_size):
    """
    Pad an image and annotation stack to the next multiple of patch size.
    Takes into account the scale factor (for later pyramid scaling) for padding.
    If kernel[0] is not None, also pad along the Z axis.

    Parameters
    ----------
    data_stack : np.ndarray
        Input data stack to be padded. Shape:
        - [C, Z, H, W] for images (e.g., with multiple channels)
        - [Z, H, W] for annotations
    effective_patch_size : tuple of int
        Patch size, to a multiple of which shall be padded
        Takes into account the scale factor for later pyramid scaling.

    Returns
    -------
    padded_data : np.ndarray
        Padded data stack with the same shape as the input plus padding.
    """
    # Compute padding
    z_pad, h_pad, w_pad = compute_patch_padding(data_stack.shape, effective_patch_size)

    # Split padding evenly (add extra padding to the end if padding is odd)
    z_pad_front, z_pad_back = z_pad // 2, z_pad - z_pad // 2
    h_pad_front, h_pad_back = h_pad // 2, h_pad - h_pad // 2
    w_pad_front, w_pad_back = w_pad // 2, w_pad - w_pad // 2

    # Create padding tuples
    if data_stack.ndim == 4:  # [C, Z, H, W] for images
        padding = ((0, 0), (z_pad_front, z_pad_back), (h_pad_front, h_pad_back), (w_pad_front, w_pad_back))
        mode = 'reflect'
    elif data_stack.ndim == 3:  # [Z, H, W] for annotations
        padding = ((z_pad_front, z_pad_back), (h_pad_front, h_pad_back), (w_pad_front, w_pad_back))
        mode = 'constant'

    # Apply padding
    padded_data = np.pad(data_stack, padding, mode=mode)

    return padded_data

def compute_patch_padding(data_stack_shape, effective_patch_size):
    """
    Compute the padding required to make the data stack a multiple of the patch size.
    Takes into account the scale factor (for later pyramid scaling) for padding.
    """
    if not isinstance(effective_patch_size, tuple) or len(effective_patch_size) != 3:
        raise ValueError("patch_size must be a tuple of three integers (Z, H, W), " +
                         "whereas Z can be None.")
    
    # Extract dimensions
    if len(data_stack_shape) == 4:  # [C, Z, H, W] for images
        _, z_dim, h_dim, w_dim = data_stack_shape
    elif len(data_stack_shape) == 3:  # [Z, H, W] for annotations
        z_dim, h_dim, w_dim = data_stack_shape
    else:
        raise ValueError("Invalid data_stack dimensions. Expected 3D or 4D array.")
    
    # Compute padding for each dimension
    patch_z, patch_h, patch_w = effective_patch_size
    z_pad = (patch_z - (z_dim % patch_z)) % patch_z if patch_z is not None else 0
    h_pad = (patch_h - (h_dim % patch_h)) % patch_h
    w_pad = (patch_w - (w_dim % patch_w)) % patch_w

    return z_pad, h_pad, w_pad

def pre_process_stack(data_stack, input_scaling, effective_kernel_padding, effective_patch_size):
    # Scale input
    scaled_stack = scale_img(data_stack, input_scaling)
    # Pad images and annots; take into account maximum scaling for later pyramid scaling
    padded_stack = scaled_stack
    if effective_kernel_padding is not None:
        padded_stack = pad_for_kernel(padded_stack, effective_kernel_padding)
    if effective_patch_size is not None:
        padded_stack = pad_to_patch(padded_stack, effective_patch_size)
    return padded_stack

def tile_annots_3D(img_stack, annot_stack, effective_kernel_padding, effective_patch_size): # TODO: IMPLEMENT
    return img_stack, annot_stack

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



def parallel_predict_image(image, model, classifier, param, use_dask=False):
    """
    Given a filter model and a classifier, predict the class of 
    each pixel in an image.

    Parameters
    ----------
    image: 2d array
        image to segment
    model: Hookmodel
        model to extract features from
    classifier: CatBoostClassifier
        classifier to use for prediction
    scalings: list of ints
        downsampling factors
    order: int
        interpolation order for low scale resizing
    use_min_features: bool
        if True, use the minimum number of features per layer
    image_downsample: int, optional
        downsample image by this factor before extracting features, by default 1
    use_dask: bool
        if True, use dask for parallel processing

    Returns
    -------
    predicted_image: 2d array
        predicted image with classes

    """
    
    maxblock = 1000
    nblocks_rows = image.shape[-2] // maxblock
    nblocks_cols = image.shape[-1] // maxblock
    margin = 50

    predicted_image_complete = np.zeros(image.shape[-2:], dtype=(np.uint8))
    
    if use_dask:
        from dask.distributed import Client
        import dask
        dask.config.set({'distributed.worker.daemon': False})
        client = Client()
        processes = []

    min_row_ind_collection = []
    min_col_ind_collection = []
    max_row_ind_collection = []
    max_col_ind_collection = []
    new_max_col_ind_collection = []
    new_max_row_ind_collection = []
    new_min_col_ind_collection = []
    new_min_row_ind_collection = []

    for row in range(nblocks_rows+1):
        for col in range(nblocks_cols+1):
            #print(f'row {row}/{nblocks_rows}, col {col}/{nblocks_cols}')
            max_row = np.min([image.shape[-2], (row+1)*maxblock+margin])
            max_col = np.min([image.shape[-1], (col+1)*maxblock+margin])
            min_row = np.max([0, row*maxblock-margin])
            min_col = np.max([0, col*maxblock-margin])

            min_row_ind = 0
            new_min_row_ind = 0
            if min_row > 0:
                min_row_ind = min_row + margin
                new_min_row_ind = margin
            min_col_ind = 0
            new_min_col_ind = 0
            if min_col > 0:
                min_col_ind = min_col + margin
                new_min_col_ind = margin

            max_col = (col+1)*maxblock+margin
            max_col_ind = np.min([min_col_ind+maxblock,image.shape[-1]])
            new_max_col_ind = new_min_col_ind + (max_col_ind-min_col_ind)
            if max_col > image.shape[-1]:
                max_col = image.shape[-1]

            max_row = (row+1)*maxblock+margin
            max_row_ind = np.min([min_row_ind+maxblock,image.shape[-2]])
            new_max_row_ind = new_min_row_ind + (max_row_ind-min_row_ind)
            if max_row > image.shape[-2]:
                max_row = image.shape[-2]

            image_block = image[..., min_row:max_row, min_col:max_col]

            if use_dask:
                processes.append(client.submit(
                    model.predict_image,
                    image=image_block,
                    classifier=classifier,
                    param=param))
                
                
                min_row_ind_collection.append(min_row_ind)
                min_col_ind_collection.append(min_col_ind)
                max_row_ind_collection.append(max_row_ind)
                max_col_ind_collection.append(max_col_ind)
                new_max_col_ind_collection.append(new_max_col_ind)
                new_max_row_ind_collection.append(new_max_row_ind)
                new_min_col_ind_collection.append(new_min_col_ind)
                new_min_row_ind_collection.append(new_min_row_ind)

            else:
                predicted_image = model.predict_image(
                    image=image_block,
                    classifier=classifier,
                    param=param
                )
                crop_pred = predicted_image[
                    new_min_row_ind: new_max_row_ind,
                    new_min_col_ind: new_max_col_ind]

                predicted_image_complete[min_row_ind:max_row_ind, min_col_ind:max_col_ind] = crop_pred.astype(np.uint8)

    if use_dask:
        for k in range(len(processes)):
            future = processes[k]
            out = future.result()
            future.cancel()
            del future
            predicted_image_complete[
                min_row_ind_collection[k]:max_row_ind_collection[k],
                min_col_ind_collection[k]:max_col_ind_collection[k]] = out.astype(np.uint8)
        client.close()
    
    return predicted_image_complete

rot_model = None



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