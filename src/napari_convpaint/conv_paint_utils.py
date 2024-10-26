from collections import OrderedDict
import warnings

import torch
import numpy as np
import skimage.transform
import skimage.morphology as morph
from joblib import Parallel, delayed
import einops as ein
#import xgboost as xgb


from torch import nn
from sklearn.model_selection import train_test_split

from torch.nn.functional import interpolate as torch_interpolate


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

    If the array has multiple channels (determined by the multi_channel_training flag), 
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