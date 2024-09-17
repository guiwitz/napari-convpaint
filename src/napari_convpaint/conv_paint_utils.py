from collections import OrderedDict
import warnings

import torch
import numpy as np
import skimage.transform
import skimage.morphology as morph
import pandas as pd
from joblib import Parallel, delayed
import einops as ein
#import xgboost as xgb


from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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
        Mean of the input array along non-channel dimensions. If None, the mean is calculated from the input array.
    image_std : float or nd.array
        Standard deviation of the input array along non-channel dimensions. If None, the standard deviation is calculated from the input array.
        
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

    # The computing of stats and image normalization is split into two function. This will
    # allow to normalize even a single image of the stack with the same stats as the whole stack.
    # This is done in the perspective of handling large stacks that cannot be loaded in memory.
    # This needs to be implemented:
        # - computing stats stack by stack and combining the result at the end
        # - allow to normalize a single image from a stack
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


def filter_image(image, model, scalings):
    """
    Filter an image with the first layer of a VGG16 model.
    Apply filter to each scale in scalings.

    Parameters
    ----------
    image: 2d array
        image to filter
    model: pytorch model
        first layer of VGG16
    scalings: list of ints
        downsampling factors

    Returns
    -------
    all_scales: list of 2d array
        list of filtered images. For each downsampling factor,
        there are N images, N being the number of filters of the model.

    """

    n_filters = 64
    # convert to np if dasks array
    image = np.asarray(image)

    all_scales = []
    for s in scalings:
        im_tot = image[::s, ::s].astype(np.float32)
        # im_tot = np.ones((3,im_tot.shape[0],im_tot.shape[1]), dtype=np.float32) * im_tot
        im_torch = torch.tensor(im_tot[np.newaxis, np.newaxis, ::])
        out = model(im_torch)
        out_np = out.detach().numpy()
        if s > 1:
            out_np = skimage.transform.resize(
                out_np, (1, n_filters, image.shape[0], image.shape[1]), preserve_range=True)
        all_scales.append(out_np)
    return all_scales


def filter_image_multioutputs(image, hookmodel, scalings=[1], order=0,
                              image_downsample=1):
    """Recover the outputs of chosen layers of a pytorch model. Layers and model are
    specified in the hookmodel object.
    
    Parameters
    ----------
    image : np.ndarray
        2d Image to filter
    hookmodel : Hookmodel
        Model to extract features from
    scalings : list of ints, optional
        Downsampling factors, by default None
    order : int, optional
        Interpolation order for low scale resizing,
        by default 0
    image_downsample : int, optional
        Downsample image by this factor before extracting features, by default 1

    Returns
    -------
    all_scales : list of np.ndarray
        List of filtered images. There are N x S images, N being the number of layers
        of the model, and S the number of scaling factors. Each element has shape [1, F, H, W]
        where F is the number of filters of the layer.
        
    """

    input_channels = hookmodel.named_modules[0][1].in_channels
    image = np.asarray(image, dtype=np.float32)

    if image.ndim == 2:
        image = image[::image_downsample, ::image_downsample]
        image = np.ones((input_channels, image.shape[0], image.shape[1]), dtype=np.float32) * image
    elif image.ndim == 3:
        image = image[:, ::image_downsample, ::image_downsample]
        if image.shape[0] != input_channels:
            warnings.warn(f'Image has {image.shape[0]} channels, model expects {input_channels}. Using mean projection.')
            image = np.mean(image, axis=0)
            image = np.ones((input_channels, image.shape[0], image.shape[1]), dtype=np.float32) * image

    int_mode = 'bilinear' if order > 0 else 'nearest'
    align_corners = False if order > 0 else None

    all_scales = []
    with torch.no_grad():
        for s in scalings:
            im_tot = image[:, ::s, ::s]
            im_torch = torch.tensor(im_tot[np.newaxis, ::])
            # im_torch = hookmodel.transform(im_torch)  # different normalization required
            hookmodel.outputs = []
            try:
                _ = hookmodel(im_torch)
            except AssertionError as ea:
               pass
            except Exception as ex:
                raise ex
    
            for im in hookmodel.outputs:
                if image.shape[1:3] != im.shape[2:4]:
                    im = torch_interpolate(im, size=image.shape[1:3], mode=int_mode, align_corners=align_corners)
                    '''out_np = skimage.transform.resize(
                        image=out_np,
                        output_shape=(1, out_np.shape[1], image.shape[1], image.shape[2]),
                        preserve_range=True, order=order)'''
    
                out_np = im.cpu().detach().numpy()
                all_scales.append(out_np)

    return all_scales

def parallel_predict_image(image, model, classifier, scalings=[1], order=0,
                  use_min_features=True, image_downsample=1, use_dask=False):
    """
    Given a filter model and a classifier, predict the class of 
    each pixel in an image.

    Parameters
    ----------
    image: 2d array
        image to segment
    model: Hookmodel
        model to extract features from
    classifier: sklearn classifier
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
                    scalings=scalings,
                    order=order,
                    use_min_features=use_min_features,
                    image_downsample=image_downsample))
                
                
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
                    scalings=scalings,
                    order=order,
                    use_min_features=use_min_features,
                    image_downsample=image_downsample
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

def get_features_current_layers(image, annotations, model=None, scalings=[1],
                                order=0, use_min_features=True,
                                image_downsample=1,tile_annotations=False):
    """Given a potentially multidimensional image and a set of annotations,
    extract multiscale features from multiple layers of a model.
    
    Parameters
    ----------
    image : np.ndarray
        2D or 3D Image to extract features from. If 3D but annotations are 2D,
        image is treated as multichannel and not image series
    annotations : np.ndarray
        2D, 3D Annotations (1,2) to extract features from
    model : feature extraction model
        Model to extract features from the image
    scalings : list of ints
        Downsampling factors
    order : int, optional
        Interpolation order for low scale resizing, by default 0
    use_min_features : bool, optional
        Use minimal number of features, by default True
    image_downsample : int, optional
        Downsample image by this factor before extracting features, by default 1

    Returns
    -------
    features : pandas DataFrame
        Extracted features (rows are pixel, columns are features)
    targets : pandas Series
        Target values
    """

    if model is None:
        raise ValueError('Model must be provided')

    # get indices of first dimension of non-empty annotations. Gives t/z indices
    if annotations.ndim == 3:
        non_empty = np.unique(np.where(annotations > 0)[0])
        if len(non_empty) == 0:
            warnings.warn('No annotations found')
            return None, None
    elif annotations.ndim == 2:
        non_empty = [0]
    else:
        raise Exception('Annotations must be 2D or 3D')

    all_values = []
    all_targets = []

    # find maximal padding necessary
    padding = model.get_padding() * np.max(scalings)

    # iterating over non_empty iteraties of t/z for 3D data
    for ind, t in enumerate(non_empty):

        if annotations.ndim == 2:
            if image.ndim == 3:
                current_image = np.pad(image, ((0,0), (padding,padding), (padding,padding)), mode='reflect')
            else:
                current_image = np.pad(image, padding, mode='reflect')
            current_annot = np.pad(annotations, padding, mode='constant')
        else:
            if image.ndim == 3:
                current_image = np.pad(image[t], padding, mode='reflect')
                current_annot = np.pad(annotations[t], padding, mode='constant')
            elif image.ndim == 4:
                current_image = np.pad(image[:, t], ((0,0),(padding, padding),(padding, padding)), mode='reflect')
                current_annot = np.pad(annotations[t], padding, mode='constant')

        annot_regions = skimage.morphology.label(current_annot > 0)

        if tile_annotations:
            boxes = skimage.measure.regionprops_table(annot_regions, properties=('label', 'bbox'))
        else:
            boxes = {'label': [1], 'bbox-0': [padding], 'bbox-1': [padding], 'bbox-2': [current_annot.shape[0]-padding], 'bbox-3': [current_annot.shape[1]-padding]}
        for i in range(len(boxes['label'])):
            # NOTE: This assumes that the image is already padded correctly, and the padded boxes cannot go out of bounds
            pad_size = model.get_padding()
            x_min = boxes['bbox-0'][i]-pad_size
            x_max = boxes['bbox-2'][i]+pad_size
            y_min = boxes['bbox-1'][i]-pad_size
            y_max = boxes['bbox-3'][i]+pad_size

            # temporary check that bounds are not out of image
            x_min = max(0, x_min)
            x_max = min(current_image.shape[-2], x_max)
            y_min = max(0, y_min)
            y_max = min(current_image.shape[-1], y_max)

            im_crop = current_image[...,
                x_min:x_max,
                y_min:y_max
            ]
            annot_crop = current_annot[
                x_min:x_max,
                y_min:y_max
            ]
            extracted_features = model.get_features_scaled(
                image=im_crop,
                scalings=scalings,
                order=order,
                use_min_features=use_min_features,
                image_downsample=image_downsample)
            
            if image_downsample > 1:
                annot_crop = annot_crop[::image_downsample, ::image_downsample]

            #from the [features, w, h] make a list of [features] with len nb_annotations
            mask = annot_crop > 0
            nb_features = extracted_features.shape[0]

            extracted_features = np.moveaxis(extracted_features, 0, -1) #move [w,h,features]

            extracted_features = extracted_features[mask]
            all_values.append(extracted_features)

            targets = annot_crop[annot_crop > 0]
            targets = targets.flatten()
            all_targets.append(targets)
        

    all_values = np.concatenate(all_values, axis=0)
    features = pd.DataFrame(all_values)

    all_targets = np.concatenate(all_targets, axis=0)
    targets = pd.Series(all_targets)

    return features, targets


rot_model = None

def get_rot_model(device='cpu'):
    global rot_model
    if rot_model is not None:
        return rot_model

    # create a torch model that takes 2 input tensors and convolves first with horizontal edge detector
    # and second with vertical edge detector, then calculates difference of the two outputs

    # create horizontal edge detector
    h_edge = np.zeros((1, 2, 3, 3))  # out, in, h, w
    h_edge[0, 0, 0, :] = -1
    h_edge[0, 0, 2, :] = 1
    h_edge = torch.from_numpy(h_edge).float()

    # create vertical edge detector
    v_edge = np.zeros((1, 2, 3, 3))
    v_edge[0, 1, :, 0] = 1
    v_edge[0, 1, :, 2] = -1
    v_edge = torch.from_numpy(v_edge).float()

    # create kernel summing 2 channels
    kernel = np.ones((1, 2, 1, 1))
    kernel = torch.from_numpy(kernel).float()

    # create model
    model = nn.Sequential(OrderedDict([('conv1', nn.Conv2d(in_channels=2, out_channels=2,
                                                           kernel_size=3, padding=1, bias=False)
                                        )]))
    # load edge detectors
    model.conv1.weight = nn.Parameter(torch.cat([h_edge, v_edge], dim=0))
    model.conv1.weight.requires_grad = False
    # create second layer
    model.add_module('conv2', nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, bias=False))
    # load weights for second layer
    model.conv2.weight = nn.Parameter(kernel)
    model.conv2.weight.requires_grad = False

    model.eval()
    rot_model = model.to(device)
    return rot_model


def get_rot_features(grad_x, grad_y, device='cpu'):
    rot_model = get_rot_model(device=device)

    # create torch tensors
    grad_x = torch.from_numpy(grad_x).float()
    grad_y = torch.from_numpy(grad_y).float()
    # stack tensors
    grad_xy = torch.stack([grad_x, grad_y], dim=1)
    # convolve with edge detectors
    rot = rot_model(grad_xy).detach().numpy()
    return rot[:, 0]  # single out channels, orig channels are in the batch dimension


def append_grad_rot(features, add_grad, add_rot, feature_info, device):
    """
    Append gradient and rotation features to a feature array according to add_grad and add_rot flags
    and updates the feature_info dictionary accordingly.
    Parameters
    ----------
    features: np.ndarray
        feature array, 3D, CHW
    add_grad: bool
        if True, add gradient features: gradient in x, gradient in y. Triples feature number.
    add_rot: bool
        if True, add rotor features: convolves grad features with corresponding "edge detector" and sums.
        adds one feature per original feature.
    feature_info: dict
        feature_info[idx] = [idx, name=f'{layer_name}_{sign}_{scale}_{idx:04d}', mult, scale, layer_name,
         ch_idx, type='gradx'|'grady'|'rot'|'orig', useful:bool]

    Returns
    -------
    features: np.ndarray
    """
    n_features = features.shape[0]
    res_features = features

    mult_factor_sign = {1:'+', -1:'-'}
    
    if add_grad:
        grad_x = np.gradient(features, axis=1)
        grad_y = np.gradient(features, axis=2)
        all_features = [features, grad_x, grad_y]

        for idx in range(n_features):
            fi_idx = feature_info[idx]
            mult, scale, layer_name, ch_idx = fi_idx[2:6]
            sign=mult_factor_sign[mult]

            gx_idx = idx + n_features
            gy_idx = idx + 2 * n_features
            feature_info[gx_idx] = [gx_idx, f'{layer_name}_{sign}_{scale}_{gx_idx:04d}', mult, scale, layer_name,
                                    ch_idx, 'gradx', True]
            feature_info[gy_idx] = [gy_idx, f'{layer_name}_{sign}_{scale}_{gy_idx:04d}', mult, scale, layer_name,
                                    ch_idx, 'grady', True]

        if add_rot:
            rot = get_rot_features(grad_x, grad_y, device=device)
            all_features.append(rot)

            for idx in range(n_features):
                fi_idx = feature_info[idx]
                mult, scale, layer_name, ch_idx = fi_idx[2:6]
                sign=mult_factor_sign[mult]

                rot_idx = idx + 3 * n_features
                feature_info[rot_idx] = [rot_idx, f'{layer_name}_{sign}_{scale}_{rot_idx:04d}', mult, scale, layer_name,
                                         ch_idx, 'rot', True]

        res_features = np.concatenate(all_features, axis=0)

    return res_features


def fill_useless(features_all_samples, feature_info):
    """
    Update useless flag in feature_info dictionary based on features:
    if feature has zero variance, feature is useless

    Parameters
    ----------
    features_all_samples: list of pandas DataFrame (raw pixel, columns feature)
    feature_info: dict
        see definition in append_grad_rot

    Returns
    -------
    None
    """

    # update useless flag in feature_info dictionary based on features:
    # if feature has zero variance, feature is useless
    n_features = features_all_samples[0].shape[1]
    for idx in range(n_features):
        # fill values for all samples from dataframe column
        all_values = np.concatenate([np.asarray(f.iloc[:, idx]) for f in features_all_samples], axis=0)
        # if feature has zero variance, feature is useless
        feature_info[idx][7] = np.var(all_values) > 0


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


def get_features_single_img_rich(model, image, annotation, scalings=[1],
                                 full_annotation=True,
                                 add_grad=False,
                                 add_rot=False,
                                 add_neg=True,
                                 order=0):
    """Given a 2D image and annotation,
        extract multiscale features from multiple layers of a model,
        add gradient and rotation features,
        select only pixels in annotations if not full_annotation otherwise balance background and foreground,
        return features, labels, and feature info dictionary.

        Parameters
        ----------
        model : Hookmodel
            Model to extract features from
        image : np.ndarray
            2D or 3D Image to extract features from
        annotation : np.ndarray
            2D, 3D Annotations (1,2) to extract features from
        scalings : list of ints
            Downsampling factors
        full_annotation : bool
            if True, use all pixels in annotations, otherwise balance background and foreground
            if True - assumes annotations are 0,1,..., where 0 is bg. otherwise assumes 0 is not annotated,
             and 0 is background, 1 is first class, etc. Default True
        add_grad : bool
            if True, add gradient features: gradient in x, gradient in y. Triples feature number. Default False.
        add_rot : bool
            if True, add rotor features: convolve grad features with corresponding "edge detector" and sum.
            Default False.
        add_neg : bool
            if True, add all features also for `-image`
            Default True.
        order : int, optional
            Interpolation order for low scale resizing, by default 0
        use_min_features : bool, optional
            Use minimal number of features, by default True

        Returns
        -------
        features : pandas DataFrame
            Extracted features (rows are pixel, columns are features)
        targets : pandas Series
            Target values
        feature_info : dict
                    feature_info[idx] = [idx, name=f'{layer_name}_{sign}_{scale}_{idx:04d}', mult, scale, layer_name, ch_idx,
                             type='gradx'|'grady'|'rot'|'orig', useful:bool]
        """

    # get indices of first dimension of non-empty annotation. Gives t/z indices
    if annotation.ndim != 2 or image.ndim != 2:
        raise Exception('annotation must be 2D')
    assert (not add_rot) or add_grad, 'add_grad must be True if add_rot is True'

    if full_annotation:
        annotation = annotation + 1

    mult_factors = [1, -1] if add_neg else [1]
    mult_factor_sign = {1:'+', -1:'-'}
    feature_info = {}
    extracted_features = []
    
    for mf in mult_factors:
        sign = mult_factor_sign[mf]
        image_f = (-image) if mf==-1 else image
        
        extracted_features.append(np.expand_dims(image_f, axis=(0, 1)))
        features = filter_image_multioutputs(image_f, model, scalings, order=order)
        extracted_features.extend(features)
        
        idx = len(feature_info)
        feature_info[idx] = [idx, f'raw_{sign}_1_1', mf, 1, 'raw', 0, 'raw', True]
    
        for scale in scalings:
            for l_idx, (layer_name, n_features) in enumerate(zip(model.selected_layers, model.features_per_layer)):
                for ch_idx in range(n_features):
                    idx = len(feature_info)
                    feature_info[idx] = [idx, f'{layer_name}_{sign}_{scale}_{idx:04d}', mf, scale, layer_name, ch_idx, 'orig', True]

    extracted_features = np.concatenate([] + extracted_features, axis=1)
    assert len(extracted_features) == 1, 'only one image at a time'
    extracted_features = extracted_features[0]
    
    extracted_features = append_grad_rot(extracted_features, add_grad, add_rot, feature_info, device=model.device)

    # select pixels in annotations

    features, targets = extract_annotated_pixels(extracted_features, annotation, full_annotation=full_annotation)
    features = pd.DataFrame(features)
    targets = pd.Series(targets)

    return features, targets, feature_info


def get_features_all_samples_rich(model, images, annotations, scalings=[1],
                                  full_annotation=True,
                                  add_grad=False,
                                  add_rot=False,
                                  add_neg=True,
                                  order=0):
    """
    Given a potentially multidimensional image and a set of annotations,
    extract multiscale features from multiple layers of a model,
    add gradient and rotation features,
    select only pixels in annotations if not full_annotation otherwise balance background and foreground,
    return features, labels, and feature info dictionary.

    Parameters
    ----------
    model : Hookmodel
        Model to extract features from

    images : np.ndarray
        2D or 3D Image to extract features from
    annotations : np.ndarray
        2D, 3D Annotations (1,2) to extract features from
    scalings : list of ints
        Downsampling factors
    full_annotation : bool
        if True, use all pixels in annotations, otherwise balance background and foreground
        if True - assumes annotations are 0,1,..., where 0 is bg. otherwise assumes 0 is not annotated,
         and 0 is background, 1 is first class, etc. Default True
        add_grad : bool
            if True, add gradient features: gradient in x, gradient in y. Triples feature number. Default False.
        add_rot : bool
            if True, add rotor features: convolve grad features with corresponding "edge detector" and sum.
            Default False.
        add_neg : bool
            if True, add all features also for `-image`
            Default True.
    order : int, optional
        Interpolation order for low scale resizing, by default 0
    use_min_features : bool, optional
        Use minimal number of features, by default True

    Returns
    -------
    features : list of pandas DataFrame
        Extracted features (rows are pixel, columns are features), per sample
    targets : list of pandas Series
        Target values, per sample
    feature_info : dict
        feature_info[idx] = [idx, name=f'{layer_name}_{scale}_{idx:04d}', scale, layer_name, ch_idx,
         type='gradx'|'grady'|'rot'|'orig', useful:bool]

    """

    # get indices of first dimension of non-empty annotation. Gives t/z indices
    if annotations.ndim == 2:
        assert images.ndim == 2, 'images must be 2D (HW) in case of 2D annotations'
        annotations = np.expand_dims(annotations, axis=0)
        images = np.expand_dims(images, axis=0)

    if annotations.ndim == 3:
        max_annot_idx = np.max(annotations, axis=(1, 2))
        non_empty = np.argwhere(max_annot_idx > 0)[:, 0]
        if len(non_empty) == 0:
            warnings.warn('No annotations found')
            return None, None, None
    else:
        raise Exception('Annotations must be 2D or 3D')

    features_all_samples = []
    targets_all_samples = []

    feature_info = {}
    for t in non_empty:
        current_image = images[t]
        current_annot = annotations[t]

        features, targets, feature_info = get_features_single_img_rich(model, current_image, current_annot,
                                                                       scalings=scalings,
                                                                       full_annotation=full_annotation,
                                                                       add_grad=add_grad,
                                                                       add_rot=add_rot,
                                                                       add_neg=add_neg,
                                                                       order=order
                                                                       )
        features_all_samples.append(features)
        targets_all_samples.append(targets)

    fill_useless(features_all_samples, feature_info)

    return features_all_samples, targets_all_samples, feature_info


def train_classifier(features, targets):
    """Train a random forest classifier given a set of features and targets."""

    # train a random forest classififer
    random_forest = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    random_forest.fit(features, targets)
    
    #random_forest = xgb.XGBClassifier(tree_method="hist", n_estimators=100, n_jobs=8)
    #random_forest.fit(features, targets-1)

    return random_forest
