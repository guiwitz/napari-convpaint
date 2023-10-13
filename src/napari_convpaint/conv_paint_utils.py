from collections import OrderedDict
from typing import Any
import warnings
from pathlib import Path

import torch
import numpy as np
import skimage.transform
import skimage.morphology as morph
import pandas as pd
from joblib import dump, load
from joblib import Parallel, delayed
import yaml
import zarr
import einops as ein

import torchvision.models as models
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from torchvision import transforms
from torch.nn.functional import interpolate as torch_interpolate

from .conv_parameters import Param


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
        im_torch = torch.from_numpy(im_tot[np.newaxis, np.newaxis, ::])
        out = model.forward(im_torch)
        out_np = out.detach().numpy()
        if s > 1:
            out_np = skimage.transform.resize(
                out_np, (1, n_filters, image.shape[0], image.shape[1]), preserve_range=True)
        all_scales.append(out_np)
    return all_scales


def filter_image_multioutputs(image, hookmodel, scalings=[1], order=0, device='cpu',
                              normalize=False, image_downsample=1):
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
    device : str, optional
        Device to use for computation, by default 'cpu'
    normalize : bool, optional
        If True, normalize each channel with its mean and std, by default False
    image_downsample : int, optional
        Downsample image by this factor before extracting features, by default 1

    Returns
    -------
    all_scales : list of np.ndarray
        List of filtered images. The number of image is the sum over 
        selected output layers of NxS where N is the number of filters
        of the layer and S the number of scaling factors.
        
    """

    is_model_cuda = device == 'cuda'

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

    if normalize:
        image = (image - image.mean(axis=(1, 2), keepdims=True)) / image.std(axis=(1, 2), keepdims=True)

    int_mode = 'bilinear' if order > 0 else 'nearest'
    align_corners = False if order > 0 else None

    all_scales = []
    for s in scalings:
        im_tot = image[:, ::s, ::s]
        im_torch = torch.from_numpy(im_tot[np.newaxis, ::])
        # im_torch = hookmodel.transform(im_torch)
        hookmodel.outputs = []
        try:
            if is_model_cuda:
                im_torch = im_torch.cuda()

            _ = hookmodel(im_torch)
        except:
            pass

        for im in hookmodel.outputs:
            if image.shape[1:3] != im.shape[2:4]:
                im = torch_interpolate(im, size=image.shape[1:3], mode=int_mode, align_corners=align_corners)
                '''out_np = skimage.transform.resize(
                    image=out_np,
                    output_shape=(1, out_np.shape[1], image.shape[1], image.shape[2]),
                    preserve_range=True, order=order)'''

            if is_model_cuda:
                im = im.to('cpu')

            out_np = im.detach().numpy()
            all_scales.append(out_np)

    return all_scales


def predict_image(image, model, classifier, scalings=[1], order=0,
                  use_min_features=True, device='cpu', normalize=False, image_downsample=1):
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
    device: str, optional
        device to use for computation, by default 'cpu'
    normalize: bool, optional
        if True, normalize each channel with its mean and std, by default False
    image_downsample: int, optional
        downsample image by this factor before extracting features, by default 1


    Returns
    -------
    predicted_image: 2d array
        predicted image with classes

    """

    if use_min_features:
        max_features = np.min(model.features_per_layer)
        all_scales = filter_image_multioutputs(
            image, model, scalings=scalings, order=order,
            device=device, normalize=normalize, image_downsample=image_downsample)
        all_scales = [a[:, 0:max_features, :, :] for a in all_scales]
        tot_filters = max_features * len(all_scales)

    else:
        # max_features = np.max(model.features_per_layer)
        all_scales = filter_image_multioutputs(
            image, model, scalings=scalings, order=order,
            device=device, normalize=normalize, image_downsample=image_downsample)
        # MV:  why this?         (64, 256, 512)                2            / 3
        tot_filters = np.sum(model.features_per_layer) * len(all_scales) / len(model.features_per_layer)

    tot_filters = int(tot_filters)
    rows = np.ceil(image.shape[-2] / image_downsample).astype(int)
    cols = np.ceil(image.shape[-1] / image_downsample).astype(int)
    all_pixels = pd.DataFrame(
        np.reshape(np.concatenate(all_scales, axis=1), newshape=(tot_filters, rows * cols)).T)

    predictions = classifier.predict(all_pixels)

    predicted_image = np.reshape(predictions, [rows, cols])
    if image_downsample > 1:
        predicted_image = skimage.transform.resize(
            image=predicted_image,
            output_shape=(image.shape[-2], image.shape[-1]),
            preserve_range=True, order=1).astype(np.uint8)

    return predicted_image


def load_single_layer_vgg16(keep_rgb=False):
    """Load VGG16 model from torchvision, keep first layer only
    
    Parameters
    ----------

    Returns
    -------
    model: torch model
        model for pixel feature extraction
    keep_rgb: bool
        if True, keep model with three input channels, otherwise convert to single channel
    
    """

    vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    pretrained_dict = vgg16.state_dict()
    
    if keep_rgb:
        model = nn.Sequential(OrderedDict([('conv1', nn.Conv2d(3, 64, 3, 1, 1))]))
    else:
        model = nn.Sequential(OrderedDict([('conv1', nn.Conv2d(1, 64, 3, 1, 1))]))

    reduced_dict = model.state_dict()

    if keep_rgb:
        reduced_dict['conv1.weight'] = pretrained_dict['features.0.weight']
        reduced_dict['conv1.bias'] = pretrained_dict['features.0.bias']
    else:
        reduced_dict['conv1.weight'][:, 0, :, :] = pretrained_dict['features.0.weight'][:, :, :, :].sum(axis=1)
        reduced_dict['conv1.bias'] = pretrained_dict['features.0.bias']

    model.load_state_dict(reduced_dict)

    return model

def get_multiscale_features(model, image, annotations, scalings, order=0,
                            use_min_features=True, device='cpu',
                            normalize=False, image_downsample=1):
    """Given an image and a set of annotations, extract multiscale features
    
    Parameters
    ----------
    model : Hookmodel
        Model to extract features from
    image : np.ndarray
        Image to extract features from (currently 2D only or 2D multichannel (C,H,W))
    annotations : np.ndarray
        Annotations (1,2) to extract features from (currently 2D only)
    scalings : list of ints
        Downsampling factors
    order : int, optional
        Interpolation order for low scale resizing, by default 0
    use_min_features : bool, optional
        Use minimal number of features, by default True
    device : str, optional
        Device to use for computation, by default 'cpu'
    normalize : bool, optional
        If True, normalize each channel with its mean and std, by default False
    image_downsample : int, optional
        Downsample image by this factor before extracting features, by default 1

    Returns
    -------
    extracted_features : np.ndarray
        Extracted features. Dimensions npixels x nfeatures * nbscales
    """

    if use_min_features:
        max_features = np.min(model.features_per_layer)
    else:
        max_features = np.max(model.features_per_layer)
    # test with minimal number of features i.e. taking only n first features
    rows = np.ceil(image.shape[-2] / image_downsample).astype(int)
    cols = np.ceil(image.shape[-1] / image_downsample).astype(int)
    full_annotation = np.ones((max_features, rows, cols), dtype=np.bool_)
    full_annotation = full_annotation * annotations[::image_downsample, ::image_downsample] > 0

    all_scales = filter_image_multioutputs(
        image, model, scalings, order=order, device=device,
        normalize=normalize, image_downsample=image_downsample)
    if use_min_features:
        all_scales = [a[:, 0:max_features, :, :] for a in all_scales]
    all_values_scales = []

    for ind, a in enumerate(all_scales):
        n_features = a.shape[1]
        extract = a[0, full_annotation[0:n_features]]

        all_values_scales.append(np.reshape(extract, (n_features, int(extract.shape[0] / n_features))).T)
    extracted_features = np.concatenate(all_values_scales, axis=1)

    return extracted_features


def get_features_current_layers(model, image, annotations, scalings=[1],
                                order=0, use_min_features=True, device='cpu',
                                normalize=False, image_downsample=1):
    """Given a potentially multidimensional image and a set of annotations,
    extract multiscale features from multiple layers of a model.
    
    Parameters
    ----------
    model : Hookmodel
        Model to extract features from
    image : np.ndarray
        2D or 3D Image to extract features from. If 3D but annotations are 2D,
        image is treated as multichannel and not image series
    annotations : np.ndarray
        2D, 3D Annotations (1,2) to extract features from
    scalings : list of ints
        Downsampling factors
    order : int, optional
        Interpolation order for low scale resizing, by default 0
    use_min_features : bool, optional
        Use minimal number of features, by default True
    device : str, optional
        Device to use for computation, by default 'cpu'
    normalize : bool, optional
        If True, normalize each channel with its mean and std, by default False
    image_downsample : int, optional
        Downsample image by this factor before extracting features, by default 1

    Returns
    -------
    features : pandas DataFrame
        Extracted features (rows are pixel, columns are features)
    targets : pandas Series
        Target values
    """

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
    # iterating over non_empty iteraties of t/z for 3D data
    for ind, t in enumerate(non_empty):

        if annotations.ndim == 2:
            current_image = image
            current_annot = annotations
        else:
            current_image = image[t]
            current_annot = annotations[t]

        extracted_features = get_multiscale_features(
            model, current_image, current_annot, scalings,
            order=order, use_min_features=use_min_features,
            device=device, normalize=normalize, image_downsample=image_downsample)
        all_values.append(extracted_features)
        
        all_targets.append(current_annot[::image_downsample, ::image_downsample]
                           [current_annot[::image_downsample, ::image_downsample] > 0])
        

    all_values = np.concatenate(all_values, axis=0)
    features = pd.DataFrame(all_values)
    targets = pd.Series(np.concatenate(all_targets, axis=0))

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
        feature_info[idx] = [idx, name=f'{layer_name}_{scale}_{idx:04d}', scale, layer_name,
         ch_idx, type='gradx'|'grady'|'rot'|'orig', useful:bool]

    Returns
    -------
    features: np.ndarray
    """
    n_features = features.shape[0]
    res_features = features
    if add_grad:
        grad_x = np.gradient(features, axis=1)
        grad_y = np.gradient(features, axis=2)
        all_features = [features, grad_x, grad_y]

        for idx in range(n_features):
            fi_idx = feature_info[idx]
            scale, layer_name, ch_idx = fi_idx[2:5]

            gx_idx = idx + n_features
            gy_idx = idx + 2 * n_features
            feature_info[gx_idx] = [gx_idx, f'{layer_name}_{scale}_{gx_idx:04d}', scale, layer_name,
                                    ch_idx, 'gradx', True]
            feature_info[gy_idx] = [gy_idx, f'{layer_name}_{scale}_{gy_idx:04d}', scale, layer_name,
                                    ch_idx, 'grady', True]

        if add_rot:
            rot = get_rot_features(grad_x, grad_y, device=device)
            all_features.append(rot)

            for idx in range(n_features):
                fi_idx = feature_info[idx]
                scale, layer_name, ch_idx = fi_idx[2:5]

                rot_idx = idx + 3 * n_features
                feature_info[rot_idx] = [rot_idx, f'{layer_name}_{scale}_{rot_idx:04d}', scale, layer_name,
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
        feature_info[idx][6] = np.var(all_values) > 0


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
                                 add_grad=True,
                                 add_rot=True,
                                 order=0, device='cpu'):
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
            if True, add gradient features: gradient in x, gradient in y. Triples feature number. Default True.
        add_rot : bool
            if True, add rotor features: convolve grad features with corresponding "edge detector" and sum.
            Default True.
        order : int, optional
            Interpolation order for low scale resizing, by default 0
        use_min_features : bool, optional
            Use minimal number of features, by default True
        device : str, optional
            Device to use for computation, by default 'cpu'

        Returns
        -------
        features : pandas DataFrame
            Extracted features (rows are pixel, columns are features)
        targets : pandas Series
            Target values
        feature_info : dict
                    feature_info[idx] = [idx, name=f'{layer_name}_{scale}_{idx:04d}', scale, layer_name, ch_idx,
                             type='gradx'|'grady'|'rot'|'orig', useful:bool]
        """

    # get indices of first dimension of non-empty annotation. Gives t/z indices
    if annotation.ndim != 2 or image.ndim != 2:
        raise Exception('annotation must be 2D')
    assert (not add_rot) or add_grad, 'add_grad must be True if add_rot is True'

    if full_annotation:
        annotation = annotation + 1

    extracted_features = filter_image_multioutputs(image, model, scalings, order=order, device=device)
    extracted_features = np.concatenate([np.expand_dims(image, axis=(0, 1))] + extracted_features, axis=1)
    assert len(extracted_features) == 1, 'only one image at a time'
    extracted_features = extracted_features[0]

    feature_info = {}
    feature_info[0] = [0, f'raw_1_1', 1, 'raw', 0, 'raw', True]

    for scale in scalings:
        for l_idx, (layer_name, n_features) in enumerate(zip(model.selected_layers, model.features_per_layer)):
            for ch_idx in range(n_features):
                idx = len(feature_info)
                feature_info[idx] = [idx, f'{layer_name}_{scale}_{idx:04d}', scale, layer_name, ch_idx, 'orig', True]

    extracted_features = append_grad_rot(extracted_features, add_grad, add_rot, feature_info, device=device)

    # select pixels in annotations

    features, targets = extract_annotated_pixels(extracted_features, annotation, full_annotation=full_annotation)
    features = pd.DataFrame(features)
    targets = pd.Series(targets)

    return features, targets, feature_info


def get_features_all_samples_rich(model, images, annotations, scalings=[1],
                                  full_annotation=True,
                                  add_grad=True,
                                  add_rot=True,
                                  order=0, device='cpu'):
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
        if True, add gradient features: gradient in x, gradient in y. Triples feature number. Default True.
    add_rot : bool
        if True, add rotor features: convolves grad features with corresponding "edge detector" and sums.
        Default True.
    order : int, optional
        Interpolation order for low scale resizing, by default 0
    use_min_features : bool, optional
        Use minimal number of features, by default True
    device : str, optional
        Device to use for computation, by default 'cpu'

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
                                                                       order=order, device=device
                                                                       )
        features_all_samples.append(features)
        targets_all_samples.append(targets)

    fill_useless(features_all_samples, feature_info)

    return features_all_samples, targets_all_samples, feature_info


def train_classifier(features, targets):
    """Train a random forest classifier given a set of features and targets."""

    # train model
    # split train/test
    X, X_test, y, y_test = train_test_split(features, targets,
                                            test_size=0.2,
                                            random_state=42)

    # train a random forest classififer
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(X, y)

    return random_forest


def load_trained_classifier(model_path):
    model_path = Path(model_path)
    random_forest = load(model_path)

    param = Param()
    with open(model_path.parent.joinpath('convpaint_params.yml')) as file:
        documents = yaml.full_load(file)
    for k in documents.keys():
        setattr(param, k, documents[k])

    return random_forest, param


# encapsulate of pytorch model that includes hooks and outputs for certain layers
class Hookmodel():
    """Class to extract features from a pytorch model using hooks on chosen layers.

    Parameters
    ----------
    model_name : str
        Name of model to use. Currently only 'vgg16' and 'single_layer_vgg16',
         'single_layer_vgg16_rgb' are supported.
    model : torch model, optional
        Model to extract features from, by default None
    use_cuda : bool, optional
        Use cuda, by default False
    param : Param, optional
        Parameters for model, by default None
        
    Attributes
    ----------
    model : torch model
        Model to extract features from
    outputs : list of torch tensors
        List of outputs from hooks
    features_per_layer : list of int
        Number of features per hooked layer
    module_list : list of lists
        List of hooked layers, their names, and their indices in the model (if applicable)
    """

    def __init__(self, model_name='vgg16', model=None, use_cuda=False, param=None):

        if model is not None:
            assert model_name is None, 'model_name must be None if model is not None'
            self.model = model
        else:
            if param is not None:
                model_name = param.model_name

            if model_name == 'vgg16':
                self.model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
                # self.transform =  models.VGG16_Weights.IMAGENET1K_V1.transforms()
            elif model_name == 'efficient_netb0':
                self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            elif model_name == 'single_layer_vgg16':
                self.model = load_single_layer_vgg16(keep_rgb=False)
            elif model_name == 'single_layer_vgg16_rgb':
                self.model = load_single_layer_vgg16(keep_rgb=True)
            elif model_name == 'dino_vits16':
                self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')

        if use_cuda:
            self.model = self.model.cuda()

        self.outputs = []
        self.features_per_layer = []
        self.selected_layers = []

        self.get_layer_dict()

        if model_name == 'single_layer_vgg16':
            self.register_hooks(list(self.module_dict.keys()))

        if param is not None:
            self.register_hooks(param.model_layers)

    def __call__(self, tensor_image):

        return self.model(tensor_image)

    def hook_normal(self, module, input, output):
        self.outputs.append(output)

    def hook_last(self, module, input, output):
        self.outputs.append(output)
        assert False

    def register_hooks(self, selected_layers):  # , selected_layer_pos):

        self.features_per_layer = []
        self.selected_layers = selected_layers.copy()
        for ind in range(len(selected_layers)):

            self.features_per_layer.append(
                self.module_dict[selected_layers[ind]].out_channels)

            if ind == len(selected_layers) - 1:
                self.module_dict[selected_layers[ind]].register_forward_hook(self.hook_last)
            else:
                self.module_dict[selected_layers[ind]].register_forward_hook(self.hook_normal)

    def get_layer_dict(self):
        """Create a flat list of all modules as well as a dictionary of modules with 
        keys describing the layers."""

        named_modules = list(self.model.named_modules())
        self.named_modules = [n for n in named_modules if len(list(n[1].named_modules())) == 1]
        self.module_id_dict = dict([(x[0] + ' ' + x[1].__str__(), x[0]) for x in self.named_modules])
        self.module_dict = dict([(x[0] + ' ' + x[1].__str__(), x[1]) for x in self.named_modules])


class Classifier():
    """"
    Class to segment images. Contains both the NN model needed to extract features
    and the classifier to predict the class of each pixel. By default loads the
    single_layer_vgg16 model and the classifier is None.
     
    Parameters
    ----------
    model_path : str, optional
        Path to RF model saved as joblib. Expects a parameters file in the same
        location
    
    Attributes
    ----------
    random_forest : sklearn RF classifier
        Classifier to predict the class of each pixel
    param : Param
        Parameters for model
    model : Hookmodel
        Model to extract features from the image

    """

    def __init__(self, model_path=None):

        self.random_forest = None
        self.param = None
        self.model = None

        if model_path is not None:
            self.load_model(model_path)

        else:
            self.default_model()

    def load_model(self, model_path):
        """Load a pretrained model by loading the joblib model
        and recreating the NN from the param file."""
        
        self.random_forest, self.param = load_trained_classifier(model_path)
        self.model = Hookmodel(param=self.param)

    def default_model(self):
        """Set default model to single_layer_vgg16."""
            
        self.model = Hookmodel(model_name='single_layer_vgg16')
        self.random_forest = None
        self.param = Param(
            model_name='single_layer_vgg16',
            model_layers=list(self.model.module_dict.keys()),
            scalings=[1,2],
            order=1,
            use_min_features=False,
        )

    def save_classifier(self, save_path):
        """Save the classifier to a joblib file and the parameters to a yaml file.
        
        Parameters
        ----------
        save_path : str
            Path to save files to
        """

        dump(self.random_forest, save_path)
        self.param.random_forest = save_path
        self.param.save_parameters(Path(save_path).parent.joinpath('convpaint_params.yml'))


    def segment_image_stack(self, image, save_path=None):
        """Segment an image stack using a pretrained model. If save_path is not
        None, save the zarr file to this path. Otherwise, return numpy array
        
        Parameters
        ----------
        image : np.ndarray
            2D or 3D image stack to segment, with dimensions n,y,x or y,x
        save_path : str
            Path to save zarr file to

        Returns
        -------
        np.ndarray
            Segmented image stack. Either 2D (single image) or 3D with n,y,x
            where n is the number of images in the stack.
        """

        if not ((image.ndim == 2) | (image.ndim == 3)):
            raise Exception(f'Image must be 2D or 3D, not {image.ndim}')
        single_image=False
        if image.ndim == 2:
            single_image = True
            image = np.expand_dims(image, axis=0)
        chunks = (1, image.shape[1], image.shape[2])

        if save_path is not None:
            im_out = zarr.open(save_path, mode='w', shape=image.shape,
                               chunks=chunks, dtype=np.uint8)
        else:
            im_out = np.zeros(image.shape, dtype=np.uint8)

        for i in range(image.shape[0]):
            im_out[i] = predict_image(
                image=image[i],
                model=self.model,
                classifier=self.random_forest,
                scalings=self.param.scalings,
                order=self.param.order,
                use_min_features=self.param.use_min_features,
                device='cpu',
                normalize=self.param.normalize,
                image_downsample=self.param.image_downsample)

        if save_path is None:
            if single_image:
                return im_out[0]
            else:
                return im_out