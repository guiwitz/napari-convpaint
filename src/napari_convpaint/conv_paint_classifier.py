from pathlib import Path
import yaml
from joblib import dump, load
import zarr
import numpy as np

from .conv_paint_param import Param
from .conv_paint_nnlayers import Hookmodel
from .conv_paint_utils import compute_image_stats, normalize_image


def load_trained_classifier(model_path):
    model_path = Path(model_path)
    classifier = load(model_path)

    param = Param()
    with open(model_path.parent.joinpath('convpaint_params.yml')) as file:
        documents = yaml.full_load(file)
    for k in documents.keys():
        setattr(param, k, documents[k])

    return classifier, param


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
    classifier : CatBoost classifier
        Classifier to predict the class of each pixel
    param : Param
        Parameters for model
    fe_model : Hookmodel
        Model to extract features from the image

    """

    def __init__(self, model_path=None):

        self.classifier = None
        self.param = None
        self.fe_model = None

        if model_path is not None:
            self.load_model(model_path)

        else:
            self.default_model()

    def load_model(self, model_path):
        """Load a pretrained model by loading the joblib model
        and recreating the NN from the param file."""
        
        self.classifier, self.param = load_trained_classifier(model_path)
        
        self.fe_model = Hookmodel(param=self.param)

    def default_model(self):
        """Set default model to single_layer_vgg16."""
            
        self.fe_model = Hookmodel(model_name='single_layer_vgg16')
        self.classifier = None
        self.param = self.fe_model.get_default_param()

    def save_classifier(self, save_path):
        """Save the classifier to a joblib file and the parameters to a yaml file.
        
        Parameters
        ----------
        save_path : str
            Path to save files to
        """

        dump(self.classifier, save_path)
        self.param.classifier = save_path
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

        if self.param.normalize:
            image_mean, image_std = compute_image_stats(image)
            image = normalize_image(image, image_mean, image_std)

        for i in range(image.shape[0]):
            im_out[i] = self.fe_model.predict_image(
                image=image[i],
                classifier=self.classifier,
                param=self.param)

        if save_path is None:
            if single_image:
                return im_out[0]
            else:
                return im_out