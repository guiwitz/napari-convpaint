from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import (QWidget, QPushButton,
QVBoxLayout, QLabel, QComboBox,QFileDialog)

from joblib import dump, load
from pathlib import Path
import numpy as np
import pandas as pd
import torchvision.models as models
from torch import nn
import yaml

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from napari_guitils.gui_structures import VHGroup

from .conv_paint_utils import filter_image, predict_image, load_nn_model
from .conv_parameters import Param

class ConvPaintWidget(QWidget):
    """
    Implementation of a napari widget for interactive segmentation performed
    via a random forest model trained on annotations. The filters used to 
    generate the model features are taken from the first layer of a VGG16 model
    as proposed here: https://github.com/hinderling/napari_pixel_classifier

    Parameters
    ----------
    napari_viewer: napari.Viewer
        main napari viwer
    channel: str
        layer name to use for features extraction/classification
    """
    
    def __init__(self, napari_viewer, channel=None, parent=None):
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        
        #self.param = param
        #if self.param is None:
        self.param = Param(channel=channel)
        
        #self.scalings = [1,2]
        self.model = None
        self.random_forest = None

        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        self.select_layer_widget = QComboBox()
        self.select_layer_widget.addItems([x.name for x in self.viewer.layers])
        self._layout.addWidget(self.select_layer_widget)
        self.select_layer_widget.currentIndexChanged.connect(self.select_layer)

        self.settings_vgroup = VHGroup('Settings', orientation='G')
        self._layout.addWidget(self.settings_vgroup.gbox)

        self.num_scales_combo = QComboBox()
        self.num_scales_combo.addItems(['[1]', '[1,2]', '[1,2,4]', '[1,2,4,8]'])
        self.num_scales_combo.setCurrentText('[1,2]')
        self.num_scales_combo.currentIndexChanged.connect(self.update_scalings)
        self.settings_vgroup.glayout.addWidget(QLabel('Number of scales'), 0, 0)
        self.settings_vgroup.glayout.addWidget(self.num_scales_combo, 0, 1)

        self.add_layers_btn = QPushButton('Add annotation/predict layers')
        self.add_layers_btn.clicked.connect(self.add_annotation_layer)
        self._layout.addWidget(self.add_layers_btn)

        self.update_model_btn = QPushButton('Update model')
        self.update_model_btn.clicked.connect(self.update_model)
        self._layout.addWidget(self.update_model_btn)

        self.prediction_btn = QPushButton('Predict single frame')
        self.prediction_btn.clicked.connect(self.predict)
        self._layout.addWidget(self.prediction_btn)

        self.prediction_all_btn = QPushButton('Predict all frames')
        self.prediction_all_btn.clicked.connect(self.predict_all)
        self._layout.addWidget(self.prediction_all_btn)

        self.save_model_btn = QPushButton('Save trained model')
        self.save_model_btn.clicked.connect(self.save_model)
        self._layout.addWidget(self.save_model_btn)

        self.load_model_btn = QPushButton('Load trained model')
        self.load_model_btn.clicked.connect(self.load_model)
        self._layout.addWidget(self.load_model_btn)

        self.load_annotations_btn = QPushButton('Load annotations')
        self.load_annotations_btn.clicked.connect(self.load_annotations)
        self._layout.addWidget(self.load_annotations_btn)

        self.viewer.layers.events.removed.connect(self.update_layer_list)
        self.viewer.layers.events.inserted.connect(self.update_layer_list)

    def update_layer_list(self, event):
        
        keep_channel = self.param.channel
        self.select_layer_widget.clear()
        self.select_layer_widget.addItems([x.name for x in self.viewer.layers])
        if keep_channel in [x.name for x in self.viewer.layers]:
            self.select_layer_widget.setCurrentText(keep_channel)

    def select_layer(self):

        self.param.channel = self.select_layer_widget.currentText()

    def add_annotation_layer(self):

        self.viewer.add_labels(
            data=np.zeros((self.viewer.layers[self.param.channel].data.shape), dtype=np.uint8),
            name='annotations'
            )
        self.viewer.add_labels(
            data=np.zeros((self.viewer.layers[self.param.channel].data.shape), dtype=np.uint8),
            name='prediction'
            )

    def update_scalings(self):

        self.param.scalings = eval(self.num_scales_combo.currentText())
        #self.param.scalings = self.scalings
        
    def update_model(self):
        """Given a set of new annotations, update the random forest model."""

        if self.model is None:
            self.model = load_nn_model()
       
        n_features = 64

        # get indices of first dimension of non-empty annotations. Gives t/z indices
        non_empty = np.unique(np.where(self.viewer.layers['annotations'].data > 0)[0])
        if len(non_empty) == 0:
            raise Exception('No annotations found')
       # elif non_empty.ndim == 1:
       #     non_empty = [non_empty]

        all_values = []
        # iterating over non_empty iteraties of t/z for 3D data
        for ind, t in enumerate(non_empty):

            if self.viewer.layers[self.param.channel].data.ndim == 2:
                image = self.viewer.layers[self.param.channel].data
                annot = self.viewer.layers['annotations'].data
            else:
                image = self.viewer.layers[self.param.channel].data[t]
                annot = self.viewer.layers['annotations'].data[t]

            #image = self.viewer.layers[self.param.channel].data[t]

            full_annotation = np.ones((n_features, image.shape[0], image.shape[1]),dtype=np.bool_)
            full_annotation = full_annotation * annot>0

            all_scales = filter_image(image, self.model, self.param.scalings)
            all_values_scales=[]
            for a in all_scales:
                extract = a[0, full_annotation]
                all_values_scales.append(np.reshape(extract, (n_features, int(extract.shape[0]/n_features))).T)
            all_values.append(np.concatenate(all_values_scales, axis=1))

        all_values = np.concatenate(all_values,axis=0)
        features = pd.DataFrame(all_values)
        target_im = self.viewer.layers['annotations'].data[self.viewer.layers['annotations'].data>0]
        targets = pd.Series(target_im)

        # train model
        #split train/test
        X, X_test, y, y_test = train_test_split(features, targets, 
                                            test_size = 0.2, 
                                            random_state = 42)

        #train a random forest classififer
        self.random_forest = RandomForestClassifier(n_estimators=100)
        self.random_forest.fit(X, y)

    def predict(self):
        """Predict the segmentation of the currently viewed frame based 
        on a RF model trained with annotations"""

        if self.model is None:
            self.model = load_nn_model()
        if self.random_forest is None:
            self.update_model()

        self.check_prediction_layer_exists()
        
        if self.viewer.dims.ndim > 2:
            step = self.viewer.dims.current_step[0]
            image = self.viewer.layers[self.param.channel].data[step]
            predicted_image = predict_image(image, self.model, self.random_forest)
            self.viewer.layers['prediction'].data[step] = predicted_image
        else:
            image = self.viewer.layers[self.param.channel].data
            predicted_image = predict_image(image, self.model, self.random_forest)
            self.viewer.layers['prediction'].data = predicted_image
        
        self.viewer.layers['prediction'].refresh()

    def predict_all(self):
        """Predict the segmentation of all frames based 
        on a RF model trained with annotations"""

        if self.random_forest is None:
            raise Exception('No model found. Please train a model first.')
        
        if self.model is None:
            self.model = load_nn_model()

        self.check_prediction_layer_exists()

        for step in range(self.viewer.dims.nsteps[0]):
            image = self.viewer.layers[self.param.channel].data[step]
            predicted_image = predict_image(image, self.model, self.random_forest)
            self.viewer.layers['prediction'].data[step] = predicted_image

    def check_prediction_layer_exists(self):

        layer_names = [x.name for x in self.viewer.layers]
        if 'prediction' not in layer_names:
            self.viewer.add_labels(
                data=np.zeros((self.viewer.layers[self.param.channel].data.shape), dtype=np.uint8),
                name='prediction'
                )

    def save_model(self):
        """Select file where to save the classifier model."""

        # save sklearn model
        dialog = QFileDialog()
        save_file, _ = dialog.getSaveFileName(self, "Save model", None, "JOBLIB (*.joblib)")
        save_file = Path(save_file)
        dump(self.random_forest, save_file)
        self.param.random_forest = save_file#.as_posix()

        # save annotations
        dfs = []
        ar = self.viewer.layers['annotations'].data
        for i in range(1, 1+int(ar.max())):
            coords = np.where(ar==i)
            coords_dict = {f'd{j}':coords[j] for j in range(len(coords))}
            df = pd.DataFrame(coords_dict)
            df['val'] = i
            dfs.append(df)
        df = pd.concat(dfs)
        df.to_csv(save_file.parent.joinpath('annotations.csv'), index=False)

        # save parameters
        self.param.image_source = self.viewer.layers[self.param.channel].source.path
        self.param.save_parameters(save_file.parent.joinpath('convpaint_params.yml'))

    def load_model(self):
        """Select classifier model file to load."""

        dialog = QFileDialog()
        save_file, _ = dialog.getOpenFileName(self, "Choose model", None, "JOBLIB (*.joblib)")
        save_file = Path(save_file)
        self.random_forest = load(save_file)
        self.param.random_forest = save_file

    def load_annotations(self):
        """Select file with annotations to load."""

        project_path = Path(str(QFileDialog.getExistingDirectory(self, "Select a project folder to load",options=QFileDialog.DontUseNativeDialog)))
        if not project_path.joinpath('convpaint_params.yml').exists():
            raise FileNotFoundError(f"Project {project_path} does not exist")

        self.param = Param()
        with open(project_path.joinpath('convpaint_params.yml')) as file:
            documents = yaml.full_load(file)
        for k in documents.keys():
            setattr(self.param, k, documents[k])

        if self.param.image_source is None:
            if len(self.viewer.layers) == 0:
                raise Exception("No image is loaded and it seems the convpaint\
                    training has been done on a larger project. Load the project first.")
        else:
            image_file = Path(self.param.image_source)
            self.viewer.open(image_file)
        self.add_annotation_layer()

        annotation_file = Path(project_path.joinpath('annotations.csv'))
        df = pd.read_csv(annotation_file)
        ar = np.zeros((self.viewer.layers[self.param.channel].data.shape), dtype=np.uint8)
        for i in range(1, 1+int(df.val.max())):
            row_cols = df[df.val==i].drop('val', axis=1).values
            index_list = [row_cols[:,x] for x in range(row_cols.shape[1])]
            ar[tuple(index_list)]=i

        self.viewer.layers['annotations'].data = ar
        self.viewer.layers['annotations'].refresh()