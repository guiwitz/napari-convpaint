from qtpy.QtWidgets import (QWidget, QPushButton,QVBoxLayout,
                            QLabel, QComboBox,QFileDialog, QListWidget,
                            QCheckBox, QAbstractItemView)

from joblib import dump, load
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import yaml

from napari_guitils.gui_structures import VHGroup, TabSet
#from napari_annotation_project.project_widget import ProjectWidget 
from .conv_paint_utils import predict_image, load_nn_model
from .conv_parameters import Param
from .conv_paint_utils import (Hookmodel, get_features_current_layers, train_classifier)

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
    
    def __init__(self, napari_viewer, channel=None, parent=None, project=False):
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        
        #self.param = param
        #if self.param is None:
        self.param = Param(channel=channel)
        
        #self.scalings = [1,2]
        self.model = None
        self.random_forest = None
        self.project_widget = None
        self.features_per_layer = None

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.tab_names = ['Annotation', 'Files', 'Model']
        self.tabs = TabSet(self.tab_names)
        self.tabs.setTabEnabled(self.tabs.tab_names.index('Files'), False)
        self.main_layout.addWidget(self.tabs)

        #self._layout = QVBoxLayout()
        #self.setLayout(self._layout)

        self.select_layer_widget = QComboBox()
        self.select_layer_widget.addItems([x.name for x in self.viewer.layers])
        self.tabs.add_named_tab('Annotation', self.select_layer_widget)

        self.settings_vgroup = VHGroup('Settings', orientation='G')
        self.tabs.add_named_tab('Annotation', self.settings_vgroup.gbox)

        self.num_scales_combo = QComboBox()
        self.num_scales_combo.addItems(['[1]', '[1,2]', '[1,2,4]', '[1,2,4,8]'])
        self.num_scales_combo.setCurrentText('[1,2]')
        self.settings_vgroup.glayout.addWidget(QLabel('Number of scales'), 0, 0)
        self.settings_vgroup.glayout.addWidget(self.num_scales_combo, 0, 1)

        self.add_layers_btn = QPushButton('Add annotation/predict layers')
        self.tabs.add_named_tab('Annotation', self.add_layers_btn)

        self.update_model_btn = QPushButton('Train model on singe image')
        self.tabs.add_named_tab('Annotation', self.update_model_btn)

        self.update_model_on_project_btn = QPushButton('Train model on full project')
        self.tabs.add_named_tab('Annotation', self.update_model_on_project_btn)
        if project is False:
            self.update_model_on_project_btn.setEnabled(False)

        self.prediction_btn = QPushButton('Predict single frame')
        self.tabs.add_named_tab('Annotation', self.prediction_btn)

        self.prediction_all_btn = QPushButton('Predict all frames')
        self.tabs.add_named_tab('Annotation', self.prediction_all_btn)

        self.save_model_btn = QPushButton('Save trained model')
        self.tabs.add_named_tab('Annotation', self.save_model_btn)

        self.load_model_btn = QPushButton('Load trained model')
        self.tabs.add_named_tab('Annotation', self.load_model_btn)

        self.check_use_project = QCheckBox('Use project')
        self.check_use_project.setChecked(False)
        self.tabs.add_named_tab('Annotation', self.check_use_project)

        self.qcombo_model_type = QComboBox()
        self.qcombo_model_type.addItems(['vgg16', 'single_layer_vgg16'])
        self.tabs.add_named_tab('Model', self.qcombo_model_type)

        self.load_nnmodel_btn = QPushButton('Load nn model')
        self.tabs.add_named_tab('Model', self.load_nnmodel_btn)

        self.set_nnmodel_outputs_btn = QPushButton('Set model outputs')
        self.tabs.add_named_tab('Model', self.set_nnmodel_outputs_btn)

        self.model_output_selection = QListWidget()
        self.model_output_selection.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tabs.add_named_tab('Model', self.model_output_selection)

        if project is True:
            self._add_project()

        self.add_connections()

    def _add_project(self, event=None):
        """Add widget for multi-image project management"""

        if self.check_use_project.isChecked():
            if self.project_widget is None:
                from napari_annotation_project.project_widget import ProjectWidget
                self.project_widget = ProjectWidget(napari_viewer=self.viewer)
                self.tabs.add_named_tab('Files', self.project_widget)
            
            self.tabs.setTabEnabled(self.tabs.tab_names.index('Files'), True)
            self.update_model_on_project_btn.setEnabled(True)
        else:
            self.tabs.setTabEnabled(self.tabs.tab_names.index('Files'), False)
            self.update_model_on_project_btn.setEnabled(False)


    def add_connections(self):

        self.select_layer_widget.currentIndexChanged.connect(self.select_layer)
        self.num_scales_combo.currentIndexChanged.connect(self.update_scalings)
        self.viewer.layers.events.removed.connect(self.update_layer_list)
        self.viewer.layers.events.inserted.connect(self.update_layer_list)

        self.add_layers_btn.clicked.connect(self.add_annotation_layer)
        self.update_model_btn.clicked.connect(self.update_classifier)
        self.update_model_on_project_btn.clicked.connect(self.update_classifier_on_project)
        self.prediction_btn.clicked.connect(self.predict)
        self.prediction_all_btn.clicked.connect(self.predict_all)
        self.save_model_btn.clicked.connect(self.save_model)
        self.load_model_btn.clicked.connect(self.load_classifier)
        self.check_use_project.stateChanged.connect(self._add_project)

        self.load_nnmodel_btn.clicked.connect(self._on_load_nnmodel)
        self.set_nnmodel_outputs_btn.clicked.connect(self._on_click_define_model_outputs)

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

    def _create_output_selection(self):
        """Update list of selectable layers"""

        self.model_output_selection.clear()
        self.model_output_selection.addItems(self.model.module_dict.keys())

    def _on_click_define_model_outputs(self, event=None):
        """Using hooks setup model to give outputs at selected layers."""
        
        model_type = self.qcombo_model_type.currentText()
        self.model = Hookmodel(model_name=model_type)

        selected_layer = []
        layer_pos = []
        selected_rows = self.model_output_selection.selectedIndexes()
        selected_rows = [x.row() for x in selected_rows]
        for row in selected_rows:
            
            curr_name = self.model.module_list[row][1]
            curr_pos = self.model.module_list[row][2]

            selected_layer.append(curr_name)
            layer_pos.append(curr_pos)

        self.model.register_hooks(selected_layer=selected_layer, selected_layer_pos=layer_pos)
                                  

    def _on_load_nnmodel(self, event=None):
        """Load a neural network model. Create list of selectable layers."""
            
        self.model = Hookmodel(self.qcombo_model_type.currentText())
        self._create_output_selection()
        # if model has a single layer output, automatically initialize it
        if len(self.model.module_list) == 1:
            self.model_output_selection.setCurrentRow(0)
            self._on_click_define_model_outputs()
            self.set_nnmodel_outputs_btn.setEnabled(False)
            self.model_output_selection.setEnabled(False)
        else:
            self.set_nnmodel_outputs_btn.setEnabled(True)
            self.model_output_selection.setEnabled(True)


    def update_classifier(self):
        """Given a set of new annotations, update the random forest model."""

        if self.model is None:
            raise Exception('No feature generator model loaded')

        features, targets = get_features_current_layers(
            model=self.model,
            image=self.viewer.layers[self.param.channel].data,
            annotations=self.viewer.layers['annotations'].data,
            scalings=self.param.scalings,
        )
        self.random_forest = train_classifier(features, targets)

    def update_classifier_on_project(self):
        """Train classifier on all annotations in project."""

        if self.model is None:
            #self.model = load_nn_model()
            raise Exception('No model loaded')

        num_files = len(self.projet_widget.params.file_paths)
        if num_files == 0:
            raise Exception('No files found')
        
        all_features, all_targets = [], []
        for ind in range(num_files):
            self.projet_widget.file_list.setCurrentRow(ind)

            features, targets = get_features_current_layers(
                model=self.model,
                image=self.viewer.layers[self.param.channel].data,
                annotations=self.viewer.layers['annotations'].data,
                scalings=self.param.scalings,
            )
            if features is None:
                continue
            all_features.append(features)
            all_targets.append(targets)
        
        all_features = np.concatenate(all_features, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        self.random_forest = train_classifier(all_features, all_targets)
        

    def predict(self):
        """Predict the segmentation of the currently viewed frame based 
        on a RF model trained with annotations"""

        if self.model is None:
            #self.model = load_nn_model()
            raise Exception('No feature generator model loaded')
        
        if self.random_forest is None:
            self.update_classifier()

        self.check_prediction_layer_exists()
        
        if self.viewer.dims.ndim > 2:
            step = self.viewer.dims.current_step[0]
            image = self.viewer.layers[self.param.channel].data[step]
            predicted_image = predict_image(image, self.model, self.random_forest, self.param.scalings)
            self.viewer.layers['prediction'].data[step] = predicted_image
        else:
            image = self.viewer.layers[self.param.channel].data
            predicted_image = predict_image(image, self.model, self.random_forest, self.param.scalings)
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
            predicted_image = predict_image(image, self.model, self.random_forest, self.param.scalings)
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

        self.param.save_parameters(save_file.parent.joinpath('convpaint_params.yml'))

    def load_classifier(self):
        """Select classifier model file to load."""

        dialog = QFileDialog()
        save_file, _ = dialog.getOpenFileName(self, "Choose model", None, "JOBLIB (*.joblib)")
        save_file = Path(save_file)
        self.random_forest = load(save_file)
        
        #self.param.random_forest = save_file

        self.param = Param()
        with open(save_file.parent.joinpath('convpaint_params.yml')) as file:
            documents = yaml.full_load(file)
        for k in documents.keys():
            setattr(self.param, k, documents[k])