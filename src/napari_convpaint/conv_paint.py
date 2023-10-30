from qtpy.QtWidgets import (QWidget, QPushButton,QVBoxLayout,
                            QLabel, QComboBox,QFileDialog, QListWidget,
                            QCheckBox, QAbstractItemView, QGridLayout, QSpinBox)
from qtpy.QtCore import Qt
from magicgui.widgets import create_widget
import napari
from joblib import dump, load
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import yaml

from napari_guitils.gui_structures import VHGroup, TabSet
#from napari_annotation_project.project_widget import ProjectWidget 
from .conv_paint_utils import predict_image
from .conv_parameters import Param
from .conv_paint_utils import (Hookmodel, get_features_current_layers,
                               train_classifier, load_trained_classifier)

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
    project: bool
        use project widget for multi-image project management
    """
    
    def __init__(self, napari_viewer, parent=None, project=False):
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        
        self.param = Param()
        self.model = None
        self.random_forest = None
        self.project_widget = None
        self.features_per_layer = None
        self.selected_channel = None

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.tab_names = ['Annotation', 'Files', 'Model']
        self.tabs = TabSet(self.tab_names, tab_layouts=[QGridLayout(), None, QGridLayout()])
        self.tabs.setTabEnabled(self.tabs.tab_names.index('Files'), False)
        self.main_layout.addWidget(self.tabs)

        self.tabs.widget(0).layout().setAlignment(Qt.AlignTop)
        self.select_layer_widget = create_widget(annotation=napari.layers.Image, label='Pick image')
        self.select_layer_widget.reset_choices()
        self.viewer.layers.events.inserted.connect(self.select_layer_widget.reset_choices)
        self.viewer.layers.events.removed.connect(self.select_layer_widget.reset_choices)
        #self.select_layer_widget = QComboBox()
        #self.select_layer_widget.addItems([x.name for x in self.viewer.layers])
        self.tabs.add_named_tab('Annotation', self.select_layer_widget.native, grid_pos=[0,0,1,2])

        self.add_layers_btn = QPushButton('Add annotation/predict layers')
        self.tabs.add_named_tab('Annotation', self.add_layers_btn, grid_pos=[1,0,1,2])

        self.update_model_btn = QPushButton('Train model on single image')
        self.tabs.add_named_tab('Annotation', self.update_model_btn, grid_pos=[2,0,1,1])

        self.update_model_on_project_btn = QPushButton('Train model on full project')
        self.tabs.add_named_tab('Annotation', self.update_model_on_project_btn, grid_pos=[2,1,1,1])
        if project is False:
            self.update_model_on_project_btn.setEnabled(False)

        self.prediction_btn = QPushButton('Predict single frame')
        self.tabs.add_named_tab('Annotation', self.prediction_btn, grid_pos=[3,0,1,1])

        self.prediction_all_btn = QPushButton('Predict all frames')
        self.tabs.add_named_tab('Annotation', self.prediction_all_btn, grid_pos=[3,1,1,1])

        self.save_model_btn = QPushButton('Save trained model')
        self.tabs.add_named_tab('Annotation', self.save_model_btn, grid_pos=[4,0,1,1])

        self.load_model_btn = QPushButton('Load trained model')
        self.tabs.add_named_tab('Annotation', self.load_model_btn, grid_pos=[4,1,1,1])

        self.check_use_project = QCheckBox('Use multiple files')
        self.check_use_project.setChecked(False)
        self.tabs.add_named_tab('Annotation', self.check_use_project, grid_pos=[5,0,1,1])

        self.check_use_default_model = QCheckBox('Use default model')
        self.check_use_default_model.setChecked(True)
        self.tabs.add_named_tab('Annotation', self.check_use_default_model, grid_pos=[6,0,1,1])

        self.check_dims_is_channels = QCheckBox('Multichannel image')
        self.check_dims_is_channels.setChecked(False)
        self.check_dims_is_channels.setToolTip('If checked, the additional dimensions is not treated as time-lapse but as channels.')
        self.tabs.add_named_tab('Annotation', self.check_dims_is_channels, grid_pos=[7,0,1,1])

        self.spin_downsample = QSpinBox()
        self.spin_downsample.setMinimum(1)
        self.spin_downsample.setMaximum(10)
        self.spin_downsample.setValue(1)
        self.spin_downsample.setToolTip('Reduce image size for faster computing.')
        self.tabs.add_named_tab('Annotation', QLabel('Downsample'), grid_pos=[8,0,1,1])
        self.tabs.add_named_tab('Annotation', self.spin_downsample, grid_pos=[8,1,1,1])

        self.qcombo_model_type = QComboBox()
        self.qcombo_model_type.addItems([
            'vgg16', 'efficient_netb0', 'single_layer_vgg16', 'single_layer_vgg16_rgb'])
        self.tabs.add_named_tab('Model', self.qcombo_model_type, [0,0,1,2])

        self.load_nnmodel_btn = QPushButton('Load nn model')
        self.tabs.add_named_tab('Model', self.load_nnmodel_btn, [1,0,1,2])

        self.set_nnmodel_outputs_btn = QPushButton('Set model outputs')
        self.tabs.add_named_tab('Model', self.set_nnmodel_outputs_btn, [2,0,1,2])

        self.model_output_selection = QListWidget()
        self.model_output_selection.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tabs.add_named_tab('Model', self.model_output_selection, [3,0,1,2])

        self.num_scales_combo = QComboBox()
        self.num_scales_combo.addItems(['[1]', '[1,2]', '[1,2,4]', '[1,2,4,8]'])
        self.num_scales_combo.setCurrentText('[1,2]')
        self.tabs.add_named_tab('Model', QLabel('Number of scales'), [4,0,1,1])
        self.tabs.add_named_tab('Model', self.num_scales_combo, [4,1,1,1])

        self.check_use_min_features = QCheckBox('Use min features')
        self.check_use_min_features.setChecked(False)
        self.tabs.add_named_tab('Model', self.check_use_min_features, [5,0,1,2])

        self.spin_interpolation_order = QSpinBox()
        self.spin_interpolation_order.setMinimum(0)
        self.spin_interpolation_order.setMaximum(5)
        self.spin_interpolation_order.setValue(1)
        self.tabs.add_named_tab('Model', QLabel('Interpolation order'), [6,0,1,1])
        self.tabs.add_named_tab('Model', self.spin_interpolation_order, [6,1,1,1])
        self.tabs.setTabEnabled(self.tabs.tab_names.index('Model'), False)

        self.check_normalize = QCheckBox('Normalize')
        self.check_normalize.setChecked(True)
        self.tabs.add_named_tab('Model', self.check_normalize, [7,0,1,1])

        self.check_use_cuda = QCheckBox('Use cuda')
        self.check_use_cuda.setChecked(False)
        self.tabs.add_named_tab('Model', self.check_use_cuda, grid_pos=[7,1,1,1])


        if project is True:
            self._add_project()

        self.add_connections()
        self.select_layer()

    def _add_project(self, event=None):
        """Add widget for multi-image project management"""

        if self.check_use_project.isChecked():
            if self.project_widget is None:
                from napari_annotation_project.project_widget import ProjectWidget
                self.project_widget = ProjectWidget(napari_viewer=self.viewer)

                #self.tabs.add_named_tab('Files', self.project_widget)
                self.tabs.add_named_tab('Files', self.project_widget.file_list)
                self.tabs.add_named_tab('Files', self.project_widget.btn_remove_file)
                self.tabs.add_named_tab('Files', self.project_widget.btn_add_file)
                self.tabs.add_named_tab('Files', self.project_widget.btn_save_annotation)
                self.tabs.add_named_tab('Files', self.project_widget.btn_load_project)
            
            self.tabs.setTabEnabled(self.tabs.tab_names.index('Files'), True)
            self.update_model_on_project_btn.setEnabled(True)
        else:
            self.tabs.setTabEnabled(self.tabs.tab_names.index('Files'), False)
            self.update_model_on_project_btn.setEnabled(False)

    def _set_custom_model(self, event=None):
        """Add widget for custom model management"""

        if self.check_use_default_model.isChecked():
            self.tabs.setTabEnabled(self.tabs.tab_names.index('Model'), False)
            self.set_default_model()
        else:
            self.tabs.setTabEnabled(self.tabs.tab_names.index('Model'), True)


    def add_connections(self):
        
        self.select_layer_widget.changed.connect(self.select_layer)
        self.viewer.layers.events.removed.connect(self.reset_model)
        self.num_scales_combo.currentIndexChanged.connect(self.update_scalings)

        self.add_layers_btn.clicked.connect(self.add_annotation_layer)
        self.update_model_btn.clicked.connect(self.update_classifier)
        self.update_model_on_project_btn.clicked.connect(self.update_classifier_on_project)
        self.prediction_btn.clicked.connect(self.predict)
        self.prediction_all_btn.clicked.connect(self.predict_all)
        self.save_model_btn.clicked.connect(self.save_model)
        self.load_model_btn.clicked.connect(self.load_classifier)
        self.check_use_project.stateChanged.connect(self._add_project)
        self.check_use_default_model.stateChanged.connect(self._set_custom_model)

        self.load_nnmodel_btn.clicked.connect(self._on_load_nnmodel)
        self.set_nnmodel_outputs_btn.clicked.connect(self._on_click_define_model_outputs)

    '''def update_layer_list(self, event):
        
        keep_channel = None
        if self.selected_channel is not None:
            keep_channel = self.selected_channel
        self.select_layer_widget.clear()
        self.select_layer_widget.addItems([x.name for x in self.viewer.layers])
        if keep_channel in [x.name for x in self.viewer.layers]:
            self.select_layer_widget.setCurrentText(keep_channel)
        else:
            if self.viewer.layers:
                self.select_layer_widget.setCurrentText(self.viewer.layers[0].name)'''

    def select_layer(self, newtext=None):
        
        self.selected_channel = self.select_layer_widget.native.currentText()        

    def reset_model(self, event=None):

        if len(self.viewer.layers) == 0:
            self.model = None

    def add_annotation_layer(self):
        """Add annotation and prediction layers to viewer."""

        if self.viewer.layers[self.selected_channel].rgb:
            layer_shape = self.viewer.layers[self.selected_channel].data.shape[0:2]
        elif self.check_dims_is_channels.isChecked():
            layer_shape = self.viewer.layers[self.selected_channel].data.shape[-2::]
        else:
            layer_shape = self.viewer.layers[self.selected_channel].data.shape

        self.viewer.add_labels(
            data=np.zeros((layer_shape), dtype=np.uint8),
            name='annotations'
            )
        self.viewer.add_labels(
            data=np.zeros((layer_shape), dtype=np.uint8),
            name='prediction'
            )
        self.viewer.layers.selection.active = self.viewer.layers['annotations']

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

        selected_layers = self.get_selected_layers_names()
        self.model.register_hooks(selected_layers=selected_layers)

    def get_selected_layers_names(self):
        """Get names of selected layers."""

        selected_rows = self.model_output_selection.selectedItems()
        selected_layers = [x.text() for x in selected_rows]
        return selected_layers
                                  

    def _on_load_nnmodel(self, event=None):
        """Load a neural network model. Create list of selectable layers."""
            
        self.model = Hookmodel(self.qcombo_model_type.currentText(), use_cuda=self.check_use_cuda.isChecked())
        self._create_output_selection()
        # if model has a single layer output, automatically initialize it
        if len(self.model.named_modules) == 1:
            self.model_output_selection.setCurrentRow(0)
            self._on_click_define_model_outputs()
            self.set_nnmodel_outputs_btn.setEnabled(False)
            self.model_output_selection.setEnabled(False)
        else:
            self.set_nnmodel_outputs_btn.setEnabled(True)
            self.model_output_selection.setEnabled(True)

    def set_default_model(self, keep_rgb=False):
        """Set default model."""
        if keep_rgb:
            self.qcombo_model_type.setCurrentText('single_layer_vgg16_rgb')
        else:
            self.qcombo_model_type.setCurrentText('single_layer_vgg16')
        self.num_scales_combo.setCurrentText('[1,2]')
        self.spin_interpolation_order.setValue(1)
        self.check_use_min_features.setChecked(True)
        self._on_load_nnmodel()

    def update_classifier(self):
        """Given a set of new annotations, update the random forest model."""

        unique_labels = np.unique(self.viewer.layers['annotations'].data)
        if (not 1 in unique_labels) | (not 2 in unique_labels):
            raise Exception('You need annoations for foreground and background')
        
        if self.model is None:
            if self.check_use_default_model.isChecked():
                
                # use 2d input for 2d images or if 3d input does not represent channels
                # use 3d input for rgb or if 3d input is channels and has dims 3
                if self.viewer.layers[self.selected_channel].ndim == 2:
                    if self.viewer.layers[self.selected_channel].rgb:
                        self.set_default_model(keep_rgb=True)
                    else:
                        self.set_default_model()
                
                else:
                    if self.check_dims_is_channels.isChecked():
                        if self.viewer.layers[self.selected_channel].data.shape[0] != 3:
                            raise Exception(f'your input has dimensions {self.viewer.layers[self.selected_channel].data.shape}, but the default model only works with 3 channel images')
                        self.set_default_model(keep_rgb=True)
                    else:
                        self.set_default_model()
            else:
                raise Exception('You have to define and load a model first')
        
        device = 'cuda' if self.check_use_cuda.isChecked() else 'cpu'

        data_to_pass = self.viewer.layers[self.selected_channel].data
        if self.viewer.layers[self.selected_channel].rgb:
            data_to_pass = np.moveaxis(data_to_pass, 2, 0)
        features, targets = get_features_current_layers(
            model=self.model,
            image=data_to_pass,
            annotations=self.viewer.layers['annotations'].data,
            scalings=self.param.scalings,
            order=self.spin_interpolation_order.value(),
            use_min_features=self.check_use_min_features.isChecked(),
            device=device,
            normalize=self.check_normalize.isChecked(),
            image_downsample=self.spin_downsample.value()
        )
        self.random_forest = train_classifier(features, targets)

    def update_classifier_on_project(self):
        """Train classifier on all annotations in project."""

        if self.model is None:
            if self.check_use_default_model.isChecked():
                self.set_default_model()
            else:
                raise Exception('You have to define and load a model first')
        device = 'cuda' if self.check_use_cuda.isChecked() else 'cpu'

        num_files = len(self.project_widget.params.file_paths)
        if num_files == 0:
            raise Exception('No files found')
        
        all_features, all_targets = [], []
        for ind in range(num_files):
            self.project_widget.file_list.setCurrentRow(ind)

            data_to_pass = self.viewer.layers[self.selected_channel].data
            if self.viewer.layers[self.selected_channel].rgb:
                data_to_pass = np.moveaxis(data_to_pass, 2, 0)

            features, targets = get_features_current_layers(
                model=self.model,
                image=data_to_pass,
                annotations=self.viewer.layers['annotations'].data,
                scalings=self.param.scalings,
                order=self.spin_interpolation_order.value(),
                use_min_features=self.check_use_min_features.isChecked(),
                device=device,
                normalize=self.check_normalize.isChecked(),
                image_downsample=self.spin_downsample.value()
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
            if self.check_use_default_model.isChecked():
                self.set_default_model()
            else:
                raise Exception('You have to define and load a model first')
        device = 'cuda' if self.check_use_cuda.isChecked() else 'cpu'
        
        if self.random_forest is None:
            self.update_classifier()

        self.check_prediction_layer_exists()
        
        if (self.viewer.dims.ndim > 2) & (not self.check_dims_is_channels.isChecked()):
            step = self.viewer.dims.current_step[0]
            image = self.viewer.layers[self.selected_channel].data[step]
            predicted_image = predict_image(
                image, self.model, self.random_forest, self.param.scalings,
                order=self.spin_interpolation_order.value(),
                use_min_features=self.check_use_min_features.isChecked(),
                device=device, normalize=self.check_normalize.isChecked(),
                image_downsample=self.spin_downsample.value()
            )
            self.viewer.layers['prediction'].data[step] = predicted_image
        else:

            data_to_pass = self.viewer.layers[self.selected_channel].data
            if self.viewer.layers[self.selected_channel].rgb:
                data_to_pass = np.moveaxis(data_to_pass, 2, 0)
            predicted_image = predict_image(
                data_to_pass, self.model, self.random_forest, self.param.scalings,
                order=self.spin_interpolation_order.value(),
                use_min_features=self.check_use_min_features.isChecked(),
                device=device, normalize=self.check_normalize.isChecked(),
                image_downsample=self.spin_downsample.value()
            )
            self.viewer.layers['prediction'].data = predicted_image
        
        self.viewer.layers['prediction'].refresh()

    def predict_all(self):
        """Predict the segmentation of all frames based 
        on a RF model trained with annotations"""

        if self.random_forest is None:
            raise Exception('No model found. Please train a model first.')
        
        if self.model is None:
            if self.check_use_default_model.isChecked():
                self.set_default_model()
            else:
                raise Exception('You have to define and load a model first')
            
        device = 'cuda' if self.check_use_cuda.isChecked() else 'cpu'

        self.check_prediction_layer_exists()

        for step in range(self.viewer.dims.nsteps[0]):
            image = self.viewer.layers[self.selected_channel].data[step]
            predicted_image = predict_image(
                image, self.model, self.random_forest, self.param.scalings,
                order=self.spin_interpolation_order.value(),
                use_min_features=self.check_use_min_features.isChecked(),
                device=device, normalize=self.check_normalize.isChecked(),
                image_downsample=self.spin_downsample.value())
            self.viewer.layers['prediction'].data[step] = predicted_image

    def check_prediction_layer_exists(self):

        layer_names = [x.name for x in self.viewer.layers]
        if 'prediction' not in layer_names:

            if self.viewer.layers[self.selected_channel].rgb:
                layer_shape = self.viewer.layers[self.selected_channel].data.shape[0:2]
            elif self.check_dims_is_channels.isChecked():
                layer_shape = self.viewer.layers[self.selected_channel].data.shape[-2::]
            else:
                layer_shape = self.viewer.layers[self.selected_channel].data.shape

            self.viewer.add_labels(
                data=np.zeros((layer_shape), dtype=np.uint8),
                name='prediction'
            )

    def save_model(self, event=None, save_file=None):
        """Select file where to save the classifier model."""

        # save sklearn model
        if save_file is None:
            dialog = QFileDialog()
            save_file, _ = dialog.getSaveFileName(self, "Save model", None, "JOBLIB (*.joblib)")
        save_file = Path(save_file)
        dump(self.random_forest, save_file)
        self.param.random_forest = save_file#.as_posix()
        self.update_params()
        self.param.save_parameters(save_file.parent.joinpath('convpaint_params.yml'))

    def update_params(self):
        """Update parameters from GUI."""

        self.update_scalings()
        self.param.model_name = self.qcombo_model_type.currentText()
        self.param.model_layers = self.get_selected_layers_names()
        self.param.order = self.spin_interpolation_order.value()
        self.param.use_min_features = self.check_use_min_features.isChecked()
        self.param.image_downsample = self.spin_downsample.value()
        self.param.normalize = self.check_normalize.isChecked()
    
    def load_classifier(self, event=None, save_file=None):
        """Select classifier model file to load."""

        if save_file is None:
            dialog = QFileDialog()
            save_file, _ = dialog.getOpenFileName(self, "Choose model", None, "JOBLIB (*.joblib)")
        save_file = Path(save_file)
        self.random_forest, self.param = load_trained_classifier(save_file)

        self.update_gui_from_params()
        self.model = Hookmodel(param=self.param)


    def update_gui_from_params(self):
        """Update GUI from parameters and then update NN model with that info."""

        self.qcombo_model_type.setCurrentText(self.param.model_name)
        # load model to get layer list
        self._on_load_nnmodel()
        #self.update_scalings()
        self.num_scales_combo.setCurrentText(str(self.param.scalings))
        for sel in self.param.model_layers:
            self.model_output_selection.item(list(self.model.module_dict.keys()).index(sel)).setSelected(True)
        self._on_click_define_model_outputs()
        self.spin_interpolation_order.setValue(self.param.order)
        self.check_use_min_features.setChecked(self.param.use_min_features)
        self.spin_downsample.setValue(self.param.image_downsample)
        self.check_normalize.setChecked(self.param.normalize)
