from qtpy.QtWidgets import (QWidget, QPushButton,QVBoxLayout,
                            QLabel, QComboBox,QFileDialog, QListWidget,
                            QCheckBox, QAbstractItemView, QGridLayout, QSpinBox, QButtonGroup,
                            QRadioButton)
from qtpy.QtCore import Qt
from magicgui.widgets import create_widget
import napari
from napari.utils import progress
from joblib import dump
from pathlib import Path
import numpy as np
import pickle
import warnings


from napari_guitils.gui_structures import VHGroup, TabSet
from .conv_paint_utils import (parallel_predict_image, 
                               normalize_image, compute_image_stats)
from .conv_parameters import Param
from .conv_paint_utils import (get_features_current_layers,
                               train_classifier)
from .conv_paint_classifier import load_trained_classifier

from napari_convpaint.conv_paint_nnlayers import AVAILABLE_MODELS as NN_MODELS
from napari_convpaint.conv_paint_gaussian import AVAILABLE_MODELS as GAUSSIAN_MODELS
from napari_convpaint.conv_paint_dino import AVAILABLE_MODELS as DINO_MODELS
from napari_convpaint.conv_paint_gaussian import GaussianFeatures
from napari_convpaint.conv_paint_dino import DinoFeatures
from .conv_paint_nnlayers import Hookmodel

# Initialize the ALL_MODELS dictionary with the models that are always available
ALL_MODELS = {x: Hookmodel for x in NN_MODELS}
ALL_MODELS.update({x: GaussianFeatures for x in GAUSSIAN_MODELS})
ALL_MODELS.update({x: DinoFeatures for x in DINO_MODELS})

# Try to import CellposeFeatures and update the ALL_MODELS dictionary if successful
# Cellpose is only installed with pip install napari-convpaint[cellpose]
try:
    from napari_convpaint.conv_paint_cellpose import AVAILABLE_MODELS as CELLPOSE_MODELS
    from napari_convpaint.conv_paint_cellpose import CellposeFeatures
    ALL_MODELS.update({x: CellposeFeatures for x in CELLPOSE_MODELS})
except ImportError:
    # Handle the case where CellposeFeatures or its dependencies are not available
    print("Cellpose is not installed and is not available as feature extractor.\n"
          "Run 'pip install napari-convpaint[cellpose]' to install it.")

class ConvPaintWidget(QWidget):
    """
    Implementation of a napari widget for interactive segmentation performed
    via a random forest model trained on annotations. The filters used to 
    generate the model features are taken from the first layer of a VGG16 model
    as proposed here: https://github.com/hinderling/napari_pixel_classifier

    Parameters
    ----------
    napari_viewer: napari.Viewer
        main napari viewer
    project: bool
        use project widget for multi-image project management
    third_party: bool
        if True, widget is used as third party and will not add layers to the viewer
        by default.
    """
    
    def __init__(self, napari_viewer, parent=None, project=False, third_party=False):
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        
        self.param = Param()
        self.model = None
        self.random_forest = None
        self.project_widget = None
        self.features_per_layer = None
        self.selected_channel = None
        self.image_mean = None
        self.image_std = None
        self.third_party = third_party

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.tab_names = ['Home', 'Files', 'Model']
        self.tabs = TabSet(self.tab_names, tab_layouts=[None, None, QGridLayout()])
        self.tabs.setTabEnabled(self.tabs.tab_names.index('Files'), False)
        self.main_layout.addWidget(self.tabs)

        self.tabs.widget(0).layout().setAlignment(Qt.AlignTop)

        self.layer_selection_group = VHGroup('Layer selection', orientation='G')
        self.data_dims_group = VHGroup('Data dimensions', orientation='G')
        self.train_group = VHGroup('Train', orientation='G')
        self.predict_group = VHGroup('Segment', orientation='G')
        self.load_save_group = VHGroup('Load/Save', orientation='G')
        self.options_group = VHGroup('Options', orientation='G')
        self.tabs.add_named_tab('Home', self.layer_selection_group.gbox)
        self.tabs.add_named_tab('Home', self.train_group.gbox)
        self.tabs.add_named_tab('Home', self.predict_group.gbox)
        self.tabs.add_named_tab('Home', self.data_dims_group.gbox)
        self.tabs.add_named_tab('Home', self.load_save_group.gbox)
        self.tabs.add_named_tab('Home', self.options_group.gbox)

        # data layer
        self.select_layer_widget = create_widget(annotation=napari.layers.Image, label='Pick image')
        self.select_layer_widget.reset_choices()
        self.viewer.layers.events.inserted.connect(self.select_layer_widget.reset_choices)
        self.viewer.layers.events.removed.connect(self.select_layer_widget.reset_choices)
        # annotation layer
        self.select_annotation_layer_widget = create_widget(annotation=napari.layers.Labels, label='Pick annotation')
        self.select_annotation_layer_widget.reset_choices()
        self.viewer.layers.events.inserted.connect(self.select_annotation_layer_widget.reset_choices)
        self.viewer.layers.events.removed.connect(self.select_annotation_layer_widget.reset_choices)

        self.layer_selection_group.glayout.addWidget(QLabel('Layer to segment'), 0,0,1,1)
        self.layer_selection_group.glayout.addWidget(self.select_layer_widget.native, 0,1,1,1)
        self.layer_selection_group.glayout.addWidget(QLabel('Layer for annotation'), 1,0,1,1)
        self.layer_selection_group.glayout.addWidget(self.select_annotation_layer_widget.native, 1,1,1,1)

        self.add_layers_btn = QPushButton('Add annotations/segmentation layers')
        self.add_layers_btn.setEnabled(False)
        self.layer_selection_group.glayout.addWidget(self.add_layers_btn, 2,0,1,2)

        self.button_group_channels = QButtonGroup()
        self.radio_single_channel = QRadioButton('Single channel image/stack')
        self.radio_single_channel.setToolTip('2D images or 3D images where additional dimension is NOT channels')
        self.radio_multi_channel = QRadioButton('Multichannel image')
        self.radio_multi_channel.setToolTip('Images with an additional channel dimension')
        self.radio_rgb = QRadioButton('RGB image')
        self.radio_rgb.setToolTip('Use this option with images displayed as RGB')
        self.radio_single_channel.setChecked(True)
        [x.setEnabled(False) for x in [self.radio_multi_channel, self.radio_rgb, self.radio_single_channel]]
        self.button_group_channels.addButton(self.radio_single_channel, id=1)
        self.button_group_channels.addButton(self.radio_multi_channel, id=2)
        self.button_group_channels.addButton(self.radio_rgb, id=3)
        self.data_dims_group.glayout.addWidget(self.radio_single_channel, 0,0,1,1)
        self.data_dims_group.glayout.addWidget(self.radio_multi_channel, 1,0,1,1)
        self.data_dims_group.glayout.addWidget(self.radio_rgb, 2,0,1,1)

        self.update_model_btn = QPushButton('Train')
        self.update_model_btn.setToolTip('Train model on annotations')
        self.train_group.glayout.addWidget(self.update_model_btn, 0,0,1,1)
        self.check_use_project = QCheckBox('Use multiple files')
        self.check_use_project.setToolTip('Activate Files Tab to use multiple files to train the model')
        self.check_use_project.setChecked(False)
        self.train_group.glayout.addWidget(self.check_use_project, 1,0,1,1)

        self.update_model_on_project_btn = QPushButton('Train on multiple images')
        self.update_model_on_project_btn.setToolTip('Train on all images in project select in Files Tab')
        self.train_group.glayout.addWidget(self.update_model_on_project_btn, 1,1,1,1)
        if project is False:
            self.update_model_on_project_btn.setEnabled(False)

        self.prediction_btn = QPushButton('Segment image')
        self.prediction_btn.setEnabled(False)
        self.prediction_btn.setToolTip('Segment 2D image or current slice/frame of 3D image/movie ')
        self.predict_group.glayout.addWidget(self.prediction_btn, 0,0,1,1)
        self.prediction_all_btn = QPushButton('Segment stack')
        self.prediction_all_btn.setToolTip('Segment all slices/frames of 3D image/movie')
        self.prediction_all_btn.setEnabled(False)
        self.predict_group.glayout.addWidget(self.prediction_all_btn, 0,1,1,1)

        self.save_model_btn = QPushButton('Save trained model')
        self.save_model_btn.setToolTip('Save model as *.pickle file')
        self.save_model_btn.setEnabled(False)
        self.load_save_group.glayout.addWidget(self.save_model_btn, 0,0,1,1)

        self.load_model_btn = QPushButton('Load trained model')
        self.load_model_btn.setToolTip('Select *.pickle file to load as trained model')
        self.load_save_group.glayout.addWidget(self.load_model_btn, 0,1,1,1)

        self.reset_model_btn = QPushButton('Reset model')
        self.reset_model_btn.setToolTip('Discard current model and annotations, create new default model.')
        self.load_save_group.glayout.addWidget(self.reset_model_btn, 1,0,1,1)

        self.load_save_group.glayout.addWidget(QLabel('Current model:'), 2,0,1,1)
        self.current_model_path = QLabel('None')
        self.load_save_group.glayout.addWidget(self.current_model_path, 2,1,1,1)

        self.check_use_custom_model = QCheckBox('Use custom model')
        self.check_use_custom_model.setToolTip('Activate Model Tab to customize model')
        self.check_use_custom_model.setChecked(False)
        self.options_group.glayout.addWidget(self.check_use_custom_model, 0,0,1,1)

        self.spin_downsample = QSpinBox()
        self.spin_downsample.setMinimum(1)
        self.spin_downsample.setMaximum(10)
        self.spin_downsample.setValue(1)
        self.spin_downsample.setToolTip('Reduce image size for faster computing.')
        self.options_group.glayout.addWidget(QLabel('Downsample'), 1,0,1,1)
        self.options_group.glayout.addWidget(self.spin_downsample, 1,1,1,1)



        self.check_tile_annotations = QCheckBox('Tile annotations for training')
        self.check_tile_annotations.setChecked(False)
        self.check_tile_annotations.setToolTip('Crop around annotated regions to speed up training.\nDisable for models that extract long range features (e.g. DINO)!')
        self.options_group.glayout.addWidget(self.check_tile_annotations, 2,0,1,1)


        self.check_tile_image = QCheckBox('Tile image for segmentation')
        self.check_tile_image.setChecked(False)
        self.check_tile_image.setToolTip('Tile image to reduce memory usage.\nTake care when using models that extract long range features (e.g. DINO).')
        self.options_group.glayout.addWidget(self.check_tile_image, 3,0,1,1)

        self.button_group_normalize = QButtonGroup()
        self.radio_no_normalize = QRadioButton('No normalization')
        self.radio_no_normalize.setToolTip('No normalization is applied')
        self.radio_normalized_over_stack = QRadioButton('Normalize over stack')
        self.radio_normalized_over_stack.setToolTip('Normalize over complete z or t stack')
        self.radio_normalize_by_image = QRadioButton('Normalized by plane')
        self.radio_normalize_by_image.setToolTip('Normalize each plane individually')
        self.radio_normalized_over_stack.setChecked(True)
        #[x.setEnabled(False) for x in [self.radio_no_normalize, self.radio_normalized_over_stack, self.radio_normalize_by_image]]
        self.button_group_normalize.addButton(self.radio_no_normalize, id=1)
        self.button_group_normalize.addButton(self.radio_normalized_over_stack, id=2)
        self.button_group_normalize.addButton(self.radio_normalize_by_image, id=3)
        self.options_group.glayout.addWidget(self.radio_no_normalize, 4,0,1,1)
        self.options_group.glayout.addWidget(self.radio_normalized_over_stack, 5,0,1,1)
        self.options_group.glayout.addWidget(self.radio_normalize_by_image, 6,0,1,1)


        self.qcombo_model_type = QComboBox()
        self.qcombo_model_type.addItems(ALL_MODELS.keys())
        self.qcombo_model_type.setToolTip('Select model architecture')
        self.tabs.add_named_tab('Model', self.qcombo_model_type, [0,0,1,2])

        self.create_model_btn = QPushButton('Create model')
        self.create_model_btn.setToolTip('Create a new feature extraction model')
        self.tabs.add_named_tab('Model', self.create_model_btn, [1,0,1,2])

        self.set_nnmodel_outputs_btn = QPushButton('Set model outputs')
        self.set_nnmodel_outputs_btn.setToolTip('Select layers to use as feature extractors')
        self.tabs.add_named_tab('Model', self.set_nnmodel_outputs_btn, [2,0,1,2])

        self.model_output_selection = QListWidget()
        self.model_output_selection.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tabs.add_named_tab('Model', self.model_output_selection, [3,0,1,2])

        self.num_scales_combo = QComboBox()
        self.num_scales_combo.addItem('[1]',[1])
        self.num_scales_combo.addItem('[1,2]',[1,2])
        self.num_scales_combo.addItem('[1,2,4]',[1,2,4])
        self.num_scales_combo.addItem('[1,2,4,8]',[1,2,4,8])

        self.num_scales_combo.setCurrentText('[1,2]')
        self.tabs.add_named_tab('Model', QLabel('Number of scales'), [4,0,1,1])
        self.tabs.add_named_tab('Model', self.num_scales_combo, [4,1,1,1])

        self.check_use_min_features = QCheckBox('Use min features')
        self.check_use_min_features.setChecked(False)
        self.check_use_min_features.setToolTip('Use same number of features from each layer. Otherwise use all features from each layer.')
        self.tabs.add_named_tab('Model', self.check_use_min_features, [5,0,1,2])

        self.spin_interpolation_order = QSpinBox()
        self.spin_interpolation_order.setMinimum(0)
        self.spin_interpolation_order.setMaximum(5)
        self.spin_interpolation_order.setValue(1)
        self.spin_interpolation_order.setToolTip('Interpolation order for image rescaling')
        self.tabs.add_named_tab('Model', QLabel('Interpolation order'), [6,0,1,1])
        self.tabs.add_named_tab('Model', self.spin_interpolation_order, [6,1,1,1])
        self.tabs.setTabEnabled(self.tabs.tab_names.index('Model'), False)

        self.check_use_cuda = QCheckBox('Use cuda')
        self.check_use_cuda.setChecked(False)
        self.check_use_cuda.setToolTip('Use GPU for training and segmentation')
        self.tabs.add_named_tab('Model', self.check_use_cuda, grid_pos=[7,0,1,1])

        if project is True:
            self._add_project()

        self.add_connections()
        self.select_layer()

        self.viewer.bind_key('a', self.hide_annotation, overwrite=True)
        self.viewer.bind_key('r', self.hide_prediction, overwrite=True)

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

        if not self.check_use_custom_model.isChecked():
            self.tabs.setTabEnabled(self.tabs.tab_names.index('Model'), False)
            self.set_default_model()
        else:
            self.tabs.setTabEnabled(self.tabs.tab_names.index('Model'), True)
            self.qcombo_model_type.setCurrentText('single_layer_vgg16')

    def add_connections(self):
        
        self.select_layer_widget.changed.connect(self.select_layer)
        self.num_scales_combo.currentIndexChanged.connect(self.update_scalings_from_gui)

        self.add_layers_btn.clicked.connect(self.add_annotation_layer)
        self.update_model_btn.clicked.connect(self.update_classifier)
        self.update_model_on_project_btn.clicked.connect(self.update_classifier_on_project)
        self.prediction_btn.clicked.connect(self.predict)
        self.prediction_all_btn.clicked.connect(self.predict_all)
        self.save_model_btn.clicked.connect(self.save_model)
        self.load_model_btn.clicked.connect(self.load_model)
        self.reset_model_btn.clicked.connect(self.reset_model)
        self.check_use_project.stateChanged.connect(self._add_project)
        self.check_use_custom_model.stateChanged.connect(self._set_custom_model)

        self.create_model_btn.clicked.connect(self._on_create_model)
        self.set_nnmodel_outputs_btn.clicked.connect(self._on_click_define_model_outputs)

        self.radio_multi_channel.toggled.connect(self.reset_radio_norm_settings)
        self.radio_single_channel.toggled.connect(self.reset_radio_norm_settings)
        self.radio_rgb.toggled.connect(self.reset_radio_norm_settings)

        self.radio_no_normalize.toggled.connect(self.reset_stats)
        self.radio_normalized_over_stack.toggled.connect(self.reset_stats)
        self.radio_normalize_by_image.toggled.connect(self.reset_stats)

        self.qcombo_model_type.currentIndexChanged.connect(self.on_model_selected)



    def hide_annotation(self, event=None):
        """Hide annotation layer."""

        if self.viewer.layers['annotations'].visible == False:
            self.viewer.layers['annotations'].visible = True
        else:
            self.viewer.layers['annotations'].visible = False

    def hide_prediction(self, event=None):
        """Hide prediction layer."""

        if self.viewer.layers['segmentation'].visible == False:
            self.viewer.layers['segmentation'].visible = True
        else:
            self.viewer.layers['segmentation'].visible = False

    def select_layer(self, newtext=None):
        
        self.selected_channel = self.select_layer_widget.native.currentText()
        if self.select_layer_widget.value is None:
            self.add_layers_btn.setEnabled(False)
        else:
            self.add_layers_btn.setEnabled(True)
        
        # set radio buttons depending on selected image type
        if self.select_layer_widget.value is not None:
            if self.select_layer_widget.value.rgb:
                self.radio_rgb.setChecked(True)
                self.radio_multi_channel.setEnabled(False)
                self.radio_single_channel.setEnabled(False)
            elif self.select_layer_widget.value.ndim == 2:
                self.radio_single_channel.setChecked(True)
                self.radio_multi_channel.setEnabled(False)
                self.radio_rgb.setEnabled(False)
            elif self.select_layer_widget.value.ndim == 3:
                self.radio_rgb.setEnabled(False)
                self.radio_multi_channel.setEnabled(True)
                self.radio_single_channel.setEnabled(True)
                self.radio_single_channel.setChecked(True)
            elif self.select_layer_widget.value.ndim == 4:
                self.radio_rgb.setEnabled(False)
                self.radio_multi_channel.setEnabled(True)
                self.radio_single_channel.setEnabled(False)
                self.radio_multi_channel.setChecked(True)

            self.reset_radio_norm_settings()
            self.reset_predict_buttons_after_training()

    def reset_stats(self):
        self.image_mean, self.image_std = None, None

    def reset_radio_norm_settings(self, event=None):
        
        self.image_mean, self.image_std = None, None
        self.add_annotation_layer(force_add=False)

        if self.select_layer_widget.value.ndim == 2:
            self.radio_normalize_by_image.setEnabled(True)
            self.radio_normalize_by_image.setChecked(True)
            self.radio_normalized_over_stack.setEnabled(False)
        elif (self.select_layer_widget.value.ndim == 3) and (self.radio_multi_channel.isChecked()):
            self.radio_normalize_by_image.setEnabled(True)
            self.radio_normalize_by_image.setChecked(True)
            self.radio_normalized_over_stack.setEnabled(False)
        else:
            self.radio_normalize_by_image.setEnabled(True)
            self.radio_normalized_over_stack.setEnabled(True)
            self.radio_normalized_over_stack.setChecked(True)

            
    def get_image_stats(self):
        
        # put channels in format (C)(T,Z)YX
        data = self.get_data_channel_first()

        if self.radio_normalized_over_stack.isChecked():
            if self.radio_multi_channel.isChecked() | self.radio_rgb.isChecked():
                self.image_mean, self.image_std = compute_image_stats(
                    image=data,
                    ignore_n_first_dims=1)
            else:
                self.image_mean, self.image_std = compute_image_stats(
                    image=data,
                    ignore_n_first_dims=None)
        elif self.radio_normalize_by_image.isChecked():
            # also takes into account CXY case (2D multi-channel image) as first dim is dropped
            self.image_mean, self.image_std = compute_image_stats(
                image=data,
                ignore_n_first_dims=data.ndim-2)

    def reset_model(self, event=None):

        self.model = None
        self.random_forest = None
        self.prediction_btn.setEnabled(False)
        self.prediction_all_btn.setEnabled(False)
        self.save_model_btn.setEnabled(False)
        self.current_model_path.setText('None')


        if self.select_layer_widget.value is None:
            [x.setEnabled(False) for x in [self.radio_multi_channel, self.radio_rgb, self.radio_single_channel]]
        else:
            self.select_layer()
        
    def add_annotation_layer(self, event=None, force_add=True):
        """Add annotation and prediction layers to viewer. If the layer already exists,
        remove it and add a new one. If the widget is used as third party (self.third_party=True),
        no layer is added if it didn't exist before, unless force_add=True (e.g. when the user click
        on the add layer button)"""

        self.event = event
        if self.select_layer_widget.value is None:
            raise Exception('Please select an image layer first')
        
        if self.viewer.layers[self.selected_channel].rgb:
            layer_shape = self.viewer.layers[self.selected_channel].data.shape[0:-1]
        elif self.radio_multi_channel.isChecked():
            layer_shape = self.viewer.layers[self.selected_channel].data.shape[1::]
        else:
            layer_shape = self.viewer.layers[self.selected_channel].data.shape

        annotation_exists = False
        if 'annotations' in self.viewer.layers:
            self.viewer.layers.remove('annotations')
            annotation_exists = True

        if (not self.third_party) | (force_add) | (self.third_party & annotation_exists):
            self.viewer.add_labels(
                data=np.zeros((layer_shape), dtype=np.uint8),
                name='annotations'
                )
        
        segmentation_exists = False
        if 'segmentation' in self.viewer.layers:
            self.viewer.layers.remove('segmentation')
            segmentation_exists = True

        if (not self.third_party) | (force_add) | (self.third_party & segmentation_exists):
            self.viewer.add_labels(
                data=np.zeros((layer_shape), dtype=np.uint8),
                name='segmentation'
                )
        
        if 'annotations' in self.viewer.layers:
            self.viewer.layers.selection.active = self.viewer.layers['annotations']
            self.select_annotation_layer_widget.value = self.viewer.layers['annotations']

    def update_scalings_from_gui(self):
        self.param.scalings = self.num_scales_combo.currentData()

    def update_scalings_from_param(self):
        index = self.num_scales_combo.findData(self.param.scalings)
        if index != -1:
            self.num_scales_combo.setCurrentIndex(index)
        else:
            self.num_scales_combo.addItem(str(self.param.scalings), self.param.scalings)
            self.num_scales_combo.setCurrentIndex(self.num_scales_combo.count()-1)

    def _create_output_selection(self):
        """Update list of selectable layers"""

        self.model_output_selection.clear()
        self.model_output_selection.addItems(self.model.selectable_layer_keys.keys())

    def _on_click_define_model_outputs(self, event=None):
        """Using hooks setup model to give outputs at selected layers."""
        
        model_type = self.qcombo_model_type.currentText()
        model_class = ALL_MODELS[model_type]
        self.model = model_class(model_name=model_type, use_cuda=self.check_use_cuda.isChecked())

        selected_layers = self.get_selected_layers_names()
        self.model.register_hooks(selected_layers=selected_layers)

    def get_selected_layers_names(self):
        """Get names of selected layers."""

        selected_rows = self.model_output_selection.selectedItems()
        selected_layers = [x.text() for x in selected_rows]
        return selected_layers
                                  

    def _on_create_model(self, event=None):
        """Create a neural network model that will be used for feature extraction."""
        self.update_params_from_gui()
        self.model = self.create_model(self.param)
        self.update_gui_from_model()
        self.current_model_path.setText('Unsaved')


    def create_model(self, param):
        """Create a model based on the given parameters."""
        model_class = ALL_MODELS[param.model_name]
        model = model_class(
            model_name=param.model_name,
            use_cuda=param.use_cuda
        )
        
        if isinstance(model, Hookmodel):
            if param.model_layers:
                model.register_hooks(selected_layers=param.model_layers)
            elif len(model.named_modules) == 1:
                model.register_hooks(selected_layers=[list(model.module_dict.keys())[0]])
        
        return model
    
    def update_gui_from_model(self):
        """Update GUI based on the current model."""
        if self.model is None:
            self.set_nnmodel_outputs_btn.setEnabled(False)
            self.model_output_selection.setEnabled(False)
            self.model_output_selection.clear()
        elif isinstance(self.model, Hookmodel):
            self._create_output_selection()
            if len(self.model.named_modules) == 1:
                self.model_output_selection.setCurrentRow(0)
                self.set_nnmodel_outputs_btn.setEnabled(False)
                self.model_output_selection.setEnabled(False)
            else:
                self.set_nnmodel_outputs_btn.setEnabled(True)
                self.model_output_selection.setEnabled(True)
                for layer in self.param.model_layers:
                    items = self.model_output_selection.findItems(layer, Qt.MatchExactly)
                    for item in items:
                        item.setSelected(True)
        else:
            self.set_nnmodel_outputs_btn.setEnabled(False)
            self.model_output_selection.setEnabled(False)
            self.model_output_selection.clear()


    def update_params_from_gui(self):
        """Update parameters from GUI."""
        self.update_scalings_from_gui()
        self.param.model_name = self.qcombo_model_type.currentText()
        self.param.model_layers = self.get_selected_layers_names()
        self.param.order = self.spin_interpolation_order.value()
        self.param.use_min_features = self.check_use_min_features.isChecked()
        self.param.image_downsample = self.spin_downsample.value()
        self.param.normalize = self.button_group_normalize.checkedId()
        self.param.use_cuda = self.check_use_cuda.isChecked()

    def update_gui_from_params(self):
        """Update GUI from parameters."""
        self.update_scalings_from_param()
        self.qcombo_model_type.setCurrentText(self.param.model_name)
        self.spin_interpolation_order.setValue(self.param.order)
        self.check_use_min_features.setChecked(self.param.use_min_features)
        self.spin_downsample.setValue(self.param.image_downsample)
        self.button_group_normalize.button(self.param.normalize).setChecked(True)
        self.check_use_cuda.setChecked(self.param.use_cuda)


    def set_default_model(self):#, keep_rgb=False):
        """Set default model."""
        '''if keep_rgb:
            self.qcombo_model_type.setCurrentText('single_layer_vgg16_rgb')
        else:
            self.qcombo_model_type.setCurrentText('single_layer_vgg16')'''
        self.qcombo_model_type.setCurrentText('single_layer_vgg16')
        self.num_scales_combo.setCurrentText('[1,2]')
        self.spin_interpolation_order.setValue(1)
        self.check_use_min_features.setChecked(True)
        self._on_create_model()

    def get_data_channel_first(self):
        """Get data from selected channel. If RGB, move channel axis to first position."""
            
        image_stack = self.viewer.layers[self.selected_channel].data
        if self.viewer.layers[self.selected_channel].rgb:
            image_stack = np.moveaxis(image_stack, -1, 0)
        return image_stack

    def get_selectedlayer_data(self):
        """Get data from selected channel. Output has channel (if present) in 
        first position and is normalized."""

        image_stack = self.get_data_channel_first()
        if self.image_mean is None:
            self.get_image_stats()

        if not self.radio_no_normalize.isChecked():
            image_stack = normalize_image(
                image=image_stack,
                image_mean=self.image_mean,
                image_std=self.image_std)
            
        return image_stack

    def update_classifier(self):
        """Given a set of new annotations, update the random forest model."""

        unique_labels = np.unique(self.select_annotation_layer_widget.value.data)
        if (not 1 in unique_labels) | (not 2 in unique_labels):
            raise Exception('You need annotations for at least foreground and background')
        
        if self.model is None:
            if not self.check_use_custom_model.isChecked():
                self.set_default_model()
            else:
                raise Exception('You have to define and load a model first')

        image_stack = self.get_selectedlayer_data()
        
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(True)

        with progress(total=0) as pbr:
            self.current_model_path.setText('In training')
            pbr.set_description(f"Training")
            features, targets = get_features_current_layers(
                image=image_stack,
                annotations=self.select_annotation_layer_widget.value.data,
                model=self.model,
                scalings=self.param.scalings,
                order=self.spin_interpolation_order.value(),
                use_min_features=self.check_use_min_features.isChecked(),
                image_downsample=self.spin_downsample.value(),
                tile_annotations=self.check_tile_annotations.isChecked(),
            )
            self.random_forest = train_classifier(features, targets)
            self.reset_predict_buttons_after_training()
            self.save_model_btn.setEnabled(True)
            self.current_model_path.setText('Unsaved')
            
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(False)

    def reset_predict_buttons_after_training(self):
        
        if (self.model is not None) and (self.select_layer_widget.value is not None):
            self.prediction_btn.setEnabled(True)
            if self.select_layer_widget.value == 2:
                self.prediction_all_btn.setEnabled(False)
            elif self.select_layer_widget.value.ndim == 3:
                if self.radio_multi_channel.isChecked():
                    self.prediction_all_btn.setEnabled(False)
                else:
                    self.prediction_all_btn.setEnabled(True)
            else:
                self.prediction_all_btn.setEnabled(True)


    def update_classifier_on_project(self):
        """Train classifier on all annotations in project.
        !!!! Need to double-check if normalization is done correctly for projects !!!!"""

        if self.model is None:
            if not self.check_use_custom_model.isChecked():
                self.set_default_model()
            else:
                raise Exception('You have to define and load a model first')

        num_files = len(self.project_widget.params.file_paths)
        if num_files == 0:
            raise Exception('No files found')

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(True)
        
        self.viewer.layers.events.removed.disconnect(self.reset_model)
        with progress(total=0) as pbr:
            pbr.set_description(f"Training")
            self.current_model_path.setText('In training')
            all_features, all_targets = [], []
            for ind in range(num_files):
                self.project_widget.file_list.setCurrentRow(ind)

                image_stack = self.get_selectedlayer_data()

                features, targets = get_features_current_layers(
                    model=self.model,
                    image=image_stack,
                    annotations=self.select_annotation_layer_widget.value.data,
                    scalings=self.param.scalings,
                    order=self.spin_interpolation_order.value(),
                    use_min_features=self.check_use_min_features.isChecked(),
                    image_downsample=self.spin_downsample.value(),
                    tile_annotations=self.check_tile_annotations.isChecked(),
                )
                if features is None:
                    continue
                all_features.append(features)
                all_targets.append(targets)
            
            all_features = np.concatenate(all_features, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)

            self.random_forest = train_classifier(all_features, all_targets)
            self.reset_predict_buttons_after_training()
            self.save_model_btn.setEnabled(True)
            self.current_model_path.setText('Unsaved')

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(False)

        self.viewer.layers.events.removed.connect(self.reset_model)
        

    def predict(self):
        """Predict the segmentation of the currently viewed frame based 
        on a RF model trained with annotations"""

        if self.model is None:
            if not self.check_use_custom_model.isChecked():
                self.set_default_model()
            else:
                raise Exception('You have to define and load a model first')
        
        if self.random_forest is None:
            self.update_classifier()

        self.check_prediction_layer_exists()

        if self.image_mean is None:
            self.get_image_stats()
        
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(True)

        with progress(total=0) as pbr:
            
            pbr.set_description(f"Prediction")

            if self.viewer.dims.ndim == 2:
                image = self.get_selectedlayer_data()


            elif self.viewer.dims.ndim == 3:
                if self.radio_multi_channel.isChecked():
                    image = self.get_selectedlayer_data()
                elif self.radio_single_channel.isChecked():
                    step = self.viewer.dims.current_step[0]
                    image = self.viewer.layers[self.selected_channel].data[step]
                    if not self.radio_no_normalize.isChecked():
                        if self.radio_normalized_over_stack.isChecked():
                            image_mean = self.image_mean
                            image_std = self.image_std
                        elif self.radio_normalize_by_image.isChecked():
                            image_mean = self.image_mean[step]
                            image_std = self.image_std[step]
                        image = normalize_image(image=image, image_mean=image_mean, image_std=image_std)  
                elif self.radio_rgb.isChecked():
                    step = self.viewer.dims.current_step[0]
                    image = self.viewer.layers[self.selected_channel].data[step]
                    image = np.moveaxis(image, -1, 0)
                    
                    if not self.radio_no_normalize.isChecked():
                        if self.radio_normalized_over_stack.isChecked():
                            image_mean = self.image_mean[:,0,::]
                            image_std = self.image_std[:,0,::]
                        else:
                            image_mean = self.image_mean[:,step]
                            image_std = self.image_std[:,step]
                        image = normalize_image(image=image, image_mean=image_mean, image_std=image_std)
                        
            elif self.viewer.dims.ndim == 4:
                # for 4D channel is always first, t/z second
                step = self.viewer.dims.current_step[1]
                image = self.viewer.layers[self.selected_channel].data[:, step]
                if not self.radio_no_normalize.isChecked():
                    if self.radio_normalized_over_stack.isChecked():
                        image_mean = self.image_mean[:,0,::]
                        image_std = self.image_std[:,0,::]
                    else:
                        image_mean = self.image_mean[:,step]
                        image_std = self.image_std[:,step]
                    image = normalize_image(image=image, image_mean=image_mean, image_std=image_std)

            if self.check_tile_image.isChecked():
                predicted_image = parallel_predict_image(
                    image=image, model=self.model, classifier=self.random_forest,
                    scalings=self.param.scalings,
                    order=self.spin_interpolation_order.value(),
                    use_min_features=self.check_use_min_features.isChecked(),
                    image_downsample=self.spin_downsample.value(),
                    use_dask=False)
            else:
                predicted_image = self.model.predict_image(
                    image=image, classifier=self.random_forest, scalings=self.param.scalings,
                    order=self.spin_interpolation_order.value(),
                    use_min_features=self.check_use_min_features.isChecked(),
                    image_downsample=self.spin_downsample.value(),
                )
            if self.viewer.dims.ndim == 2:
                self.viewer.layers['segmentation'].data = predicted_image
            elif (self.viewer.dims.ndim == 3) and (self.radio_multi_channel.isChecked()):
                self.viewer.layers['segmentation'].data = predicted_image
            else:
                self.viewer.layers['segmentation'].data[step] = predicted_image
            self.viewer.layers['segmentation'].refresh()

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(False)
    

    def predict_all(self):
        """Predict the segmentation of all frames based 
        on a RF model trained with annotations"""

        if self.random_forest is None:
            raise Exception('No model found. Please train a model first.')
        
        if self.model is None:
            if not self.check_use_custom_model.isChecked():
                self.set_default_model()
            else:
                raise Exception('You have to define and load a model first')
            
        self.check_prediction_layer_exists()

        if self.image_mean is None:
            self.get_image_stats()

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(True)

        image_stack = self.viewer.layers[self.selected_channel].data
        if self.radio_rgb.isChecked():
            image_stack = np.moveaxis(image_stack, -1, 0)
        if not self.radio_no_normalize.isChecked():
            image_stack = normalize_image(image=image_stack,
                                          image_mean=self.image_mean,
                                          image_std=self.image_std)
        if image_stack.ndim == 3:
            num_steps = image_stack.shape[0]
        elif image_stack.ndim == 4:
            num_steps = image_stack.shape[1]           
        for step in progress(range(num_steps)):
            if image_stack.ndim == 3:
                image = image_stack[step]
            elif image_stack.ndim == 4:
                image = image_stack[:, step]
            else:
                raise Exception(f'Image stack has wrong dimensionality {image_stack.ndim}')
            
            if self.check_tile_image.isChecked():
                predicted_image = parallel_predict_image(
                image, self.model, self.random_forest, self.param.scalings,
                order=self.spin_interpolation_order.value(),
                use_min_features=self.check_use_min_features.isChecked(),
                image_downsample=self.spin_downsample.value(),
                use_dask=False
            )
            else:
                predicted_image = self.model.predict_image(
                image=image, classifier=self.random_forest, scalings=self.param.scalings,
                order=self.spin_interpolation_order.value(),
                use_min_features=self.check_use_min_features.isChecked(),
                image_downsample=self.spin_downsample.value()
            )
            self.viewer.layers['segmentation'].data[step] = predicted_image
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(False)

    def check_prediction_layer_exists(self):

        layer_names = [x.name for x in self.viewer.layers]
        if 'segmentation' not in layer_names:

            if self.viewer.layers[self.selected_channel].rgb:
                layer_shape = self.viewer.layers[self.selected_channel].data.shape[0:2]
            elif self.radio_multi_channel.isChecked():
                layer_shape = self.viewer.layers[self.selected_channel].data.shape[-2::]
            else:
                layer_shape = self.viewer.layers[self.selected_channel].data.shape

            self.viewer.add_labels(
                data=np.zeros((layer_shape), dtype=np.uint8),
                name='segmentation'
            )

    def save_model(self, event=None, save_file=None):
            """Select file where to save the classifier model along with the model parameters."""
            if self.random_forest is None:
                raise Exception('No model found. Please train a model first.')
            
            if save_file is None:
                dialog = QFileDialog()
                save_file, _ = dialog.getSaveFileName(self, "Save model", None, "PICKLE (*.pickle)")
            save_file = Path(save_file)
            
            # Update parameters from GUI before saving
            self.update_params_from_gui()
            
            # Save random forest, parameters, and model state
            save_data = {
                'random_forest': self.random_forest,
                'params': self.param,
                'model_state': self.model.state_dict() if hasattr(self.model, 'state_dict') else None
            }
            
            with open(save_file, 'wb') as f:
                pickle.dump(save_data, f)
            
            self.current_model_path.setText(save_file.name)

    def load_model(self, event=None, save_file=None):
        """Select classifier model file to load along with the model parameters."""
        if save_file is None:
            dialog = QFileDialog()
            save_file, _ = dialog.getOpenFileName(self, "Choose model", None, "PICKLE (*.pickle)")
        save_file = Path(save_file)

        #tick the custom model checkbox
        self.check_use_custom_model.setChecked(True)
        
        # Load random forest, parameters, and model state
        with open(save_file, 'rb') as f:
            data = pickle.load(f)
        
        self.random_forest = data['random_forest']
        self.param = data['params']
        
        # Create model based on loaded parameters
        self.model = self.create_model(self.param)
        
        # Load model state if available
        if data['model_state'] is not None and hasattr(self.model, 'load_state_dict'):
            self.model.load_state_dict(data['model_state'])
        
        self.update_gui_from_params()
        self.update_gui_from_model()


        self.current_model_path.setText(save_file.name)
        self.reset_predict_buttons_after_training()


    def update_params_from_gui(self):
        """Update parameters from GUI."""

        self.update_scalings_from_gui()
        self.param.model_name = self.qcombo_model_type.currentText()
        self.param.model_layers = self.get_selected_layers_names()
        self.param.order = self.spin_interpolation_order.value()
        self.param.use_min_features = self.check_use_min_features.isChecked()
        self.param.image_downsample = self.spin_downsample.value()
        self.param.normalize = self.button_group_normalize.checkedId()


    def update_gui_from_params(self):
        """Update GUI from parameters and then update the model with that info."""

        self.qcombo_model_type.setCurrentText(self.param.model_name)
        self.update_scalings_from_param()

        self.spin_interpolation_order.setValue(self.param.order)
        self.check_use_min_features.setChecked(self.param.use_min_features)
        self.spin_downsample.setValue(self.param.image_downsample)
        self.button_group_normalize.button(self.param.normalize).setChecked(True)


    def on_model_selected(self, index):
        """Update GUI to show selectable layers of model chosen from drop-down."""
        model_type = self.qcombo_model_type.currentText()
        model_class = ALL_MODELS[model_type]
        temp_model = model_class(model_name=model_type, use_cuda=self.check_use_cuda.isChecked())
        
        if isinstance(temp_model, Hookmodel):
            self._create_output_selection_for_temp_model(temp_model)
            self.set_nnmodel_outputs_btn.setEnabled(True)
            self.model_output_selection.setEnabled(True)
        else:
            self.model_output_selection.clear()
            self.set_nnmodel_outputs_btn.setEnabled(False)
            self.model_output_selection.setEnabled(False)

        #if model is DINOv2, set some recommended settings
        if model_type == 'dinov2_vits14_reg':
            self.spin_interpolation_order.setValue(0)
            self.check_use_min_features.setChecked(False)
            self.check_tile_annotations.setChecked(False)
            self.check_tile_image.setChecked(False)
            self.num_scales_combo.setCurrentText('[1]')


    def _create_output_selection_for_temp_model(self, temp_model):
        self.model_output_selection.clear()
        self.model_output_selection.addItems(temp_model.selectable_layer_keys.keys())

