from qtpy.QtWidgets import (QWidget, QPushButton,QVBoxLayout,
                            QLabel, QComboBox,QFileDialog, QListWidget,
                            QCheckBox, QAbstractItemView, QGridLayout, QSpinBox, QButtonGroup,
                            QRadioButton,QDoubleSpinBox)
from qtpy.QtCore import Qt
from magicgui.widgets import create_widget
import napari
from napari.utils import progress
from napari_guitils.gui_structures import VHGroup, TabSet
from pathlib import Path
import numpy as np
import warnings
from datetime import datetime

from .conv_paint import (get_features_current_layers,
                         get_all_models, train_classifier,
                         load_model, create_model, save_model)
from .conv_paint_utils import (parallel_predict_image, 
                               normalize_image, compute_image_stats)
from .conv_parameters import Param
from .conv_paint_nnlayers import Hookmodel




### Define the main widget class

class ConvPaintWidget(QWidget):
    """
    Implementation of a napari widget for interactive segmentation performed
    via a CatBoost Classifier trained on annotations. The filters used to 
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

### Define the basic structure of the widget
    
    def __init__(self, napari_viewer, parent=None, init_project=False, third_party=False):
        super().__init__(parent=parent)
        self.viewer = napari_viewer
        self.image_mean = None
        self.image_std = None
        self.param = Param()
        self.model = None
        self.trained = False
        self.temp_model = None
        self.classifier = None
        self.project_widget = None
        self.features_per_layer = None
        self.selected_channel = None
        self.third_party = third_party
        # Plugin options
        self.keep_layers = False
        self.auto_seg = False
        self.use_custom_model = False

        ### Define Constants

        # Define the default model
        self.DEFAULT_FE = 'vgg16'
        self.DEFAULT_LAYERS_INDEX = 0
        self.DEFAULT_SCALINGS_TEXT = '[1,2]'
        self.DEFAULT_INTERPOLATION_ORDER = 1
        self.DEFAULT_USE_MIN_FEATURES = True
        self.DEFAULT_USE_CUDA = False

        # Define the default classifier parameters
        self.DEFAULT_ITERATIONS = 50
        self.DEFAULT_LEARNING_RATE = 0.1
        self.DEFAULT_DEPTH = 5

        # Create main layout
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # Create and add tabs
        self.tab_names = ['Home', 'Project', 'Model options']
        self.tabs = TabSet(self.tab_names, tab_layouts=[None, None, QGridLayout()])
        self.main_layout.addWidget(self.tabs)
        # Disable project tab as long as not activated
        self.tabs.setTabEnabled(self.tabs.tab_names.index('Project'), False) # R: Why only index? If custom is "respnsive", so should be the index
        
        # Align rows in the tabs "Home" and "Model options" on top
        self.tabs.widget(self.tabs.tab_names.index('Home')).layout().setAlignment(Qt.AlignTop)
        self.tabs.widget(self.tabs.tab_names.index('Model options')).layout().setAlignment(Qt.AlignTop)

        # === HOME TAB ===

        # Create groups
        self.layer_selection_group = VHGroup('Layer selection', orientation='G')
        self.train_group = VHGroup('Train', orientation='G')
        self.segment_group = VHGroup('Segment', orientation='G')
        self.load_save_group = VHGroup('Load/Save', orientation='G')
        self.image_processing_group = VHGroup('Image processing', orientation='G')
        self.acceleration_group = VHGroup('Acceleration', orientation='G')
        self.model_group = VHGroup('Model', orientation='G')

        # Add groups to the tab
        self.tabs.add_named_tab('Home', self.layer_selection_group.gbox)
        self.tabs.add_named_tab('Home', self.train_group.gbox)
        self.tabs.add_named_tab('Home', self.segment_group.gbox)
        self.tabs.add_named_tab('Home', self.load_save_group.gbox)
        self.tabs.add_named_tab('Home', self.image_processing_group.gbox)
        self.tabs.add_named_tab('Home', self.acceleration_group.gbox)
        self.tabs.add_named_tab('Home', self.model_group.gbox)

        # Add elements to "Layer selection" group
        # Image layer (widget for selecting the layer to segment)
        self.image_layer_selection_widget = create_widget(annotation=napari.layers.Image, label='Pick image')
        self.image_layer_selection_widget.reset_choices()
        # Annotation layer
        self.annotation_layer_selection_widget = create_widget(annotation=napari.layers.Labels, label='Pick annotation')
        self.annotation_layer_selection_widget.reset_choices()

        # Add widgets to layout
        self.layer_selection_group.glayout.addWidget(QLabel('Image layer'), 0,0,1,1)
        self.layer_selection_group.glayout.addWidget(self.image_layer_selection_widget.native, 0,1,1,1)
        self.layer_selection_group.glayout.addWidget(QLabel('Annotation layer'), 1,0,1,1)
        self.layer_selection_group.glayout.addWidget(self.annotation_layer_selection_widget.native, 1,1,1,1)
        # Add button for adding annotation/segmentation layers
        self.add_layers_btn = QPushButton('Add annotations/segmentation layers')
        self.add_layers_btn.setEnabled(True)
        self.layer_selection_group.glayout.addWidget(self.add_layers_btn, 2,0,1,2)
        # Checkbox for keeping old layers
        # self.check_keep_layers = QCheckBox('Keep old layers')
        # self.check_keep_layers.setToolTip('Keep old annotation and segmentation layers when creating new ones.')
        # self.check_keep_layers.setChecked(self.keep_layers)
        # self.layer_selection_group.glayout.addWidget(self.check_keep_layers, 2,1,1,1)

        # Add buttons for "Train" group
        self.train_classifier_btn = QPushButton('Train')
        self.train_classifier_btn.setToolTip('Train model on annotations')
        self.train_group.glayout.addWidget(self.train_classifier_btn, 0,0,1,1)
        self.check_auto_seg = QCheckBox('Auto segment')
        self.check_auto_seg.setToolTip('Automatically segment image after training')
        self.check_auto_seg.setChecked(self.auto_seg)
        self.train_group.glayout.addWidget(self.check_auto_seg, 0,1,1,1)
        # Project checkbox
        self.check_use_project = QCheckBox('Project mode (multiple files)')
        self.check_use_project.setToolTip('Activate Project tab to use multiple files for training the classifier')
        self.check_use_project.setChecked(False)
        self.train_group.glayout.addWidget(self.check_use_project, 1,0,1,1)
        # Project button
        self.train_classifier_on_project_btn = QPushButton('Train on Project')
        self.train_classifier_on_project_btn.setToolTip('Train on all images loaded in Project tab')
        self.train_group.glayout.addWidget(self.train_classifier_on_project_btn, 1,1,1,1)
        if init_project is False:
            self.train_classifier_on_project_btn.setEnabled(False)

        # Add buttons for "Segment" group
        self.segment_btn = QPushButton('Segment image')
        self.segment_btn.setEnabled(False)
        self.segment_btn.setToolTip('Segment 2D image or current slice/frame of 3D image/movie ')
        self.segment_group.glayout.addWidget(self.segment_btn, 0,0,1,1)
        self.segment_all_btn = QPushButton('Segment stack')
        self.segment_all_btn.setToolTip('Segment all slices/frames of 3D image/movie')
        self.segment_all_btn.setEnabled(False)
        self.segment_group.glayout.addWidget(self.segment_all_btn, 0,1,1,1)

        # Add buttons for "Load/Save" group
        self.save_model_btn = QPushButton('Save trained model')
        self.save_model_btn.setToolTip('Save model as *.pickle file')
        self.save_model_btn.setEnabled(False)
        self.load_save_group.glayout.addWidget(self.save_model_btn, 0,0,1,1)
        self.load_model_btn = QPushButton('Load trained model')
        self.load_model_btn.setToolTip('Select *.pickle file to load as trained model')
        self.load_save_group.glayout.addWidget(self.load_model_btn, 0,1,1,1)

        # Add buttons for "Image Processing" group
        # Radio buttons for "Data Dimensions"
        self.button_group_channels = QButtonGroup()
        self.radio_single_channel = QRadioButton('Single channel image/stack')
        self.radio_single_channel.setToolTip('2D images or 3D images where additional dimension is NOT channels')
        self.radio_multi_channel = QRadioButton('Multichannel image')
        self.radio_multi_channel.setToolTip('Images with an additional channel dimension')
        self.radio_rgb = QRadioButton('RGB image')
        self.radio_rgb.setToolTip('Use this option with images displayed as RGB')
        self.radio_single_channel.setChecked(True)
        self.channel_buttons = [self.radio_single_channel, self.radio_multi_channel, self.radio_rgb]
        for x in self.channel_buttons: x.setEnabled(False)
        self.button_group_channels.addButton(self.radio_single_channel, id=1)
        self.button_group_channels.addButton(self.radio_multi_channel, id=2)
        self.button_group_channels.addButton(self.radio_rgb, id=3)
        self.image_processing_group.glayout.addWidget(self.radio_single_channel, 0,0,1,1)
        self.image_processing_group.glayout.addWidget(self.radio_multi_channel, 1,0,1,1)
        self.image_processing_group.glayout.addWidget(self.radio_rgb, 2,0,1,1)
        # "Normalize" radio buttons
        self.button_group_normalize = QButtonGroup()
        self.radio_no_normalize = QRadioButton('No normalization')
        self.radio_no_normalize.setToolTip('No normalization is applied')
        self.radio_normalized_over_stack = QRadioButton('Normalize over stack')
        self.radio_normalized_over_stack.setToolTip('Normalize over complete z or t stack')
        self.radio_normalize_by_image = QRadioButton('Normalized by plane')
        self.radio_normalize_by_image.setToolTip('Normalize each plane individually')
        self.radio_normalized_over_stack.setChecked(True)
        self.norm_buttons = [self.radio_no_normalize, self.radio_normalized_over_stack, self.radio_normalize_by_image]
        self.button_group_normalize.addButton(self.radio_no_normalize, id=1)
        self.button_group_normalize.addButton(self.radio_normalized_over_stack, id=2)
        self.button_group_normalize.addButton(self.radio_normalize_by_image, id=3)
        self.image_processing_group.glayout.addWidget(self.radio_no_normalize, 0,1,1,1)
        self.image_processing_group.glayout.addWidget(self.radio_normalized_over_stack, 1,1,1,1)
        self.image_processing_group.glayout.addWidget(self.radio_normalize_by_image, 2,1,1,1)

        # Add buttons for "Model" group
        # Current model label
        self.model_description1 = QLabel('None')
        self.model_group.glayout.addWidget(self.model_description1, 0,0,1,2)
        self.current_model_path = QLabel('not trained')
        # self.model_group.glayout.addWidget(self.current_model_path, 0,1,1,1)
        # "Use custom Model" checkbox
        # self.check_use_custom_model = QCheckBox('Use custom Model')
        # self.check_use_custom_model.setToolTip('Activate Tab to customize Feature Extractor and Classifier')
        # self.check_use_custom_model.setChecked(False)
        # self.model_group.glayout.addWidget(self.check_use_custom_model, 1,0,1,1)
        # Reset model button
        self._reset_model_btn = QPushButton('Reset to default model')
        self._reset_model_btn.setToolTip('Discard current model, create new default model.')
        self.model_group.glayout.addWidget(self._reset_model_btn, 1,0,1,2)

        # Add elements to "Acceleration" group
        # "Downsample" spinbox
        self.spin_downsample = QSpinBox()
        self.spin_downsample.setMinimum(1)
        self.spin_downsample.setMaximum(10)
        self.spin_downsample.setValue(1)
        self.spin_downsample.setToolTip('Reduce image size for faster computing.')
        self.acceleration_group.glayout.addWidget(QLabel('Downsample'), 0,0,1,1)
        self.acceleration_group.glayout.addWidget(self.spin_downsample, 0,1,1,1)
        # "Tile annotations" checkbox
        self.check_tile_annotations = QCheckBox('Tile annotations for training')
        self.check_tile_annotations.setChecked(False)
        self.check_tile_annotations.setToolTip('Crop around annotated regions to speed up training.\nDisable for models that extract long range features (e.g. DINO)!')
        self.acceleration_group.glayout.addWidget(self.check_tile_annotations, 1,0,1,1)
        # "Tile image" checkbox
        self.check_tile_image = QCheckBox('Tile image for segmentation')
        self.check_tile_image.setChecked(False)
        self.check_tile_image.setToolTip('Tile image to reduce memory usage.\nTake care when using models that extract long range features (e.g. DINO).')
        self.acceleration_group.glayout.addWidget(self.check_tile_image, 1,1,1,1)

        # === MODEL TAB ===

        # Create three groups, 'Current model', 'Feature extractor' and 'Classifier parameters (CatBoost)'
        self.current_model_group = VHGroup('Current model', orientation='G')
        self.fe_group = VHGroup('Feature extractor', orientation='G')
        self.classifier_params_group = VHGroup('Classifier (CatBoost)', orientation='G')
        # Add groups to the tab
        self.tabs.add_named_tab('Model options', self.current_model_group.gbox, [0, 0, 1, 2])
        self.tabs.add_named_tab('Model options', self.fe_group.gbox, [2, 0, 8, 2])
        self.tabs.add_named_tab('Model options', self.classifier_params_group.gbox, [10, 0, 3, 2])
        self.tabs.setTabEnabled(self.tabs.tab_names.index('Model options'), True)
        
        # Current model and update button
        self.model_description2 = QLabel('None')
        self.current_model_group.glayout.addWidget(self.model_description2, 0, 0, 1, 2)

        # Add "FE architecture" combo box to FE group
        self.qcombo_model_type = QComboBox()
        self.qcombo_model_type.addItems(sorted(get_all_models().keys()))
        self.qcombo_model_type.setToolTip('Select architecture of feature extraction model.')
        self.fe_group.glayout.addWidget(self.qcombo_model_type, 2, 0, 1, 2)

        # Add "FE layers" list to FE group
        self.fe_layer_selection = QListWidget()
        self.fe_layer_selection.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.fe_layer_selection.setFixedHeight(200)
        # self.fe_layer_selection.setMaximumHeight(200)
        # self.fe_layer_selection.setMinimumHeight(40)
        # Turn off stretching
        # self.fe_group.glayout.setRowStretch(2, 0)
        self.fe_group.glayout.addWidget(self.fe_layer_selection, 3, 0, 1, 2)

        # Create and add scales selection to FE group
        self.fe_scaling_factors = QComboBox()
        self.fe_scaling_factors.addItem('[1]',[1])
        self.fe_scaling_factors.addItem('[1,2]',[1,2])
        self.fe_scaling_factors.addItem('[1,2,4]',[1,2,4])
        self.fe_scaling_factors.addItem('[1,2,4,8]',[1,2,4,8])
        self.fe_scaling_factors.setCurrentText('[1,2]')
        self.fe_group.glayout.addWidget(QLabel('Downscaling factors'), 4, 0, 1, 1)
        self.fe_group.glayout.addWidget(self.fe_scaling_factors, 4, 1, 1, 1)

        # Add interpolation order spinbox to FE group
        self.spin_interpolation_order = QSpinBox()
        self.spin_interpolation_order.setMinimum(0)
        self.spin_interpolation_order.setMaximum(5)
        self.spin_interpolation_order.setValue(1)
        self.spin_interpolation_order.setToolTip('Interpolation order for image rescaling')
        self.fe_group.glayout.addWidget(QLabel('Interpolation order'), 5, 0, 1, 1)
        self.fe_group.glayout.addWidget(self.spin_interpolation_order, 5, 1, 1, 1)

        # Add min features checkbox to FE group
        self.check_use_min_features = QCheckBox('Use min features')
        self.check_use_min_features.setChecked(False)
        self.check_use_min_features.setToolTip('Use same number of features from each layer. Otherwise use all features from each layer.')
        self.fe_group.glayout.addWidget(self.check_use_min_features, 6, 0, 1, 1)

        # Add use cuda checkbox to FE group
        self.check_use_cuda = QCheckBox('Use cuda')
        self.check_use_cuda.setChecked(False)
        self.check_use_cuda.setToolTip('Use GPU for training and segmentation')
        self.fe_group.glayout.addWidget(self.check_use_cuda, 6, 1, 1, 1)

        # Add "set" buttons to FE group
        self.set_fe_btn = QPushButton('Set feature extractor')
        self.set_fe_btn.setToolTip('Set the feature extraction model')
        self.fe_group.glayout.addWidget(self.set_fe_btn, 7, 0, 1, 1)
        self.reset_default_fe_btn = QPushButton('Reset to default')
        self.reset_default_fe_btn.setToolTip('Set the feature extractor back to the default model')
        self.fe_group.glayout.addWidget(self.reset_default_fe_btn, 7, 1, 1, 1)

        # Add classifier parameters
        self.spin_iterations = QSpinBox()
        self.spin_iterations.setMinimum(1)
        self.spin_iterations.setMaximum(1000)
        self.spin_iterations.setValue(self.DEFAULT_ITERATIONS)
        self.spin_iterations.setToolTip('Set the number of iterations for the classifier')
        self.classifier_params_group.glayout.addWidget(QLabel('Iterations'), 0, 0, 1, 1)
        self.classifier_params_group.glayout.addWidget(self.spin_iterations, 0, 1, 1, 1)

        self.spin_learning_rate = QDoubleSpinBox()
        self.spin_learning_rate.setMinimum(0.001)
        self.spin_learning_rate.setMaximum(1.0)
        self.spin_learning_rate.setSingleStep(0.01)
        self.spin_learning_rate.setValue(self.DEFAULT_LEARNING_RATE)
        self.spin_learning_rate.setToolTip('Set the learning rate for the classifier')
        self.classifier_params_group.glayout.addWidget(QLabel('Learning Rate'), 1, 0, 1, 1)
        self.classifier_params_group.glayout.addWidget(self.spin_learning_rate, 1, 1, 1, 1)

        self.spin_depth = QSpinBox()
        self.spin_depth.setMinimum(1)
        self.spin_depth.setMaximum(20)
        self.spin_depth.setValue(self.DEFAULT_DEPTH)
        self.spin_depth.setToolTip('Set the depth of the trees for the classifier')
        self.classifier_params_group.glayout.addWidget(QLabel('Depth'), 2, 0, 1, 1)
        self.classifier_params_group.glayout.addWidget(self.spin_depth, 2, 1, 1, 1)

        self.set_default_classif_btn = QPushButton('Restore defaults')
        self.set_default_classif_btn.setEnabled(True)
        self.classifier_params_group.glayout.addWidget(self.set_default_classif_btn, 3, 0, 1, 2)

        # === === ===

        # If project mode is initially activated, add project tab and widget
        if init_project is True:
            self._on_use_project()

        # Add connections and initialize by selecting layer and setting default model and params
        self.add_connections()
        self._set_default_model()
        self._on_select_layer()

        # Add key bindings
        self.viewer.bind_key('a', self.toggle_annotation, overwrite=True)
        self.viewer.bind_key('r', self.toggle_prediction, overwrite=True)

    def add_connections(self):

        # Reset layer choices in dropdown when layers are renamed; also bind this behaviour to inserted layers
        for layer in self.viewer.layers:
            layer.events.name.connect(self.image_layer_selection_widget.reset_choices)
        self.viewer.layers.events.inserted.connect(self._on_insert_layer)

        # Reset layer choices in dropdowns when napari layers are added or removed
        for layer_widget_reset in [self.annotation_layer_selection_widget.reset_choices,
                                   self.image_layer_selection_widget.reset_choices]:
            self.viewer.layers.events.inserted.connect(layer_widget_reset)
            self.viewer.layers.events.removed.connect(layer_widget_reset)
        
        # Change layer selection when choosing a new image layer in dropdown
        self.image_layer_selection_widget.changed.connect(self._on_select_layer)
        # NOTE: Changing annotation layer does not have to trigger anything
        self.add_layers_btn.clicked.connect(self._on_add_annot_seg_layers)
        self.check_keep_layers.stateChanged.connect(lambda: setattr(
            self, 'keep_layers', self.check_keep_layers.isChecked()))


        # Train
        self.train_classifier_btn.clicked.connect(self._on_train)
        self.check_auto_seg.stateChanged.connect(lambda: setattr(
            self, 'auto_seg', self.check_auto_seg.isChecked()))
        self.check_use_project.stateChanged.connect(self._on_use_project)
        self.train_classifier_on_project_btn.clicked.connect(self._on_train_on_project)

        # Segment
        self.segment_btn.clicked.connect(self._on_predict)
        self.segment_all_btn.clicked.connect(self._on_predict_all)

        # Load/Save
        self.save_model_btn.clicked.connect(self._on_save_model)
        self.load_model_btn.clicked.connect(self._on_load_model)

        # Image Processing
        self.radio_single_channel.toggled.connect(self._on_data_dim_changed)
        self.radio_multi_channel.toggled.connect(self._on_data_dim_changed)
        self.radio_rgb.toggled.connect(self._on_data_dim_changed)
        self.radio_no_normalize.toggled.connect(self._on_norm_changed)
        self.radio_normalized_over_stack.toggled.connect(self._on_norm_changed)
        self.radio_normalize_by_image.toggled.connect(self._on_norm_changed)

        # Model
        # self.check_use_custom_model.stateChanged.connect(self._on_use_custom_model)
        self._reset_model_btn.clicked.connect(self._on_reset_model)

        # Acceleration
        self.spin_downsample.valueChanged.connect(lambda: setattr(
            self.param, 'image_downsample', self.spin_downsample.value()))
        self.check_tile_annotations.stateChanged.connect(lambda: setattr(
            self.param, 'tile_annotations', self.check_tile_annotations.isChecked()))
        self.check_tile_image.stateChanged.connect(lambda: setattr(
            self.param, 'tile_image', self.check_tile_image.isChecked()))

        # Model tab
        # Feature extractor
        self.qcombo_model_type.currentIndexChanged.connect(self._on_model_selected)
        self.set_fe_btn.clicked.connect(self._on_set_fe_model)
        self.reset_default_fe_btn.clicked.connect(self._on_reset_default_fe)
        self.fe_layer_selection.itemSelectionChanged.connect(self._on_fe_layer_selection_changed)
        self.fe_scaling_factors.currentIndexChanged.connect(self._on_fe_scalings_changed)
        # The changing of these parameters shall only be applied when the user clicks the button
        # self.spin_interpolation_order.valueChanged.connect(lambda: setattr(self.param, 'order', self.spin_interpolation_order.value()))
        # self.check_use_cuda.stateChanged.connect(lambda: setattr(self.param, 'use_cuda', self.check_use_cuda.isChecked()))
        # self.check_use_min_features.stateChanged.connect(lambda: setattr(self.param, 'use_min_features', self.check_use_min_features.isChecked()))
        # Classifier
        self.spin_iterations.valueChanged.connect(lambda: setattr(self.param, 'classif_iterations', self.spin_iterations.value()))
        self.spin_learning_rate.valueChanged.connect(lambda: setattr(self.param, 'classif_learning_rate', self.spin_learning_rate.value()))
        self.spin_depth.valueChanged.connect(lambda: setattr(self.param, 'classif_depth', self.spin_depth.value()))
        self.set_default_classif_btn.clicked.connect(self._on_reset_classif_params)

    # Visibility toggles for key bindings

    def toggle_annotation(self, event=None):
        """Hide annotation layer."""

        if self.viewer.layers['annotations'].visible == False:
            self.viewer.layers['annotations'].visible = True
        else:
            self.viewer.layers['annotations'].visible = False

    def toggle_prediction(self, event=None):
        """Hide prediction layer."""

        if self.viewer.layers['segmentation'].visible == False:
            self.viewer.layers['segmentation'].visible = True
        else:
            self.viewer.layers['segmentation'].visible = False

### Define the detailed behaviour of the widget

    # Handling of Napari layers

    def _on_insert_layer(self, event=None):
        """Bind the update of layer choices in dropdowns to the renaming of inserted layers."""
        layer = event.value
        layer.events.name.connect(self.image_layer_selection_widget.reset_choices)
        layer.events.name.connect(self.annotation_layer_selection_widget.reset_choices)

    # Layer selection

    def _on_select_layer(self, newtext=None):
        """Assign the layer to segment and update data radio buttons accordingly"""
        self.selected_channel = self.image_layer_selection_widget.native.currentText()
        if self.image_layer_selection_widget.value is not None:
            self._on_data_dim_changed()

        # Enable button to add annotation and segmentation layers if there is data in the image layer
        if self.image_layer_selection_widget.value is None:
            self.add_layers_btn.setEnabled(False)
        else:
            self.add_layers_btn.setEnabled(True)
        
        # Set radio buttons depending on selected image type
        if self.image_layer_selection_widget.value is not None:
            self._reset_radio_data_dims()
            self._reset_radio_norm_settings()
            self._reset_predict_buttons()

    def _on_add_annot_seg_layers(self, event=None, force_add=True):
        """Add empty annotation and segmentation layers if not already present."""
        self._add_empty_layers(event, force_add)

    # Train

    def _on_train(self):
        """Given a set of new annotations, update the CatBoost classifier."""

        # Check if annotations of at least 2 classes are present
        unique_labels = np.unique(self.annotation_layer_selection_widget.value.data)
        unique_labels = unique_labels[unique_labels != 0]
        if len(unique_labels) < 2:
            raise Exception('You need annotations for at least foreground and background')
        # Check if model is loaded
        if self.model is None:
            if not self.use_custom_model:
                self._set_default_model()
            else:
                raise Exception('You have to define and load a model first')

        image_stack = self.get_data_channel_first_norm()
        
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(True)

        with progress(total=0) as pbr:
            self.current_model_path.setText('in training')
            self._set_model_description()
            pbr.set_description(f"Training")
            features, targets = get_features_current_layers(
                image=image_stack,
                annotations=self.annotation_layer_selection_widget.value.data,
                model=self.model,
                param=self.param,
            )
            self.classifier = train_classifier(
                features, targets,
                iterations=self.param.classif_iterations,
                learning_rate=self.param.classif_learning_rate,
                depth=self.param.classif_depth,
        )
            self.current_model_path.setText('trained, unsaved')
            self.trained = True
            self._reset_predict_buttons()
            self.save_model_btn.setEnabled(True)
            self._set_model_description()
            
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(False)

        if self.auto_seg:
            self._on_predict()

    def _on_use_project(self, event=None):
        """Add widget for multi-image project management if not already added."""

        if self.check_use_project.isChecked():
            if self.project_widget is None:
                from napari_annotation_project.project_widget import ProjectWidget
                self.project_widget = ProjectWidget(napari_viewer=self.viewer)

                # self.tabs.add_named_tab('Project', self.project_widget)
                self.tabs.add_named_tab('Project', self.project_widget.file_list)
                self.tabs.add_named_tab('Project', self.project_widget.btn_add_file)
                self.tabs.add_named_tab('Project', self.project_widget.btn_remove_file)
                self.tabs.add_named_tab('Project', self.project_widget.btn_save_annotation)
                self.tabs.add_named_tab('Project', self.project_widget.btn_load_project)
            
            self.tabs.setTabEnabled(self.tabs.tab_names.index('Project'), True)
            self.train_classifier_on_project_btn.setEnabled(True)
        else:
            self.tabs.setTabEnabled(self.tabs.tab_names.index('Project'), False)
            self.train_classifier_on_project_btn.setEnabled(False)

    def _on_train_on_project(self):
        """Train classifier on all annotations in project.
        !!!! Need to double-check if normalization is done correctly for projects !!!!"""

        if self.model is None:
            if not self.use_custom_model:
                self._set_default_model()
            else:
                raise Exception('You have to define and load a model first')

        num_files = len(self.project_widget.params.file_paths)
        if num_files == 0:
            raise Exception('No files found')

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(True)
        
        self.viewer.layers.events.removed.disconnect(self._on_reset_model)
        with progress(total=0) as pbr:
            pbr.set_description(f"Training")
            self.current_model_path.setText('in training')
            all_features, all_targets = [], []
            for ind in range(num_files):
                self.project_widget.file_list.setCurrentRow(ind)

                image_stack = self.get_data_channel_first_norm()

                features, targets = get_features_current_layers(
                    model=self.model,
                    image=image_stack,
                    annotations=self.annotation_layer_selection_widget.value.data,
                    param=self.param,
                )
                if features is None:
                    continue
                all_features.append(features)
                all_targets.append(targets)
            
            all_features = np.concatenate(all_features, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            self.classifier = train_classifier(all_features, all_targets)

            self.current_model_path.setText('trained, unsaved')
            self.trained = True
            self._reset_predict_buttons()
            self.save_model_btn.setEnabled(True)
            self._set_model_description()

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(False)

        self.viewer.layers.events.removed.connect(self._on_reset_model)

    # Predict

    def _on_predict(self):
        """Predict the segmentation of the currently viewed frame based 
        on a RF model trained with annotations"""

        if self.model is None:
            if not self.use_custom_model:
                self._set_default_model()
            else:
                raise Exception('You have to define and load a model first')
        
        if self.classifier is None:
            self._on_train()

        self._check_create_prediction_layer()

        if self.image_mean is None:
            self.get_image_stats()
        
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(True)

        with progress(total=0) as pbr:
            
            pbr.set_description(f"Prediction")

            data_dims = self.get_data_dims()

            if data_dims in ['2D', '3D_multi', '2D_RGB']:
                image = self.get_data_channel_first()
                if self.param.normalize != 1:
                    image_mean = self.image_mean
                    image_std = self.image_std

            if data_dims == '3D_single':
                step = self.viewer.dims.current_step[0]
                image = self.viewer.layers[self.selected_channel].data[step]
                if self.param.normalize == 2: # over stack
                    image_mean = self.image_mean
                    image_std = self.image_std
                if self.param.normalize == 3: # by image
                    image_mean = self.image_mean[step]
                    image_std = self.image_std[step]
            
            if data_dims == '3D_RGB':
                step = self.viewer.dims.current_step[0]
                image = self.viewer.layers[self.selected_channel].data[step]
                image = np.moveaxis(image, -1, 0)
                if self.param.normalize == 2: # over stack
                    image_mean = self.image_mean[:,0,::]
                    image_std = self.image_std[:,0,::]
                if self.param.normalize == 3: # by image
                    image_mean = self.image_mean[:,step]
                    image_std = self.image_std[:,step]

            if data_dims == '4D':
                # for 4D, channel is always first, t/z second
                step = self.viewer.dims.current_step[1]
                image = self.viewer.layers[self.selected_channel].data[:, step]
                if self.param.normalize == 2: # over stack
                    image_mean = self.image_mean[:,0,::]
                    image_std = self.image_std[:,0,::]
                if self.param.normalize == 3: # by image
                    image_mean = self.image_mean[:,step]
                    image_std = self.image_std[:,step]
            
            # Normalize image (using the stats depending on the radio buttons)
            if self.param.normalize != 1:
                image = normalize_image(image=image, image_mean=image_mean, image_std=image_std)

            if self.param.tile_image:
                predicted_image = parallel_predict_image(
                    image=image, model=self.model, classifier=self.classifier,
                    param = self.param, use_dask=False)
            else:
                predicted_image = self.model.predict_image(image=image,
                                                           classifier=self.classifier,
                                                           param=self.param,)
            if data_dims in ['2D', '2D_RGB', '3D_multi']:
                self.viewer.layers['segmentation'].data = predicted_image
            else: # 3D_single, 4D, 3D_RGB
                self.viewer.layers['segmentation'].data[step] = predicted_image
            self.viewer.layers['segmentation'].refresh()

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(False)

    def _on_predict_all(self):
        """Predict the segmentation of all frames based 
        on a RF model trained with annotations"""

        if self.classifier is None:
            raise Exception('No model found. Please train a model first.')
        
        if self.model is None:
            if not self.use_custom_model:
                self._set_default_model()
            else:
                raise Exception('You have to define and load a model first')
            
        self._check_create_prediction_layer()

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
            
            if self.param.tile_image:
                predicted_image = parallel_predict_image(image=image, 
                                                         model=self.model,
                                                         classifier=self.classifier,
                                                         param = self.param,
                                                         use_dask=False)
            else:
                predicted_image = self.model.predict_image(image=image,
                                                           classifier=self.classifier,
                                                           param=self.param)
            self.viewer.layers['segmentation'].data[step] = predicted_image
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(False)

    # Load/Save

    def _on_save_model(self, event=None, save_file=None):
            """Select file where to save the classifier along with the model parameters."""
            if self.classifier is None:
                raise Exception('No trained classifier found. Please train a model first.')
            
            if save_file is None:
                dialog = QFileDialog()
                save_file, _ = dialog.getSaveFileName(self, "Save model", None, "PICKLE (*.pickle)")
            save_file = Path(save_file)
            
            # self._update_fe_params_from_gui() # NOTE: Doesn't make sense !!
            save_model(save_file, self.classifier, self.model, self.param, )        
            self.current_model_path.setText(save_file.name)
            self._set_model_description()

    def _on_load_model(self, event=None, save_file=None):
        """Select classifier file to load along with the model parameters."""
        if save_file is None:
            dialog = QFileDialog()
            save_file, _ = dialog.getOpenFileName(self, "Choose model", None, "PICKLE (*.pickle)")
        save_file = Path(save_file)

        # Tick the custom model checkbox
        # self.check_use_custom_model.setChecked(True)
        
        # Load model
        classifier, model, param, model_state = load_model(save_file)
        self.classifier = classifier
        self.param = param
        self.model = model

        # Load model state if available
        if model_state is not None and hasattr(self.model, 'load_state_dict'):
            self.model.load_state_dict(model_state)
        
        # Update GUI
        # Select the model type in the dropdown
        self.qcombo_model_type.setCurrentText(self.param.model_name)
        self._update_gui_from_param()
        self._update_gui_fe_layers_from_model()
        self.current_model_path.setText(save_file.name)
        self.trained = True
        self.save_model_btn.setEnabled(True)
        self._set_model_description()
        self._reset_predict_buttons()

    # Image Processing

    def _on_data_dim_changed(self):
        """Set the image data dimensions based on radio buttons,
        and adjust normalization options."""
        self._add_empty_layers()
        self.param.multi_channel_img = self.radio_multi_channel.isChecked()
        self._reset_radio_norm_settings()
        self._reset_classif()

    def _on_norm_changed(self):
        """Set the normalization options based on radio buttons,
        and update stats."""
        if self.radio_no_normalize.isChecked():
            self.param.normalize = 1
        elif self.radio_normalized_over_stack.isChecked():
            self.param.normalize = 2
        elif self.radio_normalize_by_image.isChecked():
            self.param.normalize = 3
        self._reset_stats()
        self._reset_classif()

    # Model

    # def _on_use_custom_model(self, event=None):
    #     """Change custom model parameter based on flag,
    #     and add tab for custom model management if needed."""
    #     if self.check_use_custom_model.isChecked():
    #         self.use_custom_model = True
    #         self.tabs.setTabEnabled(self.tabs.tab_names.index('Model options'), True)
    #     else:
    #         self.use_custom_model = False
    #         self.tabs.setTabEnabled(self.tabs.tab_names.index('Model options'), False)
    #         # self.qcombo_model_type.setCurrentText('single_layer_vgg16')
    #         self._set_default_model()

    def _on_reset_model(self, event=None):
        """Reset model to default and update GUI."""
        # self.model = None
        self.save_model_btn.setEnabled(False)
        self._set_default_model()
        self._reset_classif()
        self._reset_radio_data_dims()
        self._reset_radio_norm_settings()

    # Model Tab

    def _on_model_selected(self, index):
        """Update GUI to show selectable layers of model chosen from drop-down."""
        model_type = self.qcombo_model_type.currentText()
        model_class = get_all_models()[model_type]
        self.temp_model = model_class(model_name=model_type, use_cuda=self.param.use_cuda)

        # Get selectable layers from the temp model and update the GUI
        if isinstance(self.temp_model, Hookmodel):
            self._create_fe_layer_selection_from_temp_model(self.temp_model)
            self.fe_layer_selection.setEnabled(True)
            #check if outputs are selected
            if self.fe_layer_selection.count() > 0:
                self.set_fe_btn.setEnabled(False)
        else:
            self.fe_layer_selection.clear()
            self.fe_layer_selection.setEnabled(False)
            self.set_fe_btn.setEnabled(True)

        # Get the default FE params for the temp model and update the GUI
        default_params = self.temp_model.get_default_params().as_dict()
        for param_name, default_val in default_params.items():
            if default_val is not None:
                setattr(self.param, param_name, default_val)
        self._update_gui_from_param()

    def _on_fe_layer_selection_changed(self):
        #check if temp model is a hookmodel
        if isinstance(self.temp_model, Hookmodel):
            selected_layers = self.get_selected_layer_names()
            if len(selected_layers) == 0:
                self.set_fe_btn.setEnabled(False)
            else:
                self.set_fe_btn.setEnabled(True)
        else:
            self.set_fe_btn.setEnabled(True)

    def _on_fe_scalings_changed(self):
        # self.param.scalings = self.fe_scaling_factors.currentData()
        # self._reset_classif()
        return

    def _on_set_fe_model(self, event=None):
        """Create a neural network model that will be used for feature extraction and
        reset the classifier."""
        self._update_fe_params_from_gui()
        # self.param.model_name = self.qcombo_model_type.currentText()
        self.model = create_model(self.param)
        self._reset_classif()
        self._update_gui_fe_layers_from_model()

    def _on_reset_default_fe(self, event=None):
        self._reset_fe_params()

    def _on_reset_classif_params(self):
        self._reset_classif_params()
        self._reset_classif()
    

### Helper functions

    # R: Check this in detail
    def _add_empty_layers(self, event=None, force_add=True):
        """Add annotation and prediction layers to viewer. If the layer already exists,
        remove it and add a new one. If the widget is used as third party (self.third_party=True),
        no layer is added if it didn't exist before, unless force_add=True (e.g. when the user click
        on the add layer button)"""

        if self.keep_layers:
            self.create_annot_seg_copies()

        self.event = event # R: Why is this needed?
        if self.image_layer_selection_widget.value is None:
            raise Exception('Please select an image layer first')
        
        if self.viewer.layers[self.selected_channel].rgb:
            layer_shape = self.viewer.layers[self.selected_channel].data.shape[0:-1]
        elif self.param.multi_channel_img:
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
        
        # Activate the annotation layer, select it in the dropdown and activate paint mode
        if 'annotations' in self.viewer.layers:
            self.viewer.layers.selection.active = self.viewer.layers['annotations']
            self.annotation_layer_selection_widget.value = self.viewer.layers['annotations']
            self.viewer.layers['annotations'].mode = 'paint'

    def create_annot_seg_copies(self):
        """Create copies of the annotations and segmentation layers."""
        image_name = self.image_layer_selection_widget.value.name
        data_dim = self.radio_rgb.isChecked()*"RGB" + self.radio_multi_channel.isChecked()*"multiCh" + self.radio_single_channel.isChecked()*"singleCh"
        # timestamp = datetime.now().strftime("%y%m%d%H%M%S")
        if 'annotations' in self.viewer.layers:
            self.viewer.add_labels(
                data=self.viewer.layers['annotations'].data.copy(),
                name=f'annotations_{image_name}_{data_dim}' #_{timestamp}'
                )
        if 'segmentation' in self.viewer.layers:
            self.viewer.add_labels(
                data=self.viewer.layers['segmentation'].data.copy(),
                name=f'segmentation_{image_name}_{data_dim}' #_{timestamp}'
                )

    def _reset_radio_data_dims(self):
        """set radio buttons depending on selected image type"""
        if self.image_layer_selection_widget.value is None:
            for x in self.channel_buttons: x.setEnabled(False)
            return
        
        elif self.image_layer_selection_widget.value.rgb:
            self.radio_rgb.setChecked(True)
            self.radio_multi_channel.setEnabled(False)
            self.radio_single_channel.setEnabled(False)
        elif self.image_layer_selection_widget.value.ndim == 2:
            self.radio_single_channel.setChecked(True)
            self.radio_multi_channel.setEnabled(False)
            self.radio_rgb.setEnabled(False)
        elif self.image_layer_selection_widget.value.ndim == 3:
            self.radio_rgb.setEnabled(False)
            self.radio_multi_channel.setEnabled(True)
            self.radio_single_channel.setEnabled(True)
            self.radio_single_channel.setChecked(True)
        elif self.image_layer_selection_widget.value.ndim == 4:
            self.radio_rgb.setEnabled(False)
            self.radio_multi_channel.setEnabled(True)
            self.radio_single_channel.setEnabled(False)
            self.radio_multi_channel.setChecked(True)

    def _reset_radio_norm_settings(self, event=None):
        
        self.image_mean, self.image_std = None, None

        if self.image_layer_selection_widget.value is None:
            for x in self.norm_buttons: x.setEnabled(False)
            return
        
        if self.image_layer_selection_widget.value.ndim == 2:
            self.radio_normalize_by_image.setEnabled(True)
            self.radio_normalize_by_image.setChecked(True)
            self.radio_normalized_over_stack.setEnabled(False)
        elif (self.image_layer_selection_widget.value.ndim == 3) and (self.param.multi_channel_img):
            self.radio_normalize_by_image.setEnabled(True)
            self.radio_normalize_by_image.setChecked(True)
            self.radio_normalized_over_stack.setEnabled(False)
        else:
            self.radio_normalize_by_image.setEnabled(True)
            self.radio_normalized_over_stack.setEnabled(True)
            self.radio_normalized_over_stack.setChecked(True)

    def _reset_predict_buttons(self):
        # We need a trained model and an image layer needs to be selected
        if (self.trained) and (self.image_layer_selection_widget.value is not None):
            data_dims = self.get_data_dims()
            self.segment_btn.setEnabled(True)
            is_stacked = data_dims in ['4D', '3D_single', '3D_RGB']
            self.segment_all_btn.setEnabled(is_stacked)
        else:
            self.segment_btn.setEnabled(False)
            self.segment_all_btn.setEnabled(False)

    def _reset_stats(self):
        self.image_mean, self.image_std = None, None

    def _update_fe_params_from_gui(self):
        """Update parameters from GUI."""
        # Model
        self.param.model_name = self.qcombo_model_type.currentText()
        self.param.model_layers = self.get_selected_layer_names()
        self.param.order = self.spin_interpolation_order.value()
        self.param.use_min_features = self.check_use_min_features.isChecked()
        self.param.use_cuda = self.check_use_cuda.isChecked()
        self.param.scalings = self.fe_scaling_factors.currentData()

    def _update_gui_fe_layers_from_model(self):
        """Update GUI based on the current model."""
        if self.model is None:
            self.fe_layer_selection.setEnabled(False)
            self.fe_layer_selection.clear()
        elif isinstance(self.model, Hookmodel):
            self._create_fe_layer_selection_from_model()
            if len(self.model.named_modules) == 1:
                self.fe_layer_selection.setCurrentRow(0)
                self.fe_layer_selection.setEnabled(False)
            else:
                self.fe_layer_selection.setEnabled(True)
                for layer in self.param.model_layers:
                    items = self.fe_layer_selection.findItems(layer, Qt.MatchExactly)
                    for item in items:
                        item.setSelected(True)
        else:
            self.fe_layer_selection.setEnabled(False)
            self.fe_layer_selection.clear()

    def _create_fe_layer_selection_from_model(self):
        """Update list of selectable layers from the model."""
        self.fe_layer_selection.clear()
        self.fe_layer_selection.addItems(self.model.selectable_layer_keys.keys())
 
    def _create_fe_layer_selection_from_temp_model(self, temp_model):
        """Update list of selectable layers using the temp model (for choosing)."""
        self.fe_layer_selection.clear()
        self.fe_layer_selection.addItems(temp_model.selectable_layer_keys.keys())   

    def _update_gui_scalings_from_param(self):
        index = self.fe_scaling_factors.findData(self.param.scalings)
        if index != -1:
            self.fe_scaling_factors.setCurrentIndex(index)
        else:
            self.fe_scaling_factors.addItem(str(self.param.scalings), self.param.scalings)
            self.fe_scaling_factors.setCurrentIndex(self.fe_scaling_factors.count()-1)

    def _update_gui_from_param(self):
        """Update GUI from parameters."""
        # Image processing parameters
        # NOTE: WHAT TO DO ABOUT MULTICHANNEL ?
        self.button_group_normalize.button(self.param.normalize).setChecked(True)
        # Acceleration parameters
        self.spin_downsample.setValue(self.param.image_downsample)
        self.check_tile_annotations.setChecked(self.param.tile_annotations)
        self.check_tile_image.setChecked(self.param.tile_image)
        # Feature Extractor
        self.qcombo_model_type.setCurrentText(self.param.model_name)
        # self.fe_layer_selection.clear()
        self._update_gui_scalings_from_param()
        self.spin_interpolation_order.setValue(self.param.order)
        self.check_use_min_features.setChecked(self.param.use_min_features)
        self.check_use_cuda.setChecked(self.param.use_cuda)
        # Classifier
        self.spin_iterations.setValue(self.param.classif_iterations)
        self.spin_learning_rate.setValue(self.param.classif_learning_rate)
        self.spin_depth.setValue(self.param.classif_depth)

    def _check_create_prediction_layer(self):
        """Check if segmentation layer exists and create it if not."""
        layer_names = [x.name for x in self.viewer.layers]
        if 'segmentation' not in layer_names:
            data_dims = self.get_data_dims()
            # Define the shape of the segmentation layer by the type of image
            if data_dims in ['2D_RGB', '2D']:
                layer_shape = self.viewer.layers[self.selected_channel].data.shape[0:2]
            if data_dims == '3D_RGB':
                layer_shape = self.viewer.layers[self.selected_channel].data.shape[0:3]
            else:
                layer_shape = self.viewer.layers[self.selected_channel].data.shape[-2:]
            self.viewer.add_labels(
                data=np.zeros((layer_shape), dtype=np.uint8), name='segmentation'
                )

    def _reset_classif(self):
        self.classifier = None
        self.current_model_path.setText('not trained')
        self.trained = False
        self.save_model_btn.setEnabled(False)
        self._reset_predict_buttons()
        self._set_model_description()

    def _reset_classif_params(self):
        """Reset classifier parameters to default values."""
        # In the widget
        self.spin_iterations.setValue(self.DEFAULT_ITERATIONS)
        self.spin_learning_rate.setValue(self.DEFAULT_LEARNING_RATE)
        self.spin_depth.setValue(self.DEFAULT_DEPTH)
        # In the param object (not done through bindings if values in the widget are not changed)
        self.param.classif_iterations = self.DEFAULT_ITERATIONS
        self.param.classif_learning_rate = self.DEFAULT_LEARNING_RATE
        self.param.classif_depth = self.DEFAULT_DEPTH

    def _reset_fe_params(self):
        # Reset the gui values for the FE
        self.qcombo_model_type.setCurrentText(self.DEFAULT_FE)
        self.fe_layer_selection.clearSelection()
        self.fe_layer_selection.setCurrentRow(self.DEFAULT_LAYERS_INDEX)
        self.fe_scaling_factors.setCurrentText(self.DEFAULT_SCALINGS_TEXT)
        self.spin_interpolation_order.setValue(self.DEFAULT_INTERPOLATION_ORDER)
        self.check_use_min_features.setChecked(self.DEFAULT_USE_MIN_FEATURES)
        self.check_use_cuda.setChecked(self.DEFAULT_USE_CUDA)
        # Set default values in param object (by mimicking a click on the "Set FE" button)
        self._on_set_fe_model()

    def _set_default_model(self):#, keep_rgb=False):
        """Set default model."""
        self._reset_classif_params()
        self._reset_fe_params()

    def _set_model_description(self):
        if self.model is None:
            descr = 'No model loaded'
            return
        model_name = self.param.model_name if not self.param.model_name is None else 'None'
        num_layers = len(self.param.model_layers) if not self.param.model_layers is None else 0
        num_scalings = len(self.param.scalings) if not self.param.scalings is None else 0
        descr = (model_name +
        f': {num_layers} layer' + ('s' if num_layers > 1 else '') +
        f', {num_scalings} scaling' + ('s' if num_scalings > 1 else '') + 
        f' ({self.current_model_path.text()})')
        self.model_description1.setText(descr)
        self.model_description2.setText(descr)

    def get_selected_layer_names(self):
        """Get names of selected layers."""
        selected_rows = self.fe_layer_selection.selectedItems()
        selected_layers = [x.text() for x in selected_rows]
        return selected_layers

    def get_data_channel_first(self):
        """Get data from selected channel. If RGB, move channel axis to first position."""
        image_stack = self.viewer.layers[self.selected_channel].data
        if self.get_data_dims() in ['2D_RGB', '3D_RGB']:
            image_stack = np.moveaxis(image_stack, -1, 0)
        return image_stack
        
    def get_data_channel_first_norm(self):
        """Get data from selected channel. Output has channel (if present) in 
        first position and is normalized."""
        image_stack = self.get_data_channel_first()
        if self.image_mean is None:
            self.get_image_stats()
        # Normalize image
        if self.param.normalize != 1:
            image_stack = normalize_image(
                image=image_stack,
                image_mean=self.image_mean,
                image_std=self.image_std)
        return image_stack

    def get_image_stats(self):
        # put channels in format (C)(T,Z)YX
        data = self.get_data_channel_first()
        data_dims = self.get_data_dims()

        if self.param.normalize == 2: # normalize over stack
            if data_dims in ["4D", "3D_multi", "2D_RGB", "3D_RGB"]:
                self.image_mean, self.image_std = compute_image_stats(
                    image=data,
                    ignore_n_first_dims=1) # ignore channels dimension, but norm over stack
            else: # 2D or 3D_single
                self.image_mean, self.image_std = compute_image_stats(
                    image=data,
                    ignore_n_first_dims=None) # for 3D_single, norm over stack
        elif self.param.normalize == 3: # normalize by image
            # also takes into account CXY case (3D multi-channel image) as first dim is dropped
            self.image_mean, self.image_std = compute_image_stats(
                image=data,
                ignore_n_first_dims=data.ndim-2) # ignore all but the spatial dimensions

    def get_data_dims(self):
        """Get data dimensions. Returns '4D', '2D', 'RGB', '3D_multi' or '3D_single'."""
        num_dims = self.image_layer_selection_widget.value.ndim
        if (num_dims == 1 or num_dims > 4):
            raise Exception('Image has wrong number of dimensions')
        if num_dims == 4:
            return '4D'
        if num_dims == 2:
            if self.image_layer_selection_widget.value.rgb:
                return '2D_RGB'
            else:
                return '2D'
        else: # 3D
            if self.image_layer_selection_widget.value.rgb:
                return '3D_RGB'
            if self.param.multi_channel_img:
                return '3D_multi'
            else:
                return '3D_single'