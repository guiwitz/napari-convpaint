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
from .conv_paint_param import Param
from .conv_paint_nnlayers import Hookmodel


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
        self.fe_model = None
        self.trained = False
        self.temp_model = None
        self.classifier = None
        self.project_widget = None
        self.selected_channel = None
        self.third_party = third_party
        # Plugin options and attributes
        self.keep_layers = False # Keep old layers when adding new ones
        self.auto_seg = True # Automatically segment after training
        self.old_data_tag = "None" # Tag for the data before layers are added
        self.add_layers_flag = True # Flag to prevent adding layers twice on one trigger
        self.current_model_path = 'not trained' # Path to the current model (if saved)

        ### Define Constants

        # General parameters
        self.DEFAULT_MULTICHANNEL = False
        self.DEFAULT_NORM = 2
        self.DEFAULT_DOWNSAMPLE = 1
        self.DEFAULT_TILE_ANNOTS = True
        self.DEFAULT_TILE_IMG = False

        # Define the default FE model
        self.DEFAULT_FE = 'vgg16'
        self.DEFAULT_LAYERS_INDEX = 0
        self.DEFAULT_SCALINGS_TEXT = '[1,2,4]'
        self.DEFAULT_INTERPOLATION_ORDER = 0
        self.DEFAULT_USE_MIN_FEATURES = False
        self.DEFAULT_USE_CUDA = False

        # Define the default classifier parameters
        self.DEFAULT_ITERATIONS = 50
        self.DEFAULT_LEARNING_RATE = 0.1
        self.DEFAULT_DEPTH = 5

        ### Build the widget

        # Create main layout
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # Create and add tabs
        self.tab_names = ['Home', 'Model options'] #['Home', 'Project', 'Model options']
        self.tabs = TabSet(self.tab_names, tab_layouts=[None, QGridLayout()]) # [None, None, QGridLayout()])
        self.main_layout.addWidget(self.tabs)
        # Disable project tab as long as not activated
        # self.tabs.setTabEnabled(self.tabs.tab_names.index('Project'), False)
        
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
        self.layer_selection_group.glayout.addWidget(self.add_layers_btn, 2,0,1,1)
        # Checkbox for keeping old layers
        self.check_keep_layers = QCheckBox('Keep old layers')
        self.check_keep_layers.setToolTip('Keep old annotation and segmentation layers when creating new ones.')
        self.check_keep_layers.setChecked(self.keep_layers)
        self.layer_selection_group.glayout.addWidget(self.check_keep_layers, 2,1,1,1)

        # Add buttons for "Train" group
        self.train_classifier_btn = QPushButton('Train')
        self.train_classifier_btn.setToolTip('Train model on annotations')
        self.train_group.glayout.addWidget(self.train_classifier_btn, 0,0,1,1)
        self.check_auto_seg = QCheckBox('Auto segment')
        self.check_auto_seg.setToolTip('Automatically segment image after training')
        self.check_auto_seg.setChecked(self.auto_seg)
        self.train_group.glayout.addWidget(self.check_auto_seg, 0,1,1,1)
        # Project checkbox
        # self.check_use_project = QCheckBox('Project mode (multiple files)')
        # self.check_use_project.setToolTip('Activate Project tab to use multiple files for training the classifier')
        # self.check_use_project.setChecked(False)
        # self.train_group.glayout.addWidget(self.check_use_project, 1,0,1,1)
        # Project button
        # self.train_classifier_on_project_btn = QPushButton('Train on Project')
        # self.train_classifier_on_project_btn.setToolTip('Train on all images loaded in Project tab')
        # self.train_group.glayout.addWidget(self.train_classifier_on_project_btn, 1,1,1,1)
        # if init_project is False:
        #     self.train_classifier_on_project_btn.setEnabled(False)

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
        # Create a label to act as a vertical divider
        # for row in [0,1,2]:
        divider_label = QLabel('¦\n¦\n¦\n¦')
        divider_label.setAlignment(Qt.AlignCenter)
        divider_label.setStyleSheet("font-size: 13px; color: rgba(120, 120, 120, 35%);")
        # Add the divider to the layout
        self.image_processing_group.glayout.addWidget(divider_label, 0, 1, 3, 1)
        # "Normalize" radio buttons
        self.button_group_normalize = QButtonGroup()
        self.radio_no_normalize = QRadioButton('No normalization')
        self.radio_no_normalize.setToolTip('No normalization is applied')
        self.radio_normalize_over_stack = QRadioButton('Normalize over stack')
        self.radio_normalize_over_stack.setToolTip('Normalize over complete z or t stack')
        self.radio_normalize_by_image = QRadioButton('Normalized by plane')
        self.radio_normalize_by_image.setToolTip('Normalize each plane individually')
        self.radio_normalize_over_stack.setChecked(True)
        self.norm_buttons = [self.radio_no_normalize, self.radio_normalize_over_stack, self.radio_normalize_by_image]
        self.button_group_normalize.addButton(self.radio_no_normalize, id=1)
        self.button_group_normalize.addButton(self.radio_normalize_over_stack, id=2)
        self.button_group_normalize.addButton(self.radio_normalize_by_image, id=3)
        self.image_processing_group.glayout.addWidget(self.radio_no_normalize, 0,2,1,1)
        self.image_processing_group.glayout.addWidget(self.radio_normalize_over_stack, 1,2,1,1)
        self.image_processing_group.glayout.addWidget(self.radio_normalize_by_image, 2,2,1,1)

        # Add buttons for "Model" group
        # Current model description label
        self.model_description1 = QLabel('None')
        self.model_group.glayout.addWidget(self.model_description1, 0,0,1,2)
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
        self.qcombo_fe_type = QComboBox()
        self.qcombo_fe_type.addItems(sorted(get_all_models().keys()))
        self.qcombo_fe_type.setToolTip('Select architecture of feature extraction model.')
        self.fe_group.glayout.addWidget(self.qcombo_fe_type, 2, 0, 1, 2)

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
        self.fe_group.glayout.addWidget(QLabel('Pyramid downscaling factors'), 4, 0, 1, 1)
        self.fe_group.glayout.addWidget(self.fe_scaling_factors, 4, 1, 1, 1)

        # Add interpolation order spinbox to FE group
        self.spin_interpolation_order = QSpinBox()
        self.spin_interpolation_order.setMinimum(0)
        self.spin_interpolation_order.setMaximum(5)
        self.spin_interpolation_order.setValue(1)
        self.spin_interpolation_order.setToolTip('Interpolation order for image rescaling')
        self.fe_group.glayout.addWidget(QLabel('Pyramid interpolation order'), 5, 0, 1, 1)
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
        self.reset_default_fe_btn = QPushButton('Reset to default')
        self.reset_default_fe_btn.setToolTip('Set the feature extractor back to the default model')
        self.fe_group.glayout.addWidget(self.reset_default_fe_btn, 7, 0, 1, 1)
        self.set_fe_btn = QPushButton('Set feature extractor')
        self.set_fe_btn.setToolTip('Set the feature extraction model')
        self.fe_group.glayout.addWidget(self.set_fe_btn, 7, 1, 1, 1)

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

        self.set_default_clf_btn = QPushButton('Reset to defaults')
        self.set_default_clf_btn.setEnabled(True)
        self.classifier_params_group.glayout.addWidget(self.set_default_clf_btn, 3, 0, 1, 1)

        # === === ===

        # If project mode is initially activated, add project tab and widget
        # if init_project is True:
        #     self._on_use_project()

        # Add connections and initialize by setting default model and params
        self.add_connections()
        self._reset_default_general_params()
        self._reset_clf_params()
        self._reset_fe_params()
        self._update_gui_from_param()
        self._on_select_layer()

        # Add key bindings
        self.viewer.bind_key('a', self.toggle_annotation, overwrite=True)
        self.viewer.bind_key('r', self.toggle_prediction, overwrite=True)

    def add_connections(self):

        # Reset layer choices in dropdown when layers are renamed; also bind this behaviour to inserted layers
        for layer in self.viewer.layers:
            layer.events.name.connect(self.image_layer_selection_widget.reset_choices)
        self.viewer.layers.events.inserted.connect(self._on_insert_layer)

        # === HOME TAB ===

        # Reset layer choices in dropdowns when napari layers are added or removed
        for layer_widget_reset in [self.annotation_layer_selection_widget.reset_choices,
                                   self.image_layer_selection_widget.reset_choices]:
            self.viewer.layers.events.inserted.connect(layer_widget_reset)
            self.viewer.layers.events.removed.connect(layer_widget_reset)
        
        # Change layer selection when choosing a new image layer in dropdown
        self.image_layer_selection_widget.changed.connect(self._on_select_layer)
        self.annotation_layer_selection_widget.changed.connect(self._on_select_annot)
        self.add_layers_btn.clicked.connect(self._on_add_annot_seg_layers)
        self.check_keep_layers.stateChanged.connect(lambda: setattr(
            self, 'keep_layers', self.check_keep_layers.isChecked()))

        # Train
        self.train_classifier_btn.clicked.connect(self._on_train)
        self.check_auto_seg.stateChanged.connect(lambda: setattr(
            self, 'auto_seg', self.check_auto_seg.isChecked()))
        # self.check_use_project.stateChanged.connect(self._on_use_project)
        # self.train_classifier_on_project_btn.clicked.connect(self._on_train_on_project)

        # Segment
        self.segment_btn.clicked.connect(self._on_predict)
        self.segment_all_btn.clicked.connect(self._on_predict_all)

        # Load/Save
        self.save_model_btn.clicked.connect(self._on_save_model)
        self.load_model_btn.clicked.connect(self._on_load_model)

        # Image Processing; only trigger from buttons that are activated (checked)
        self.radio_single_channel.toggled.connect(lambda checked: 
                                                  checked and self._on_data_dim_changed())
        self.radio_multi_channel.toggled.connect(lambda checked:
                                                 checked and self._on_data_dim_changed())
        self.radio_rgb.toggled.connect(lambda checked:
                                       checked and self._on_data_dim_changed())
        self.radio_no_normalize.toggled.connect(lambda checked:
                                                checked and self._on_norm_changed())
        self.radio_normalize_over_stack.toggled.connect(lambda checked:
                                                         checked and self._on_norm_changed())
        self.radio_normalize_by_image.toggled.connect(lambda checked:
                                                      checked and self._on_norm_changed())

        # Model
        self._reset_model_btn.clicked.connect(self._on_reset_model)

        # Acceleration
        self.spin_downsample.valueChanged.connect(lambda: setattr(
            self.param, 'image_downsample', self.spin_downsample.value()))
        self.check_tile_annotations.stateChanged.connect(lambda: setattr(
            self.param, 'tile_annotations', self.check_tile_annotations.isChecked()))
        self.check_tile_image.stateChanged.connect(lambda: setattr(
            self.param, 'tile_image', self.check_tile_image.isChecked()))

        # === MODEL OPTIONS TAB ===

        # Feature extractor
        self.qcombo_fe_type.currentIndexChanged.connect(self._on_fe_selected)
        self.set_fe_btn.clicked.connect(self._on_set_fe_model)
        self.reset_default_fe_btn.clicked.connect(self._on_reset_default_fe)
        self.fe_layer_selection.itemSelectionChanged.connect(self._on_fe_layer_selection_changed)
        self.fe_scaling_factors.currentIndexChanged.connect(self._on_fe_scalings_changed)
        # NOTE: Changing interpolation_order, use_min_features and use_cuda of FE
        # shall only be applied when the user clicks the button to set the FE
        
        # Classifier
        self.spin_iterations.valueChanged.connect(lambda: setattr(self.param, 'clf_iterations', self.spin_iterations.value()))
        self.spin_learning_rate.valueChanged.connect(lambda: setattr(self.param, 'clf_learning_rate', self.spin_learning_rate.value()))
        self.spin_depth.valueChanged.connect(lambda: setattr(self.param, 'clf_depth', self.spin_depth.value()))
        self.set_default_clf_btn.clicked.connect(self._on_reset_clf_params)

### Visibility toggles for key bindings

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
            self.add_layers_flag = False # Turn off layer creation, since we do it below
            self._get_image_stats()
            # Set radio buttons depending on selected image type
            self._reset_radio_data_dim_choices()
            self._reset_radio_norm_choices()
            self._reset_predict_buttons()
            # Add empty layers and save old data tag for next addition
            self._add_empty_layers()
            self._set_old_data_tag()
            # Activate button to add annotation and segmentation layers
            self.add_layers_btn.setEnabled(True)
        else:
            self.add_layers_btn.setEnabled(False)
        # Allow other methods again to add layers
        self.add_layers_flag = True

    def _on_select_annot(self, newtext=None):
        """Check if annotation layer dimensions are compatible with image, and raise warning if not."""
        if self.image_layer_selection_widget.value is not None:
            if self._check_annotation_layer_shape() is False:
                warnings.warn('Annotation layer has wrong shape for the selected data')

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
        # Check if annotations layer has correct shape for the chosen data type
        if self._check_annotation_layer_shape() is False:
            raise Exception('Annotation layer has wrong shape for the chosen data')
        # Check if model is loaded
        if self.fe_model is None:
            raise Exception('You have to define and load a model first')
        
        # Set the current model path to 'in training' and adjsut the model description
        self.current_model_path= 'in training'
        self._set_model_description()

        # Start training
        image_stack_norm = self._get_data_channel_first_norm()
        
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(True)

        with progress(total=0) as pbr:
            pbr.set_description(f"Training")
            features, targets = get_features_current_layers(
                image=image_stack_norm,
                annotations=self.annotation_layer_selection_widget.value.data,
                model=self.fe_model,
                param=self.param,
            )
            self.classifier = train_classifier(
                features, targets,
                iterations=self.param.clf_iterations,
                learning_rate=self.param.clf_learning_rate,
                depth=self.param.clf_depth,
        )
    
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(False)

        # Set the current model path to 'trained, unsaved' and adjust the model description
        self.current_model_path = 'trained, unsaved'
        self.trained = True
        self._reset_predict_buttons()
        self.save_model_btn.setEnabled(True)
        self._set_model_description()
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

        if self.fe_model is None:
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
            self.current_model_path = 'in training'
            all_features, all_targets = [], []
            for ind in range(num_files):
                self.project_widget.file_list.setCurrentRow(ind)

                image_stack = self._get_data_channel_first_norm()

                features, targets = get_features_current_layers(
                    model=self.fe_model,
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

            self.current_model_path = 'trained, unsaved'
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
        # Perform checks as preparation for prediction
        if self.fe_model is None:
            raise Exception('You have to define and load a model first')
        
        if self.classifier is None:
            self._on_train()

        self._check_create_prediction_layer()

        if self.image_mean is None:
            self._get_image_stats()

        # Start prediction
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(True)

        with progress(total=0) as pbr:
            
            pbr.set_description(f"Prediction")

            data_dims = self._get_data_dims()

            # Get image data and stats depending on the data dim and norm radio buttons
            if data_dims in ['2D', '2D_RGB', '3D_multi']:
                image = self._get_data_channel_first()
                if self.param.normalize != 1:
                    image_mean = self.image_mean
                    image_std = self.image_std

            if data_dims == '3D_single':
                step = self.viewer.dims.current_step[0]
                image = self.viewer.layers[self.selected_channel].data[step]
                if self.param.normalize == 2: # over stack (only 1 value in stack dim)
                    image_mean = self.image_mean
                    image_std = self.image_std
                if self.param.normalize == 3: # by image (use values for current step)
                    image_mean = self.image_mean[step]
                    image_std = self.image_std[step]
            
            if data_dims == '3D_RGB':
                step = self.viewer.dims.current_step[0]
                image = self.viewer.layers[self.selected_channel].data[step]
                image = np.moveaxis(image, -1, 0)
                if self.param.normalize == 2: # over stack (only 1 value in stack dim)
                    image_mean = self.image_mean[:,0,::]
                    image_std = self.image_std[:,0,::]
                if self.param.normalize == 3: # by image (use values for current step)
                    image_mean = self.image_mean[:,step]
                    image_std = self.image_std[:,step]

            if data_dims == '4D':
                # for 4D, channel is always first, t/z second
                step = self.viewer.dims.current_step[1]
                image = self.viewer.layers[self.selected_channel].data[:, step]
                if self.param.normalize == 2: # over stack (only 1 value in stack dim)
                    image_mean = self.image_mean[:,0,::]
                    image_std = self.image_std[:,0,::]
                if self.param.normalize == 3: # by image (use values for current step)
                    image_mean = self.image_mean[:,step]
                    image_std = self.image_std[:,step]
            
            # Normalize image (using the stats based on the radio buttons)
            if self.param.normalize != 1:
                image = normalize_image(image=image, image_mean=image_mean, image_std=image_std)

            # Predict image
            if self.param.tile_image:
                predicted_image = parallel_predict_image(
                    image=image, model=self.fe_model, classifier=self.classifier,
                    param = self.param, use_dask=False)
            else:
                predicted_image = self.fe_model.predict_image(image=image,
                                                           classifier=self.classifier,
                                                           param=self.param,)
            
            # Update segmentation layer
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
        
        data_dims = self._get_data_dims()
        if data_dims not in ['3D_single', '3D_RGB', '4D']:
            raise Exception(f'Image stack has wrong dimensionality {data_dims}')

        if self.fe_model is None:
            raise Exception('You have to define and load a model first')

        if self.classifier is None:
            self._on_train()
                    
        self._check_create_prediction_layer()

        if self.image_mean is None:
            self._get_image_stats()

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(True)

        # Get normalized stack data (entire stack, and stats prepared given the radio buttons)
        image_stack_norm = self._get_data_channel_first_norm() # Normalize the entire stack
        
        # Step through the stack and predict each image
        num_steps = image_stack_norm.shape[-3]
        for step in progress(range(num_steps)):
            if data_dims == '3D_single':
                image = image_stack_norm[step]
            elif data_dims in ['3D_RGB', '4D']:
                image = image_stack_norm[:, step]

            # Predict the current step            
            if self.param.tile_image:
                predicted_image = parallel_predict_image(
                    image=image, model=self.fe_model, classifier=self.classifier,
                    param = self.param, use_dask=False)
            else:
                predicted_image = self.fe_model.predict_image(image=image,
                                                           classifier=self.classifier,
                                                           param=self.param)
            # Update segmentation layer
            self.viewer.layers['segmentation'].data[step] = predicted_image
        self.viewer.layers['segmentation'].refresh()

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(False)

    # Load/Save

    def _on_save_model(self, event=None, save_file=None):
        """Select file where to save the classifier along with the model parameters."""
        if self.classifier is None:
            raise Exception('No trained classifier found. Please train a model first.')
        # Get file path
        if save_file is None:
            dialog = QFileDialog()
            save_file, _ = dialog.getSaveFileName(self, "Save model", None, "PICKLE (*.pickle)")
        save_file = Path(save_file)
        # Save model
        save_model(save_file, self.classifier, self.fe_model, self.param, )
        # Adjust the model description
        self.current_model_path = save_file.name
        self._set_model_description()

    def _on_load_model(self, event=None, save_file=None):
        """Select classifier file to load along with the model parameters."""
        # Get file path
        if save_file is None:
            dialog = QFileDialog()
            save_file, _ = dialog.getOpenFileName(self, "Choose model", None, "PICKLE (*.pickle)")
        save_file = Path(save_file)
        classifier, model, param, model_state = load_model(save_file)

        # Load params and update GUI from them
        self.param = param
        # Check if the loaded multichannel setting is compatible with data
        data_dims = self._get_data_dims()
        if data_dims in ['2D', '2D_RGB', '3D_RGB'] and self.param.multi_channel_img:
            warnings.warn(f'The loaded model works with multichannel, but the data is {data_dims}.' +
                            'This might cause problems.')
        # Check if the loaded normalization setting is compatible with data
        if data_dims in ['2D', '2D_RGB', '3D_multi'] and self.param.normalize == 2:
            warnings.warn(f'The loaded model normalizes over stack, but the data is {data_dims}.' +
                            'This might cause problems.')
        # Update GUI from the loaded parameters
        self._update_gui_from_param()
        
        # Load model and model state if available
        self.fe_model = model
        if model_state is not None and hasattr(self.fe_model, 'load_state_dict'):
            self.fe_model.load_state_dict(model_state)
        # Update GUI to show selectable layers of model chosen from drop-down
        self._update_gui_fe_layers_from_model()
        
        # Load classifier; do this after GUI updates so they do not reset the classifier
        self.classifier = classifier
        
        # Adjust trained flag, save button and predict buttons, and update model description
        self.trained = True
        self.save_model_btn.setEnabled(True)
        self._reset_predict_buttons()
        self.current_model_path = save_file.name
        self._set_model_description()

    # Image Processing

    def _on_data_dim_changed(self):
        """Set the image data dimensions based on radio buttons,
        reset classifier, and adjust normalization options."""
        self.param.multi_channel_img = self.radio_multi_channel.isChecked()
        self._reset_clf()
        self._reset_radio_norm_choices()
        if self.add_layers_flag: # Add layers only if not triggered from layer selection
            self._add_empty_layers()
            self._set_old_data_tag()

    def _on_norm_changed(self):
        """Set the normalization options based on radio buttons,
        and update stats."""
        if self.radio_no_normalize.isChecked():
            self.param.normalize = 1
        elif self.radio_normalize_over_stack.isChecked():
            self.param.normalize = 2
        elif self.radio_normalize_by_image.isChecked():
            self.param.normalize = 3
        # self._get_image_stats() # NOTE: Don't we want to set it right away?
        self.image_mean, self.image_std = None, None
        self._reset_clf()

    # Model

    def _on_reset_model(self, event=None):
        """Reset model to default and update GUI."""
        self.save_model_btn.setEnabled(False)
        self._reset_clf_params()
        self._reset_fe_params()
        self._reset_clf()

    ### Model Tab

    def _on_fe_selected(self, index):
        """Update GUI to show selectable layers of model chosen from drop-down."""
        # Create a temporary model to get the layers (to display) and default parameters
        fe_type = self.qcombo_fe_type.currentText()
        model_class = get_all_models()[fe_type]
        self.temp_model = model_class(model_name=fe_type, use_cuda=self.param.fe_use_cuda)
        # Update the GUI to show the FE layers of the temp model
        self._update_gui_fe_layer_choice_from_temp_model()
        # Get the default FE params for the temp model and update the GUI
        default_param = self.temp_model.get_default_param()
        self._update_gui_fe_scalings(default_param.fe_scalings)
        val_to_setter = {
            "fe_name": self.qcombo_fe_type.setCurrentText,
            "fe_order": self.spin_interpolation_order.setValue,
            "fe_use_min_features": self.check_use_min_features.setChecked,
            "fe_use_cuda": self.check_use_cuda.setChecked
        }
        for attr, setter in val_to_setter.items():
            val = getattr(default_param, attr, None)
            if val is not None:
                if isinstance(val, list): val = str(val)
                setter(val)

    def _on_fe_layer_selection_changed(self):
        """Enable the set button based on the model type."""
        if isinstance(self.temp_model, Hookmodel):
            selected_layers = self._get_selected_layer_names()
            if len(selected_layers) == 0:
                self.set_fe_btn.setEnabled(False)
            else:
                self.set_fe_btn.setEnabled(True)
        else:
            self.set_fe_btn.setEnabled(True)

    def _on_fe_scalings_changed(self):
        """Update param object only when FE is set."""
        return

    def _on_set_fe_model(self, event=None):
        """Create a neural network model that will be used for feature extraction and
        reset the classifier."""
        # Update FE parameters from the GUI
        self.param.fe_name = self.qcombo_fe_type.currentText()
        self.param.fe_layers = self._get_selected_layer_names()
        self.param.fe_scalings = self.fe_scaling_factors.currentData()
        self.param.fe_order = self.spin_interpolation_order.value()
        self.param.fe_use_min_features = self.check_use_min_features.isChecked()
        self.param.fe_use_cuda = self.check_use_cuda.isChecked()

        # Get default non-FE params from temp model and update the GUI (also setting the params)
        default_param = self.temp_model.get_default_param()
        enforced_params = [] # List of enforced parameters for raising a warning
        # Multichannel
        if ((default_param.multi_channel_img is not None) and
            (self.param.multi_channel_img != default_param.multi_channel_img)):
            data_dims = self._get_data_dims()
            # Catch case where multichannel is enforced on incompatible data
            if data_dims in ['2D', '2D_RGB', '3D_RGB'] and default_param.multi_channel_img:
                warnings.warn(f'The feature extractor tried to enforce multichannel on {data_dims} data.' +
                              'This is not supported and will be ignored.')
            else: # If data is compatible, enforce the model's default multichannel setting
                enforced_params.append('multi_channel_img')
                if default_param.multi_channel_img: # If the default model is multi-channel, enforce it
                    self.param.multi_channel_img = True
                    self.radio_multi_channel.setChecked(True)
                else: # If the default model is non-multichannel, reset data dims according to data
                    self.param.multi_channel_img = False
                    self._reset_radio_data_dim_choices()
                self._reset_radio_norm_choices() # Update norm options, since multi_channel_img changed
        # Normalization
        if ((default_param.normalize is not None) and
            (self.param.normalize != default_param.normalize)):
            data_dims = self._get_data_dims()
            if data_dims in ['2D', '2D_RGB', '3D_multi'] and default_param.normalize == 2:
                warnings.warn(f'The feature extractor tried to enforce normalization over stack on {data_dims} data.' +
                              'This is not supported and will be ignored.')
            else:
                enforced_params.append('normalize')
                self.button_group_normalize.button(default_param.normalize).setChecked(True)
        # Other params
        val_to_setter = {
            "image_downsample": self.spin_downsample.setValue,
            "tile_annotations": self.check_tile_annotations.setChecked,
            "tile_image": self.check_tile_image.setChecked,
            "clf_iterations": self.spin_iterations.setValue,
            "clf_learning_rate": self.spin_learning_rate.setValue,
            "clf_depth": self.spin_depth.setValue
        }
        for attr, setter in val_to_setter.items():
            val = getattr(default_param, attr, None)
            if val is not None and val != getattr(self.param, attr):
                enforced_params.append(attr)
                if isinstance(val, list):
                    val = str(val)
                setter(val)
        if enforced_params: warnings.warn(f'The feature extractor enforced the parameters {enforced_params}')

        # Create the model and reset the classifier
        self.fe_model = create_model(self.param)
        self._reset_clf()

    def _on_reset_default_fe(self, event=None):
        """Reset the feature extraction model to the default model."""
        self._reset_fe_params()

    def _on_reset_clf_params(self):
        """Reset the classifier parameters to the default values
        and discard the trained model."""
        self._reset_clf_params()
        self._reset_clf()
    
### Helper functions

    def _add_empty_layers(self, event=None, force_add=True):
        """Add annotation and prediction layers to viewer. If the layer already exists,
        remove it and add a new one. If the widget is used as third party (self.third_party=True),
        no layer is added if it didn't exist before, unless force_add=True (e.g. when the user click
        on the add layer button)"""

        if self.keep_layers:
            self._create_annot_seg_copies()

        if self.image_layer_selection_widget.value is None:
            raise Exception('Please select an image layer first')
        
        layer_shape = self._get_annot_shape()

        # Add segmentation layer; do this first, so that the annotation layer is on top
        segmentation_exists = 'segmentation' in self.viewer.layers
        if segmentation_exists:
            self.viewer.layers.remove('segmentation')

        if (not self.third_party) | (self.third_party & segmentation_exists) | (force_add):
            self.viewer.add_labels(
                data=np.zeros((layer_shape), dtype=np.uint8),
                name='segmentation'
                )
    
        # Add annotation layer
        annotation_exists = 'annotations' in self.viewer.layers
        if annotation_exists:
            self.viewer.layers.remove('annotations')

        if (not self.third_party) | (self.third_party & annotation_exists) | (force_add):
            self.viewer.add_labels(
                data=np.zeros((layer_shape), dtype=np.uint8),
                name='annotations'
                )
        
        # Activate the annotation layer, select it in the dropdown and activate paint mode
        if 'annotations' in self.viewer.layers:
            self.viewer.layers.selection.active = self.viewer.layers['annotations']
            self.annotation_layer_selection_widget.value = self.viewer.layers['annotations']
            self.viewer.layers['annotations'].mode = 'paint'
            self.viewer.layers['annotations'].brush_size = 3

    def _create_annot_seg_copies(self):
        """Create copies of the annotations and segmentation layers."""
        if 'annotations' in self.viewer.layers:
            self.viewer.add_labels(
                data=self.viewer.layers['annotations'].data.copy(),
                name=f'annotations_{self.old_data_tag}'
                )
        if 'segmentation' in self.viewer.layers:
            self.viewer.add_labels(
                data=self.viewer.layers['segmentation'].data.copy(),
                name=f'segmentation_{self.old_data_tag}'
                )

    def _reset_radio_data_dim_choices(self):
        """Set radio buttons active/inactive depending on selected image type.
        If current parameter is not valid, set a default."""
        if self.image_layer_selection_widget.value is None:
            for x in self.channel_buttons: x.setEnabled(False)
            return
        
        if self.image_layer_selection_widget.value.rgb: # If the image is RGB, this is the only option
            self.radio_single_channel.setEnabled(False)
            self.radio_multi_channel.setEnabled(False)
            self.radio_rgb.setEnabled(True)
            self.radio_rgb.setChecked(True)
        elif self.image_layer_selection_widget.value.ndim == 2: # If non-rgb 2D, only single channel
            self.radio_single_channel.setEnabled(True)
            self.radio_multi_channel.setEnabled(False)
            self.radio_rgb.setEnabled(False)
            self.radio_single_channel.setChecked(True)
        elif self.image_layer_selection_widget.value.ndim == 3: # If 3D, single or multi channel
            self.radio_single_channel.setEnabled(True)
            self.radio_multi_channel.setEnabled(True)
            self.radio_rgb.setEnabled(False)
            # multi_channel_val = self.param.multi_channel_img is not None and self.param.multi_channel_img
            self.radio_single_channel.setChecked(not self.param.multi_channel_img) # Choice depending on param
            self.radio_multi_channel.setChecked(self.param.multi_channel_img)
        elif self.image_layer_selection_widget.value.ndim == 4: # If 4D, it must be multi channel
            self.radio_single_channel.setEnabled(False)
            self.radio_multi_channel.setEnabled(True)
            self.radio_rgb.setEnabled(False)
            self.radio_multi_channel.setChecked(True)

    def _reset_radio_norm_choices(self, event=None):
        """Set radio buttons active/inactive depending on selected image type.
        If current parameter is not valid, set a default."""
        # Reset the stats; not necessary, since they are recalculated anyway if changing the norm mode
        # self.image_mean, self.image_std = None, None

        if self.image_layer_selection_widget.value is None:
            for x in self.norm_buttons: x.setEnabled(False)
            return
        
        data_dims = self._get_data_dims()
        
        if data_dims in ['2D', '2D_RGB', '3D_multi']:
            self.radio_no_normalize.setEnabled(True)
            self.radio_normalize_over_stack.setEnabled(False)
            self.radio_normalize_by_image.setEnabled(True)
            if self.param.normalize == 2: # If initially over stack, reset to by image
                self.radio_normalize_by_image.setChecked(True)
            else:
                self.button_group_normalize.button(self.param.normalize).setChecked(True)
        else: # 3D_single, 4D, 3D_RGB
            self.radio_no_normalize.setEnabled(True)
            self.radio_normalize_over_stack.setEnabled(True)
            self.radio_normalize_by_image.setEnabled(True)
            self.button_group_normalize.button(self.param.normalize).setChecked(True)

    def _reset_predict_buttons(self):
        """Enable or disable predict buttons based on the current state."""
        # We need a trained model and an image layer needs to be selected
        if self.trained and self.image_layer_selection_widget.value is not None:
            data_dims = self._get_data_dims()
            self.segment_btn.setEnabled(True)
            is_stacked = data_dims in ['4D', '3D_single', '3D_RGB']
            self.segment_all_btn.setEnabled(is_stacked)
        else:
            self.segment_btn.setEnabled(False)
            self.segment_all_btn.setEnabled(False)

    def _update_gui_fe_layers_from_model(self):
        """Update GUI FE layer selection based on the current (e.g. loaded) model."""
        if self.fe_model is None:
            self.fe_layer_selection.setEnabled(False)
            self.fe_layer_selection.clear()
            return
        
        if isinstance(self.fe_model, Hookmodel):
            self.fe_layer_selection.clear()
            layer_keys = self.fe_model.selectable_layer_keys
            self.fe_layer_selection.addItems(layer_keys)
            self.fe_layer_selection.setEnabled(len(layer_keys) > 1) # Only need to enable if multiple layers available
            for layer in self.param.fe_layers:
                items = self.fe_layer_selection.findItems(layer, Qt.MatchExactly)
                for item in items:
                    item.setSelected(True)
        else: # For non-hookmodels, disable the selection
            self.fe_layer_selection.clear()
            self.fe_layer_selection.setEnabled(False)
        self.set_fe_btn.setEnabled(True)

    def _update_gui_fe_layer_choice_from_temp_model(self):
        """Update GUI selectable FE layers based on the temporary model."""
        # Get selectable layers from the temp model and update the GUI
        self.set_fe_btn.setEnabled(False)
        if isinstance(self.temp_model, Hookmodel):
            self.fe_layer_selection.clear()
            layer_keys = self.temp_model.selectable_layer_keys
            self.fe_layer_selection.addItems(layer_keys)   
            if len(layer_keys) == 1: # If only one, we can auto-select
                self.fe_layer_selection.setCurrentRow(0)
                self.fe_layer_selection.setEnabled(False)
                self.set_fe_btn.setEnabled(True)
            else: # If multiple layers available, enable the selection
                self.fe_layer_selection.setEnabled(True)
        else: # For non-hookmodels, disable the selection, and enable the set button
            self.fe_layer_selection.clear()
            self.fe_layer_selection.setEnabled(False)
            self.set_fe_btn.setEnabled(True)

    def _update_gui_fe_scalings(self, fe_scalings=None):
        """Update GUI FE scalings (e.g. from the temporary model or param object)."""
        if fe_scalings is None:
            return
        # Find index of the input value (-1 if not found)
        index = self.fe_scaling_factors.findData(fe_scalings)
        if index != -1:
            self.fe_scaling_factors.setCurrentIndex(index)
        # If not in list yet, add and select it
        else:
            self.fe_scaling_factors.addItem(str(fe_scalings), fe_scalings)
            self.fe_scaling_factors.setCurrentIndex(self.fe_scaling_factors.count()-1)

    def _update_gui_from_param(self):
        """Update GUI from parameters."""

        self._reset_radio_data_dim_choices()
        # Set radio buttons depending on param (possibly enforcing a choice)
        if self.param.multi_channel_img:
            self.radio_single_channel.setChecked(False)
            self.radio_multi_channel.setChecked(True)
            self.radio_rgb.setChecked(False)
        elif (self.image_layer_selection_widget.value is not None and 
              self.image_layer_selection_widget.value.rgb):
            self.radio_single_channel.setChecked(False)
            self.radio_multi_channel.setChecked(False)
            self.radio_rgb.setChecked(True)
        else:
            self.radio_single_channel.setChecked(True)
            self.radio_multi_channel.setChecked(False)
            self.radio_rgb.setChecked(False)
        self._reset_radio_norm_choices()

        self.button_group_normalize.button(self.param.normalize).setChecked(True)

        val_to_setter = {
            "image_downsample": self.spin_downsample.setValue,
            "tile_annotations": self.check_tile_annotations.setChecked,
            "tile_image": self.check_tile_image.setChecked,
            "fe_name": self.qcombo_fe_type.setCurrentText,
            "fe_order": self.spin_interpolation_order.setValue,
            "fe_use_min_features": self.check_use_min_features.setChecked,
            "fe_use_cuda": self.check_use_cuda.setChecked,
            "clf_iterations": self.spin_iterations.setValue,
            "clf_learning_rate": self.spin_learning_rate.setValue,
            "clf_depth": self.spin_depth.setValue
        }
        for attr, setter in val_to_setter.items():
            val = getattr(self.param, attr, None)
            if val is not None:
                if isinstance(val, list): val = str(val)
                setter(val)

        self._update_gui_fe_scalings(self.param.fe_scalings)

    def _check_create_prediction_layer(self):
        """Check if segmentation layer exists and create it if not."""
        layer_names = [x.name for x in self.viewer.layers]
        if 'segmentation' not in layer_names:
            layer_shape = self._get_annot_shape()
            self.viewer.add_labels(
                data=np.zeros((layer_shape), dtype=np.uint8), name='segmentation'
                )
            
    def _get_annot_shape(self):
        """Get shape of annotations and segmentation layers to create."""
        data_dims = self._get_data_dims()
        if data_dims in ['2D_RGB', '2D']:
            return self.viewer.layers[self.selected_channel].data.shape[0:2]
        if data_dims in ['3D_RGB', '3D_single']:
            return self.viewer.layers[self.selected_channel].data.shape[0:3]
        else: # 3D_multi, 4D
            return self.viewer.layers[self.selected_channel].data.shape[1:]

    def _reset_clf(self):
        """Discard the trained classifier."""
        self.classifier = None
        self.current_model_path = 'not trained'
        self.trained = False
        self.save_model_btn.setEnabled(False)
        self._reset_predict_buttons()
        self._set_model_description()

    def _reset_clf_params(self):
        """Reset classifier parameters to default values."""
        # In the widget, which will also trigger to adjust the param object
        self.spin_iterations.setValue(self.DEFAULT_ITERATIONS)
        self.spin_learning_rate.setValue(self.DEFAULT_LEARNING_RATE)
        self.spin_depth.setValue(self.DEFAULT_DEPTH)
        # In the param object (not done through bindings if values in the widget are not changed)
        self.param.clf_iterations = self.DEFAULT_ITERATIONS
        self.param.clf_learning_rate = self.DEFAULT_LEARNING_RATE
        self.param.clf_depth = self.DEFAULT_DEPTH

    def _reset_fe_params(self):
        """Reset feature extraction parameters to default values."""
        # Reset the gui values for the FE, which will also trigger to adjust the param object
        self.qcombo_fe_type.setCurrentText(self.DEFAULT_FE)
        self.fe_layer_selection.clearSelection()
        self.fe_layer_selection.setCurrentRow(self.DEFAULT_LAYERS_INDEX)
        self.fe_scaling_factors.setCurrentText(self.DEFAULT_SCALINGS_TEXT)
        self.spin_interpolation_order.setValue(self.DEFAULT_INTERPOLATION_ORDER)
        self.check_use_min_features.setChecked(self.DEFAULT_USE_MIN_FEATURES)
        self.check_use_cuda.setChecked(self.DEFAULT_USE_CUDA)
        # Set default values in param object (by mimicking a click on the "Set FE" button)
        self._on_set_fe_model()

    def _reset_default_model_params(self):
        """Set model back to default (FE and Classifier)."""


    def _reset_default_general_params(self):
        """Set general parameters back to default."""
        # Set defaults in GUI
        self.radio_single_channel.setChecked(not self.DEFAULT_MULTICHANNEL)
        self.radio_multi_channel.setChecked(self.DEFAULT_MULTICHANNEL)
        self.radio_rgb.setChecked(False)
        self.button_group_normalize.button(self.DEFAULT_NORM).setChecked(True)
        self.spin_downsample.setValue(self.DEFAULT_DOWNSAMPLE)
        self.check_tile_annotations.setChecked(self.DEFAULT_TILE_ANNOTS)
        self.check_tile_image.setChecked(self.DEFAULT_TILE_IMG)
        # Set defaults in param object (not done through bindings if values in the widget are not changed)
        self.param.multi_channel_img = self.DEFAULT_MULTICHANNEL
        self.param.normalize = self.DEFAULT_NORM
        self.param.image_downsample = self.DEFAULT_DOWNSAMPLE
        self.param.tile_annotations = self.DEFAULT_TILE_ANNOTS
        self.param.tile_image = self.DEFAULT_TILE_IMG

    def _set_model_description(self):
        """Set the model description text."""
        if self.fe_model is None:
            descr = 'No model loaded'
            return
        fe_name = self.param.fe_name if not self.param.fe_name is None else 'None'
        num_layers = len(self.param.fe_layers) if not self.param.fe_layers is None else 0
        num_scalings = len(self.param.fe_scalings) if not self.param.fe_scalings is None else 0
        descr = (fe_name +
        f': {num_layers} layer' + ('' if num_layers == 1 else 's') +
        f', {num_scalings} scaling' + ('' if num_scalings == 1 else 's') + 
        f' ({self.current_model_path})')
        self.model_description1.setText(descr)
        self.model_description2.setText(descr)
        
    def _set_old_data_tag(self):
        """Set the old data tag based on the current image layer and data dimensions."""
        image_name = self.image_layer_selection_widget.value.name
        data_dim_str = (self.radio_rgb.isChecked()*"RGB" +
                    self.radio_multi_channel.isChecked()*"multiCh" +
                    self.radio_single_channel.isChecked()*"singleCh")
        # timestamp = datetime.now().strftime("%y%m%d%H%M%S")
        self.old_data_tag = f"{image_name}_{data_dim_str}"

    def _get_selected_layer_names(self):
        """Get names of selected layers."""
        selected_rows = self.fe_layer_selection.selectedItems()
        selected_layers = [x.text() for x in selected_rows]
        return selected_layers

    def _get_data_channel_first(self):
        """Get data from selected channel. If RGB, move channel axis to first position."""
        image_stack = self.viewer.layers[self.selected_channel].data
        if self._get_data_dims() in ['2D_RGB', '3D_RGB']:
            image_stack = np.moveaxis(image_stack, -1, 0)
        return image_stack
        
    def _get_data_channel_first_norm(self):
        """Get data from selected channel. Output has channel (if present) in 
        first position and is normalized."""
        image_stack = self._get_data_channel_first()
        if self.image_mean is None:
            self._get_image_stats()
        # Normalize image
        if self.param.normalize != 1:
            image_stack = normalize_image(
                image=image_stack,
                image_mean=self.image_mean,
                image_std=self.image_std)
        return image_stack

    def _get_image_stats(self):
        """Get image stats depending on the normalization settings and data dimensions."""
        # If no image is selected, set stats to None
        if self.image_layer_selection_widget.value is None:
            self.image_mean, self.image_std = None, None
            return
        
        # put channels in format (C, T/Z, H, W)
        data = self._get_data_channel_first()
        data_dims = self._get_data_dims()

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
                image=data, ignore_n_first_dims=data.ndim-2) # ignore all but the spatial dimensions

    def _get_data_dims(self):
        """Get data dimensions. Returns '2D', '2D_RGB', '3D_RGB', '3D_multi', '3D_single' or '4D'."""
        num_dims = self.image_layer_selection_widget.value.ndim
        if (num_dims == 1 or num_dims > 4):
            raise Exception('Image has wrong number of dimensions')
        if num_dims == 2:
            if self.image_layer_selection_widget.value.rgb:
                return '2D_RGB'
            else:
                return '2D'
        if num_dims == 3:
            if self.image_layer_selection_widget.value.rgb:
                return '3D_RGB'
            if self.param.multi_channel_img:
                return '3D_multi'
            else:
                return '3D_single'
        if num_dims == 4:
            return '4D'
        
    def _check_annotation_layer_shape(self):
        """Check if the annotation layer has the same shape as the image layer."""
        if self.annotation_layer_selection_widget.value is None:
            return False
        if self.image_layer_selection_widget.value is None:
            return False
        if self.annotation_layer_selection_widget.value.data.shape != self._get_annot_shape():
            return False
        return True