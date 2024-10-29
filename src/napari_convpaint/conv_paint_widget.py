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

from .conv_paint import (get_features_current_layers,
                         get_all_models, train_classifier,
                         load_model, create_model, save_model)
from .conv_paint_utils import (parallel_predict_image, 
                               normalize_image, compute_image_stats)
from .conv_parameters import Param
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
        
        self.param = Param()
        self.model = None
        self.temp_model = None
        self.classifier = None
        self.project_widget = None
        self.features_per_layer = None
        self.selected_channel = None
        self.image_mean = None
        self.image_std = None
        self.third_party = third_party
        self.auto_seg = False

        # Create main layout
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # Create and add tabs
        self.tab_names = ['Home', 'Project', 'Model']
        self.tabs = TabSet(self.tab_names, tab_layouts=[None, None, QGridLayout()])
        self.main_layout.addWidget(self.tabs)
        # Disable project tab as long as not activated
        self.tabs.setTabEnabled(self.tabs.tab_names.index('Project'), False) # R: Why only index? If custom is "respnsive", so should be the index
        
        # Align rows in the tabs "Home" and "Model" on top
        self.tabs.widget(self.tabs.tab_names.index('Home')).layout().setAlignment(Qt.AlignTop)
        self.tabs.widget(self.tabs.tab_names.index('Model')).layout().setAlignment(Qt.AlignTop)

        # === HOME TAB ===

        # Create groups
        self.layer_selection_group = VHGroup('Layer selection', orientation='G')
        self.train_group = VHGroup('Train', orientation='G')
        self.segment_group = VHGroup('Segment', orientation='G')
        self.load_save_group = VHGroup('Load/Save', orientation='G')
        self.image_processing_group = VHGroup('Image processing', orientation='G')
        self.model_group = VHGroup('Model', orientation='G')
        self.acceleration_group = VHGroup('Acceleration', orientation='G')
        # Add groups to the tab
        self.tabs.add_named_tab('Home', self.layer_selection_group.gbox)
        self.tabs.add_named_tab('Home', self.train_group.gbox)
        self.tabs.add_named_tab('Home', self.segment_group.gbox)
        self.tabs.add_named_tab('Home', self.load_save_group.gbox)
        self.tabs.add_named_tab('Home', self.image_processing_group.gbox)
        self.tabs.add_named_tab('Home', self.model_group.gbox)
        self.tabs.add_named_tab('Home', self.acceleration_group.gbox)

        # Add elements to "Layer selection" group
        # Image layer (widget for selecting the layer to segment)
        self.select_image_layer_widget = create_widget(annotation=napari.layers.Image, label='Pick image')
        self.select_image_layer_widget.reset_choices()
        # Annotation layer
        self.select_annotation_layer_widget = create_widget(annotation=napari.layers.Labels, label='Pick annotation')
        self.select_annotation_layer_widget.reset_choices()

        # Add widgets to layout
        self.layer_selection_group.glayout.addWidget(QLabel('Image layer'), 0,0,1,1)
        self.layer_selection_group.glayout.addWidget(self.select_image_layer_widget.native, 0,1,1,1)
        self.layer_selection_group.glayout.addWidget(QLabel('Annotation layer'), 1,0,1,1)
        self.layer_selection_group.glayout.addWidget(self.select_annotation_layer_widget.native, 1,1,1,1)
        # Add button for adding annotation/segmentation layers
        self.add_layers_btn = QPushButton('Add annotations/segmentation layers')
        self.add_layers_btn.setEnabled(False)
        self.layer_selection_group.glayout.addWidget(self.add_layers_btn, 2,0,1,2)

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
        #[x.setEnabled(False) for x in [self.radio_no_normalize, self.radio_normalized_over_stack, self.radio_normalize_by_image]]
        self.button_group_normalize.addButton(self.radio_no_normalize, id=1)
        self.button_group_normalize.addButton(self.radio_normalized_over_stack, id=2)
        self.button_group_normalize.addButton(self.radio_normalize_by_image, id=3)
        self.image_processing_group.glayout.addWidget(self.radio_no_normalize, 0,1,1,1)
        self.image_processing_group.glayout.addWidget(self.radio_normalized_over_stack, 1,1,1,1)
        self.image_processing_group.glayout.addWidget(self.radio_normalize_by_image, 2,1,1,1)

        # Add buttons for "Model" group
        # Current model label
        self.model_group.glayout.addWidget(QLabel('Current model:'), 0,0,1,1)
        self.current_model_path = QLabel('None')
        self.model_group.glayout.addWidget(self.current_model_path, 0,1,1,1)
        # "Use custom Model" checkbox
        self.check_use_custom_model = QCheckBox('Use custom Model')
        self.check_use_custom_model.setToolTip('Activate Tab to customize Feature Extractor and Classifier')
        self.check_use_custom_model.setChecked(False)
        self.model_group.glayout.addWidget(self.check_use_custom_model, 1,0,1,1)
        # Reset model button
        self._reset_model_btn = QPushButton('Reset model')
        self._reset_model_btn.setToolTip('Discard current model and annotations, create new default model.')
        self.model_group.glayout.addWidget(self._reset_model_btn, 1,1,1,1)

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

        # Create two groups, 'Feature extractor' and 'CatBoost Classifier parameters'
        self.fe_group = VHGroup('Feature extractor', orientation='G')
        self.classifier_params_group = VHGroup('Classifier (CatBoost)', orientation='G')
        # Add groups to the tab
        self.tabs.add_named_tab('Model', self.fe_group.gbox, [0, 0, 7, 2])
        self.tabs.add_named_tab('Model', self.classifier_params_group.gbox, [7, 0, 1, 2])
        self.tabs.setTabEnabled(self.tabs.tab_names.index('Model'), False)
        
        # Add "FE architecture" combo box to FE group
        self.qcombo_model_type = QComboBox()
        self.qcombo_model_type.addItems(sorted(get_all_models().keys()))
        self.qcombo_model_type.setToolTip('Select architecture of feature extraction model.')
        self.fe_group.glayout.addWidget(self.qcombo_model_type, 0, 0, 1, 2)

        # Add "Set Feature Extractor" button to model group
        self.set_fe_btn = QPushButton('Set Feature Extractor')
        self.set_fe_btn.setToolTip('Set the feature extraction model')
        self.fe_group.glayout.addWidget(self.set_fe_btn, 1, 0, 1, 2)

        # Add "FE layers" list to model group
        self.fe_layer_selection = QListWidget()
        self.fe_layer_selection.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.fe_layer_selection.setFixedHeight(200)
        # self.fe_layer_selection.setMaximumHeight(200)
        # self.fe_layer_selection.setMinimumHeight(40)
        # Turn off stretching
        # self.fe_group.glayout.setRowStretch(2, 0)
        self.fe_group.glayout.addWidget(self.fe_layer_selection, 2, 0, 1, 2)

        # Create and add scales selection to model group
        self.fe_scaling_factors = QComboBox()
        self.fe_scaling_factors.addItem('[1]',[1])
        self.fe_scaling_factors.addItem('[1,2]',[1,2])
        self.fe_scaling_factors.addItem('[1,2,4]',[1,2,4])
        self.fe_scaling_factors.addItem('[1,2,4,8]',[1,2,4,8])
        self.fe_scaling_factors.setCurrentText('[1,2]')
        self.fe_group.glayout.addWidget(QLabel('Downscaling factors'), 3, 0, 1, 1)
        self.fe_group.glayout.addWidget(self.fe_scaling_factors, 3, 1, 1, 1)

        # Add interpolation order spinbox to model group
        self.spin_interpolation_order = QSpinBox()
        self.spin_interpolation_order.setMinimum(0)
        self.spin_interpolation_order.setMaximum(5)
        self.spin_interpolation_order.setValue(1)
        self.spin_interpolation_order.setToolTip('Interpolation order for image rescaling')
        self.fe_group.glayout.addWidget(QLabel('Interpolation order'), 4, 0, 1, 1)
        self.fe_group.glayout.addWidget(self.spin_interpolation_order, 4, 1, 1, 1)

        # Add min features checkbox to model group
        self.check_use_min_features = QCheckBox('Use min features')
        self.check_use_min_features.setChecked(False)
        self.check_use_min_features.setToolTip('Use same number of features from each layer. Otherwise use all features from each layer.')
        self.fe_group.glayout.addWidget(self.check_use_min_features, 5, 0, 1, 1)

        # Add use cuda checkbox to model group
        self.check_use_cuda = QCheckBox('Use cuda')
        self.check_use_cuda.setChecked(False)
        self.check_use_cuda.setToolTip('Use GPU for training and segmentation')
        self.fe_group.glayout.addWidget(self.check_use_cuda, 5, 1, 1, 1)

        # Add classifier parameters
        self.spin_iterations = QSpinBox()
        self.spin_iterations.setMinimum(1)
        self.spin_iterations.setMaximum(1000)
        self.spin_iterations.setValue(50)
        self.spin_iterations.setToolTip('Set the number of iterations for the classifier')
        self.classifier_params_group.glayout.addWidget(QLabel('Iterations'), 0, 0, 1, 1)
        self.classifier_params_group.glayout.addWidget(self.spin_iterations, 0, 1, 1, 1)

        self.spin_learning_rate = QDoubleSpinBox()
        self.spin_learning_rate.setMinimum(0.001)
        self.spin_learning_rate.setMaximum(1.0)
        self.spin_learning_rate.setSingleStep(0.01)
        self.spin_learning_rate.setValue(0.1)
        self.spin_learning_rate.setToolTip('Set the learning rate for the classifier')
        self.classifier_params_group.glayout.addWidget(QLabel('Learning Rate'), 1, 0, 1, 1)
        self.classifier_params_group.glayout.addWidget(self.spin_learning_rate, 1, 1, 1, 1)

        self.spin_depth = QSpinBox()
        self.spin_depth.setMinimum(1)
        self.spin_depth.setMaximum(20)
        self.spin_depth.setValue(5)
        self.spin_depth.setToolTip('Set the depth of the trees for the classifier')
        self.classifier_params_group.glayout.addWidget(QLabel('Depth'), 2, 0, 1, 1)
        self.classifier_params_group.glayout.addWidget(self.spin_depth, 2, 1, 1, 1)

        # === === ===

        # If project mode is initially activated, add project tab and widget
        if init_project is True:
            self._add_project()

        # Add connections and initialize by selecting layer
        self.add_connections()
        self.select_layer()

        # Add key bindings
        self.viewer.bind_key('a', self.toggle_annotation, overwrite=True)
        self.viewer.bind_key('r', self.toggle_prediction, overwrite=True)

    def add_connections(self):

        # Reset layer choices in dropdown when layers are renamed; also bind this behaviour to inserted layers
        for layer in self.viewer.layers: layer.events.name.connect(self.select_image_layer_widget.reset_choices)
        self.viewer.layers.events.inserted.connect(self._on_insert_layer)

        # Reset layer choices in dropdown when napari layers are added or removed
        self.viewer.layers.events.inserted.connect(self.select_image_layer_widget.reset_choices)
        self.viewer.layers.events.removed.connect(self.select_image_layer_widget.reset_choices)
        self.viewer.layers.events.inserted.connect(self.select_annotation_layer_widget.reset_choices)
        self.viewer.layers.events.removed.connect(self.select_annotation_layer_widget.reset_choices)
        
        # Change layer selection when choosing a new image layer in dropdown
        self.select_image_layer_widget.changed.connect(self.select_layer)
        # NOTE: Changing annotation layer does not have to trigger anything
        self.add_layers_btn.clicked.connect(self.add_empty_layers)

        # Train
        self.train_classifier_btn.clicked.connect(self.train_classifier)
        self.check_auto_seg.stateChanged.connect(lambda: setattr(self, 'auto_seg', self.check_auto_seg.isChecked()))
        self.check_use_project.stateChanged.connect(self._add_project)
        self.train_classifier_on_project_btn.clicked.connect(self.train_classifier_on_project)

        # Segment
        self.segment_btn.clicked.connect(self.predict)
        self.segment_all_btn.clicked.connect(self.predict_all)

        # Load/Save
        self.save_model_btn.clicked.connect(self._on_save_model)
        self.load_model_btn.clicked.connect(self._on_load_model)

        # Image Processing
        self.radio_single_channel.toggled.connect(self._reset_radio_norm_settings)
        self.radio_multi_channel.toggled.connect(self._reset_radio_norm_settings)
        self.radio_rgb.toggled.connect(self._reset_radio_norm_settings)
        self.radio_no_normalize.toggled.connect(self._reset_stats)
        self.radio_normalized_over_stack.toggled.connect(self._reset_stats)
        self.radio_normalize_by_image.toggled.connect(self._reset_stats)

        # Model
        self.check_use_custom_model.stateChanged.connect(self._set_custom_model)
        self._reset_model_btn.clicked.connect(self._reset_model)

        # Acceleration
        self.spin_downsample.valueChanged.connect(lambda: setattr(self.param, 'image_downsample', self.spin_downsample.value()))
        self.check_tile_annotations.stateChanged.connect(lambda: setattr(self.param, 'tile_annotations', self.check_tile_annotations.isChecked()))
        self.check_tile_image.stateChanged.connect(lambda: setattr(self.param, 'tile_image', self.check_tile_image.isChecked()))

        # Model tab
        # Feature extractor
        self.qcombo_model_type.currentIndexChanged.connect(self._on_model_selected)
        self.set_fe_btn.clicked.connect(self._on_set_fe_model)
        self.fe_layer_selection.itemSelectionChanged.connect(self._fe_layer_selection_changed)
        self.fe_scaling_factors.currentIndexChanged.connect(self._update_scalings_from_gui)
        self.spin_interpolation_order.valueChanged.connect(lambda: setattr(self.param, 'order', self.spin_interpolation_order.value()))
        self.check_use_cuda.stateChanged.connect(lambda: setattr(self.param, 'use_cuda', self.check_use_cuda.isChecked()))
        self.check_use_min_features.stateChanged.connect(lambda: setattr(self.param, 'use_min_features', self.check_use_min_features.isChecked()))
        # Classifier
        self.spin_iterations.valueChanged.connect(lambda: setattr(self.param, 'classif_iterations', self.spin_iterations.value()))
        self.spin_learning_rate.valueChanged.connect(lambda: setattr(self.param, 'classif_learning_rate', self.spin_learning_rate.value()))
        self.spin_depth.valueChanged.connect(lambda: setattr(self.param, 'classif_depth', self.spin_depth.value()))

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

    # Layer selection

    def _on_insert_layer(self, event=None):
        '''Bind layer renaming to layer selection update.'''
        layer = event.value
        layer.events.name.connect(self.select_image_layer_widget.reset_choices)
        layer.events.name.connect(self.select_annotation_layer_widget.reset_choices)

    def select_layer(self, newtext=None):
        """Assign the layer to segment and update data radio buttons accordingly"""
        print(self.param)
        self.selected_channel = self.select_image_layer_widget.native.currentText()
        if self.select_image_layer_widget.value is None:
            self.add_layers_btn.setEnabled(False)
        else:
            self.add_layers_btn.setEnabled(True)
        
        # set radio buttons depending on selected image type
        if self.select_image_layer_widget.value is not None:
            if self.select_image_layer_widget.value.rgb:
                self.radio_rgb.setChecked(True)
                self.radio_multi_channel.setEnabled(False)
                self.radio_single_channel.setEnabled(False)
            elif self.select_image_layer_widget.value.ndim == 2:
                self.radio_single_channel.setChecked(True)
                self.radio_multi_channel.setEnabled(False)
                self.radio_rgb.setEnabled(False)
            elif self.select_image_layer_widget.value.ndim == 3:
                self.radio_rgb.setEnabled(False)
                self.radio_multi_channel.setEnabled(True)
                self.radio_single_channel.setEnabled(True)
                self.radio_single_channel.setChecked(True)
            elif self.select_image_layer_widget.value.ndim == 4:
                self.radio_rgb.setEnabled(False)
                self.radio_multi_channel.setEnabled(True)
                self.radio_single_channel.setEnabled(False)
                self.radio_multi_channel.setChecked(True)

            self._reset_radio_norm_settings()
            self._reset_predict_buttons()

    # R: Check this in detail
    def add_empty_layers(self, event=None, force_add=True):
        """Add annotation and prediction layers to viewer. If the layer already exists,
        remove it and add a new one. If the widget is used as third party (self.third_party=True),
        no layer is added if it didn't exist before, unless force_add=True (e.g. when the user click
        on the add layer button)"""

        self.event = event # R: Why is this needed?
        if self.select_image_layer_widget.value is None:
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

    # Train

    def train_classifier(self):
        """Given a set of new annotations, update the CatBoost classifier."""

        unique_labels = np.unique(self.select_annotation_layer_widget.value.data)
        if (not 1 in unique_labels) | (not 2 in unique_labels):
            raise Exception('You need annotations for at least foreground and background')
        
        if self.model is None:
            if not self.check_use_custom_model.isChecked():
                self.set_default_model()
            else:
                raise Exception('You have to define and load a model first')

        image_stack = self.get_selected_layer_data()
        
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
                param=self.param,
            )
            self.classifier = train_classifier(
                features, targets,
                iterations=self.param.classif_iterations,
                learning_rate=self.param.classif_learning_rate,
                depth=self.param.classif_depth,
        )
            self._reset_predict_buttons()
            self.save_model_btn.setEnabled(True)
            self.current_model_path.setText('Unsaved')
            
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(False)

        if self.auto_seg:
            self.predict()

    def _add_project(self, event=None):
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

    def train_classifier_on_project(self):
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
        
        self.viewer.layers.events.removed.disconnect(self._reset_model)
        with progress(total=0) as pbr:
            pbr.set_description(f"Training")
            self.current_model_path.setText('In training')
            all_features, all_targets = [], []
            for ind in range(num_files):
                self.project_widget.file_list.setCurrentRow(ind)

                image_stack = self.get_selected_layer_data()

                features, targets = get_features_current_layers(
                    model=self.model,
                    image=image_stack,
                    annotations=self.select_annotation_layer_widget.value.data,
                    param=self.param,
                )
                if features is None:
                    continue
                all_features.append(features)
                all_targets.append(targets)
            
            all_features = np.concatenate(all_features, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)

            self.classifier = train_classifier(all_features, all_targets)
            self._reset_predict_buttons()
            self.save_model_btn.setEnabled(True)
            self.current_model_path.setText('Unsaved')

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(False)

        self.viewer.layers.events.removed.connect(self._reset_model)

    # Predict

    def predict(self):
        """Predict the segmentation of the currently viewed frame based 
        on a RF model trained with annotations"""

        if self.model is None:
            if not self.check_use_custom_model.isChecked():
                self.set_default_model()
            else:
                raise Exception('You have to define and load a model first')
        
        if self.classifier is None:
            self.train_classifier()

        self._check_prediction_layer_exists()

        if self.image_mean is None:
            self.get_image_stats()
        
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(True)

        with progress(total=0) as pbr:
            
            pbr.set_description(f"Prediction")

            if self.viewer.dims.ndim == 2:
                image = self.get_selected_layer_data()


            elif self.viewer.dims.ndim == 3:
                if self.radio_multi_channel.isChecked():
                    image = self.get_selected_layer_data()
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
                    image=image, model=self.model, classifier=self.classifier,
                    param = self.param, use_dask=False)
            else:
                predicted_image = self.model.predict_image(image=image,
                                                           classifier=self.classifier,
                                                           param=self.param,)
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

        if self.classifier is None:
            raise Exception('No model found. Please train a model first.')
        
        if self.model is None:
            if not self.check_use_custom_model.isChecked():
                self.set_default_model()
            else:
                raise Exception('You have to define and load a model first')
            
        self._check_prediction_layer_exists()

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
            """Select file where to save the classifier model along with the model parameters."""
            if self.classifier is None:
                raise Exception('No model found. Please train a model first.')
            
            if save_file is None:
                dialog = QFileDialog()
                save_file, _ = dialog.getSaveFileName(self, "Save model", None, "PICKLE (*.pickle)")
            save_file = Path(save_file)
            
            # Update parameters from GUI before saving
            self._update_params_from_gui()
            save_model(save_file, self.classifier, self.model, self.param, )        
            self.current_model_path.setText(save_file.name)

    def _on_load_model(self, event=None, save_file=None):
        """Select classifier model file to load along with the model parameters."""
        if save_file is None:
            dialog = QFileDialog()
            save_file, _ = dialog.getOpenFileName(self, "Choose model", None, "PICKLE (*.pickle)")
        save_file = Path(save_file)

        #tick the custom model checkbox
        self.check_use_custom_model.setChecked(True)
        
        classifier, model, param, model_state = load_model(save_file)
        self.classifier = classifier
        self.param = param
        self.model = model

        # Load model state if available
        if model_state is not None and hasattr(self.model, 'load_state_dict'):
            self.model.load_state_dict(model_state)
        
        self._update_gui_from_params()
        self._update_gui_from_model()


        self.current_model_path.setText(save_file.name)
        self._reset_predict_buttons()

    # Image Processing

    def _reset_radio_norm_settings(self, event=None):
        
        self.image_mean, self.image_std = None, None
        self.add_empty_layers(force_add=False)

        if self.select_image_layer_widget.value.ndim == 2:
            self.radio_normalize_by_image.setEnabled(True)
            self.radio_normalize_by_image.setChecked(True)
            self.radio_normalized_over_stack.setEnabled(False)
        elif (self.select_image_layer_widget.value.ndim == 3) and (self.radio_multi_channel.isChecked()):
            self.radio_normalize_by_image.setEnabled(True)
            self.radio_normalize_by_image.setChecked(True)
            self.radio_normalized_over_stack.setEnabled(False)
        else:
            self.radio_normalize_by_image.setEnabled(True)
            self.radio_normalized_over_stack.setEnabled(True)
            self.radio_normalized_over_stack.setChecked(True)
    
    def _reset_stats(self):
        self.image_mean, self.image_std = None, None

    # Model

    def _set_custom_model(self, event=None):
        """Add tab for custom model management"""
        if not self.check_use_custom_model.isChecked():
            self.tabs.setTabEnabled(self.tabs.tab_names.index('Model'), False)
            # self.qcombo_model_type.setCurrentText('single_layer_vgg16')
            self.set_default_model()
        else:
            self.tabs.setTabEnabled(self.tabs.tab_names.index('Model'), True)


    def _reset_model(self, event=None):
        self.model = None
        self.classifier = None
        self.segment_btn.setEnabled(False)
        self.segment_all_btn.setEnabled(False)
        self.save_model_btn.setEnabled(False)
        self.current_model_path.setText('None')

        if self.select_image_layer_widget.value is None:
            for x in self.channel_buttons: x.setEnabled(False)
        else:
            self.select_layer()


    # Model Tab

    def _on_model_selected(self, index):
        """Update GUI to show selectable layers of model chosen from drop-down."""
        model_type = self.qcombo_model_type.currentText()
        model_class = get_all_models()[model_type]
        self.temp_model = model_class(fe_name=model_type, use_cuda=self.check_use_cuda.isChecked())
        
        if isinstance(self.temp_model, Hookmodel):
            self._create_fe_layer_selection_for_temp_model(self.temp_model)
            self.fe_layer_selection.setEnabled(True)

            #check if outputs are selected
            if self.fe_layer_selection.count() > 0:
                self.set_fe_btn.setEnabled(False)
        else:
            self.fe_layer_selection.clear()
            self.fe_layer_selection.setEnabled(False)
            self.set_fe_btn.setEnabled(True)


        # get the default params for the model
        self.param = self.temp_model.get_default_params()
        self._update_gui_from_params()

    def _on_set_fe_model(self, event=None):
        """Create a neural network model that will be used for feature extraction."""
        self._update_params_from_gui()
        self.model = create_model(self.param)
        self._update_gui_from_model()
        self.current_model_path.setText('Unsaved')

    def _fe_layer_selection_changed(self):
        #check if temp model is a hookmodel
        if isinstance(self.temp_model, Hookmodel):
            selected_layers = self.get_selected_layer_names()
            if len(selected_layers) == 0:
                self.set_fe_btn.setEnabled(False)
            else:
                self.set_fe_btn.setEnabled(True)
        else:
            self.set_fe_btn.setEnabled(True)

    def _update_scalings_from_gui(self):
        self.param.fe_scalings = self.fe_scaling_factors.currentData()

### Helper functions

    def _create_fe_layer_selection(self):
        """Update list of selectable layers"""

        self.fe_layer_selection.clear()
        self.fe_layer_selection.addItems(self.model.selectable_layer_keys.keys())

                                 
    def _update_gui_from_model(self):
        """Update GUI based on the current model."""
        if self.model is None:
            self.fe_layer_selection.setEnabled(False)
            self.fe_layer_selection.clear()
        elif isinstance(self.model, Hookmodel):
            self._create_fe_layer_selection()
            if len(self.model.named_modules) == 1:
                self.fe_layer_selection.setCurrentRow(0)
                self.fe_layer_selection.setEnabled(False)
            else:
                self.fe_layer_selection.setEnabled(True)
                for layer in self.param.fe_layers:
                    items = self.fe_layer_selection.findItems(layer, Qt.MatchExactly)
                    for item in items:
                        item.setSelected(True)
        else:
            self.fe_layer_selection.setEnabled(False)
            self.fe_layer_selection.clear()

    def _update_params_from_gui(self):
        """Update parameters from GUI."""
        self.param.image_downsample = self.spin_downsample.value()
        self.param.normalize = self.button_group_normalize.checkedId()
        self.param.fe_use_cuda = self.check_use_cuda.isChecked()

        self.param.fe_name = self.qcombo_model_type.currentText()
        self.param.fe_layers = self.get_selected_layer_names()
        self._update_scalings_from_gui()
        self.param.fe_order = self.spin_interpolation_order.value()
        self.param.fe_use_min_features = self.check_use_min_features.isChecked()

    def _update_gui_from_params(self):
        """Update GUI from parameters."""
        self.spin_downsample.setValue(self.param.image_downsample)
        self.button_group_normalize.button(self.param.normalize).setChecked(True)
        self.check_use_cuda.setChecked(self.param.fe_use_cuda)

        self.qcombo_model_type.setCurrentText(self.param.fe_name)
        self._update_scalings_from_param()
        self.spin_interpolation_order.setValue(self.param.fe_order)
        self.check_use_min_features.setChecked(self.param.fe_use_min_features)

    def _update_scalings_from_param(self):
        index = self.fe_scaling_factors.findData(self.param.fe_scalings)
        if index != -1:
            self.fe_scaling_factors.setCurrentIndex(index)
        else:
            self.fe_scaling_factors.addItem(str(self.param.fe_scalings), self.param.fe_scalings)
            self.fe_scaling_factors.setCurrentIndex(self.fe_scaling_factors.count()-1)

    def _reset_predict_buttons(self):
        
        if (self.model is not None) and (self.select_image_layer_widget.value is not None):
            self.segment_btn.setEnabled(True)
            if self.select_image_layer_widget.value == 2:
                self.segment_all_btn.setEnabled(False)
            elif self.select_image_layer_widget.value.ndim == 3:
                if self.radio_multi_channel.isChecked():
                    self.segment_all_btn.setEnabled(False)
                else:
                    self.segment_all_btn.setEnabled(True)
            else:
                self.segment_all_btn.setEnabled(True)
    
    def _check_prediction_layer_exists(self):

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

    def _create_fe_layer_selection_for_temp_model(self, temp_model):
        self.fe_layer_selection.clear()
        self.fe_layer_selection.addItems(temp_model.selectable_layer_keys.keys())

    def get_selected_layer_data(self):
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

    def get_selected_layer_names(self):
        """Get names of selected layers."""

        selected_rows = self.fe_layer_selection.selectedItems()
        selected_layers = [x.text() for x in selected_rows]
        return selected_layers

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

    def get_data_channel_first(self):
        """Get data from selected channel. If RGB, move channel axis to first position."""
            
        image_stack = self.viewer.layers[self.selected_channel].data
        if self.viewer.layers[self.selected_channel].rgb:
            image_stack = np.moveaxis(image_stack, -1, 0)
        return image_stack

    def set_default_model(self):#, keep_rgb=False):
        """Set default model."""
        '''if keep_rgb:
            self.qcombo_model_type.setCurrentText('single_layer_vgg16_rgb')
        else:
            self.qcombo_model_type.setCurrentText('single_layer_vgg16')'''
        self.qcombo_model_type.setCurrentText('single_layer_vgg16')
        self.fe_scaling_factors.setCurrentText('[1,2]')
        self.spin_interpolation_order.setValue(1)
        self.check_use_min_features.setChecked(True)
        self._on_set_fe_model()
