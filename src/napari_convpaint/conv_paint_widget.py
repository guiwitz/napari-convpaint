from qtpy.QtWidgets import (QWidget, QPushButton,QVBoxLayout,
                            QLabel, QComboBox,QFileDialog, QListWidget,
                            QCheckBox, QAbstractItemView, QGridLayout, QSpinBox, QButtonGroup,
                            QRadioButton,QDoubleSpinBox)
from qtpy import QtWidgets, QtGui, QtCore
from qtpy.QtCore import Qt, QTimer, QUrl
from magicgui.widgets import create_widget
import napari
from napari.utils import progress
from napari.utils.notifications import show_info
from napari_guitils.gui_structures import VHGroup, TabSet
from pathlib import Path
import numpy as np
import warnings

from .conv_paint_utils import normalize_image, compute_image_stats, normalize_image_percentile, normalize_image_imagenet
from .conv_paint_model import ConvpaintModel

class ConvPaintWidget(QWidget):
    """
    Implementation of a napari widget for interactive segmentation performed
    via multiple means of feature extraction combined with a CatBoost Classifier
    trained on annotations. The default filters used to generate 
    the features are taken from the first layer of a VGG16 model
    as proposed here: https://github.com/hinderling/napari_pixel_classifier

    Parameters
    ----------
    napari_viewer : napari.Viewer
        main napari viewer
    project : bool
        use project widget for multi-image project management
    third_party : bool
        if True, widget is used as third party and will not add layers to the viewer
        by default.
    """

### Define the basic structure of the widget
    
    def __init__(self, napari_viewer, parent=None, third_party=False):

        ### Initialize the widget state

        super().__init__(parent=parent)
        self.viewer = napari_viewer
        # Initialize the Convpaint Model
        self.cp_model = ConvpaintModel()
        # Get default parameters to set in widget
        self.default_cp_param = ConvpaintModel.get_default_params()
        # Create a temporary FE model for display
        self.temp_fe_model = ConvpaintModel.create_fe(self.default_cp_param.fe_name,
                                                      self.default_cp_param.fe_use_gpu)
        
        self.third_party = third_party
        self.selected_channel = None
        self.spatial_dim_info_thresh = 1000000
        self.default_brush_size = 3
        self._reset_attributes()

        ### Build the widget
        style_for_infos = "font-size: 12px; color: rgba(120, 120, 120, 80%); font-style: italic"
        style_for_shortcut_info = "font-size: 11px; color: rgba(120, 120, 120, 80%); font-style: italic"

        # Create main layout
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # Create and add tabs
        self.tab_names = ['Home', 'Model options']
        self.tab_names += ['Class labels']
        # self.tab_names += ['Files']
        self.tab_names += ['Advanced']
        tab_layouts = [None if name != 'Model options' else QGridLayout() for name in self.tab_names]
        self.tabs = TabSet(self.tab_names, tab_layouts=tab_layouts) # [None, None, QGridLayout()])
        tab_bar = self.tabs.tabBar()
        tab_bar.setSizePolicy(tab_bar.sizePolicy().horizontalPolicy(), tab_bar.sizePolicy().verticalPolicy())

        # Create docs button
        docs_button = QtWidgets.QToolButton()
        docs_button.setText("Documentation")
        docs_button.setStyleSheet("QToolButton {color: #999; text-decoration: underline; margin-left: 4px; margin-right: 8px}")
        docs_button.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(QUrl("https://guiwitz.github.io/napari-convpaint/book/Landing.html")))
        docs_button.setToolTip("Open the documentation in your browser")

        # Create a widget to hold tab bar and button side by side
        tab_header_widget = QWidget()
        tab_header_layout = QtWidgets.QHBoxLayout(tab_header_widget)
        tab_header_layout.setContentsMargins(0, 0, 0, 0)
        tab_header_layout.setSpacing(0)

        tab_header_layout.addWidget(tab_bar)
        tab_header_layout.addWidget(docs_button)

        # Add to your main layout
        self.main_layout.addWidget(tab_header_widget)
        self.main_layout.addWidget(self.tabs)
        
        # Disable project tab as long as not activated
        # self.tabs.setTabEnabled(self.tabs.tab_names.index('Project'), False)
        
        # Align rows in some tabs on top
        for tab_name in ['Home', 'Model options', 'Advanced']:
            if tab_name in self.tabs.tab_names:
                self.tabs.widget(self.tabs.tab_names.index(tab_name)).layout().setAlignment(Qt.AlignTop)

        # === HOME TAB ===

        # Create groups and separate labels
        self.model_group = VHGroup('Model', orientation='G')
        self.layer_selection_group = VHGroup('Layer selection', orientation='G')
        self.image_processing_group = VHGroup('Image type and normalization', orientation='G')
        self.train_group = VHGroup('Train/Segment', orientation='G')
        # self.segment_group = VHGroup('Segment', orientation='G')
        # self.load_save_group = VHGroup('Load/Save', orientation='G')
        self.acceleration_group = VHGroup('Acceleration and post-processing', orientation='G')
        # Create the shortcuts info
        shortcuts_text1 = 'Shift+a: Toggle annotations\nShift+s: Train\nShift+d: Predict\nShift+f: Toggle prediction'
        shortcuts_text2 = 'Shift+q: Set annotation label 1\nShift+w: Set annotation label 2\nShift+e: Set annotation label 3\nShift+r: Set annotation label 4'
        shortcuts_label1 = QLabel(shortcuts_text1)
        shortcuts_label2 = QLabel(shortcuts_text2)
        shortcuts_label1.setStyleSheet(style_for_shortcut_info)
        shortcuts_label2.setStyleSheet(style_for_shortcut_info)
        shortcuts_grid = QGridLayout()
        shortcuts_grid.addWidget(shortcuts_label1, 0, 0)
        shortcuts_grid.addWidget(shortcuts_label2, 0, 1)
        shortcuts_widget = QWidget()
        shortcuts_widget.setLayout(shortcuts_grid)

        # Create a simple box to add a number, for testing purposes
        # self.number_box = create_widget(annotation=int, label='Test')
        # self.number_box.value = 0

        # Add groups to the tab
        self.tabs.add_named_tab('Home', self.model_group.gbox)
        self.tabs.add_named_tab('Home', self.layer_selection_group.gbox)
        # self.tabs.add_named_tab('Home', self.segment_group.gbox)
        # self.tabs.add_named_tab('Home', self.load_save_group.gbox)
        self.tabs.add_named_tab('Home', self.image_processing_group.gbox)
        self.tabs.add_named_tab('Home', self.train_group.gbox)
        self.tabs.add_named_tab('Home', self.acceleration_group.gbox)
        self.tabs.widget(self.tabs.tab_names.index('Home')).layout().addWidget(shortcuts_widget)
        # self.tabs.add_named_tab('Home', self.number_box.native)

        # Add buttons for "Model" group
        # Current model description label
        self.model_description1 = QLabel('None')
        self.model_group.glayout.addWidget(self.model_description1, 0,0,1,2)
        # Save and load model buttons
        self.save_model_btn = QPushButton('Save model')
        self.save_model_btn.setToolTip('Save model as *.pkl (incl. classifier) or *.yml (parameters only) file')
        self.save_model_btn.setEnabled(True)
        self.model_group.glayout.addWidget(self.save_model_btn, 1,0,1,1)
        self.load_model_btn = QPushButton('Load model')
        self.load_model_btn.setToolTip('Select *.pkl or *.yml file to load')
        self.model_group.glayout.addWidget(self.load_model_btn, 1,1,1,1)
        # Reset model button
        self._reset_convpaint_btn = QPushButton('Reset Convpaint')
        self._reset_convpaint_btn.setToolTip('Discard current model and create new default model.')
        self.model_group.glayout.addWidget(self._reset_convpaint_btn, 2,0,1,2)

        # Add elements to "Layer selection" group
        # Image layer (widget for selecting the layer to segment)
        self.image_layer_selection_widget = create_widget(annotation=napari.layers.Image, label='Pick image')
        self._update_image_layers()
        # Annotation layer
        self.annotation_layer_selection_widget = create_widget(annotation=napari.layers.Labels, label='Pick annotation')
        self._update_annotation_layers()
        # Add widgets to layout
        self.layer_selection_group.glayout.addWidget(QLabel('Image layer'), 0,0,1,1)
        self.layer_selection_group.glayout.addWidget(self.image_layer_selection_widget.native, 0,1,1,1)
        self.layer_selection_group.glayout.addWidget(QLabel('Annotation layer'), 1,0,1,1)
        self.layer_selection_group.glayout.addWidget(self.annotation_layer_selection_widget.native, 1,1,1,1)
        # Add button for adding annotation/segmentation layers
        self.add_layers_btn = QPushButton('Add annotations layer')
        self.add_layers_btn.setEnabled(True)
        self.layer_selection_group.glayout.addWidget(self.add_layers_btn, 2,0,1,2)

        # Add buttons for "Image Processing" group
        # Radio buttons for "Data Dimensions"
        self.button_group_channels = QButtonGroup()
        self.radio_single_channel = QRadioButton('Single channel image')
        self.radio_single_channel.setToolTip('2D images or 3D images where additional dimension is NOT channels')
        self.radio_multi_channel = QRadioButton('Multichannel image')
        self.radio_multi_channel.setToolTip('Images with an additional channel dimension')
        self.radio_rgb = QRadioButton('RGB image')
        self.radio_rgb.setToolTip('This option is used with images displayed as RGB')
        self.radio_single_channel.setChecked(True)
        self.channel_buttons = [self.radio_single_channel, self.radio_multi_channel, self.radio_rgb]
        for x in self.channel_buttons: x.setEnabled(False)
        self.button_group_channels.addButton(self.radio_single_channel, id=1)
        self.button_group_channels.addButton(self.radio_multi_channel, id=2)
        self.button_group_channels.addButton(self.radio_rgb, id=3)
        self.image_processing_group.glayout.addWidget(self.radio_single_channel, 0,0,1,1)
        self.image_processing_group.glayout.addWidget(self.radio_multi_channel, 1,0,1,1)
        self.image_processing_group.glayout.addWidget(self.radio_rgb, 2,0,1,1)
        # Create and add a label to act as a vertical divider
        divider_label = QLabel('¦\n¦\n¦\n¦')
        divider_label.setAlignment(Qt.AlignCenter)
        divider_label.setStyleSheet("font-size: 13px; color: rgba(120, 120, 120, 35%);")
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

        # Add buttons for "Train/Segment" group
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
        # self.train_group.glayout.addWidget(self.train_classifier_on_project_btn, 2,0,1,2)
        # if init_project is False:
        #     self.train_classifier_on_project_btn.setEnabled(False)
        self.segment_btn = QPushButton('Segment image')
        self.segment_btn.setEnabled(False)
        self.segment_btn.setToolTip('Segment 2D image or current slice/frame of 3D image/movie ')
        self.train_group.glayout.addWidget(self.segment_btn, 1,0,1,1)
        self.segment_all_btn = QPushButton('Segment stack')
        self.segment_all_btn.setToolTip('Segment all slices/frames of 3D image/movie')
        self.segment_all_btn.setEnabled(False)
        self.train_group.glayout.addWidget(self.segment_all_btn, 1,1,1,1)

        # Add elements to "Acceleration" group
        # "Downsample" spinbox
        self.spin_downsample = QSpinBox()
        self.spin_downsample.setMinimum(-10)
        self.spin_downsample.setMaximum(10)
        self.spin_downsample.setValue(1)
        self.spin_downsample.setToolTip('Reduce image size, e.g. for faster computing (output is rescaled to original size). ' +
                                        'Negative values will instead upscale the image by the absolute value.')
        self.acceleration_group.glayout.addWidget(QLabel('Downsample input'), 1,0,1,1)
        self.acceleration_group.glayout.addWidget(self.spin_downsample, 1,1,1,1)
        # "Smoothen output" spinbox
        self.spin_smoothen = QSpinBox()
        self.spin_smoothen.setMinimum(1)
        self.spin_smoothen.setMaximum(50)
        self.spin_smoothen.setValue(1)
        self.spin_smoothen.setToolTip('Smoothen output with a filter of this size.')
        self.acceleration_group.glayout.addWidget(QLabel('Smoothen segmentation'), 2,0,1,1)
        self.acceleration_group.glayout.addWidget(self.spin_smoothen, 2,1,1,1)
        # "Tile annotations" checkbox
        self.check_tile_annotations = QCheckBox('Tile annotations for training')
        self.check_tile_annotations.setChecked(False)
        self.check_tile_annotations.setToolTip('Crop around annotated regions to speed up training.\nDisable for models that extract long range features (e.g. DINO).')
        self.acceleration_group.glayout.addWidget(self.check_tile_annotations, 0,0,1,1)
        # "Tile image" checkbox
        self.check_tile_image = QCheckBox('Tile image for segmentation')
        self.check_tile_image.setChecked(False)
        self.check_tile_image.setToolTip('Tile image to reduce memory usage.\nUse with care when using models that extract long range features (e.g. DINO).')
        self.acceleration_group.glayout.addWidget(self.check_tile_image, 0,1,1,1)

        # === MODEL TAB ===

        # Create three groups
        self.current_model_group = VHGroup('Current model', orientation='G')
        self.fe_group = VHGroup('Feature extractor', orientation='G')
        self.classifier_params_group = VHGroup('Classifier (CatBoost)', orientation='G')

        # Add groups to the tab
        self.tabs.add_named_tab('Model options', self.current_model_group.gbox, [0, 0, 1, 2])
        self.tabs.add_named_tab('Model options', self.fe_group.gbox, [2, 0, 8, 2])
        self.tabs.add_named_tab('Model options', self.classifier_params_group.gbox, [10, 0, 3, 2])
        self.tabs.setTabEnabled(self.tabs.tab_names.index('Model options'), True)
        
        # Current model
        self.model_description2 = QLabel('None')
        self.current_model_group.glayout.addWidget(self.model_description2, 0, 0, 1, 2)

        # Add "FE architecture" combo box to FE group
        self.qcombo_fe_type = QComboBox()
        self.qcombo_fe_type.addItems(sorted(ConvpaintModel.get_fe_models_types().keys()))
        self.qcombo_fe_type.setToolTip('Select architecture of feature extraction model.')
        num_items = self.qcombo_fe_type.count()
        self.qcombo_fe_type.setMaxVisibleItems(num_items) # Make sure the dropdown shows all items
        self.fe_group.glayout.addWidget(self.qcombo_fe_type, 1, 0, 1, 2)

        # Add "FE description" label to FE group
        self.FE_description = QLabel(self.temp_fe_model.get_description())
        self.FE_description.setWordWrap(True)
        self.fe_group.glayout.addWidget(self.FE_description, 2, 0, 1, 2)

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

        # Add use gpu checkbox to FE group
        self.check_use_gpu = QCheckBox('Use GPU')
        self.check_use_gpu.setChecked(False)
        self.check_use_gpu.setToolTip('Use GPU for training and segmentation')
        self.fe_group.glayout.addWidget(self.check_use_gpu, 6, 1, 1, 1)

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
        self.spin_iterations.setValue(self.default_cp_param.clf_iterations)
        self.spin_iterations.setToolTip('Set the number of iterations for the classifier')
        self.classifier_params_group.glayout.addWidget(QLabel('Iterations'), 0, 0, 1, 1)
        self.classifier_params_group.glayout.addWidget(self.spin_iterations, 0, 1, 1, 1)

        self.spin_learning_rate = QDoubleSpinBox()
        self.spin_learning_rate.setMinimum(0.001)
        self.spin_learning_rate.setMaximum(1.0)
        self.spin_learning_rate.setSingleStep(0.01)
        self.spin_learning_rate.setValue(self.default_cp_param.clf_learning_rate)
        self.spin_learning_rate.setToolTip('Set the learning rate for the classifier')
        self.classifier_params_group.glayout.addWidget(QLabel('Learning Rate'), 1, 0, 1, 1)
        self.classifier_params_group.glayout.addWidget(self.spin_learning_rate, 1, 1, 1, 1)

        self.spin_depth = QSpinBox()
        self.spin_depth.setMinimum(1)
        self.spin_depth.setMaximum(20)
        self.spin_depth.setValue(self.default_cp_param.clf_depth)
        self.spin_depth.setToolTip('Set the depth of the trees for the classifier')
        self.classifier_params_group.glayout.addWidget(QLabel('Depth'), 2, 0, 1, 1)
        self.classifier_params_group.glayout.addWidget(self.spin_depth, 2, 1, 1, 1)

        self.set_default_clf_btn = QPushButton('Reset to defaults')
        self.set_default_clf_btn.setEnabled(True)
        self.classifier_params_group.glayout.addWidget(self.set_default_clf_btn, 3, 0, 1, 2)

        # === CLASS LABELS TAB ===

        if 'Class labels' in self.tab_names:
            # Create the main layout
            self.class_labels_layout = QGridLayout()
            self.class_labels_layout.setAlignment(Qt.AlignTop)
            self.class_labels_widget = QWidget()
            self.class_labels_widget.setLayout(self.class_labels_layout)
            
            # Add text to instruct the user (note that it is optional to use)
            class_label_text = QLabel('Set the names of the classes (optional):')
            class_label_text.setWordWrap(True)
            # class_label_text.setStyleSheet("font-size: 11px; color: rgba(120, 120, 120, 70%)")#; font-style: italic")
            self.class_labels_layout.addWidget(class_label_text, 0, 0, 1, 10)

            # Add buttons ("add class", "remove class" and reset)
            self.add_class_btn = QPushButton('Add class')
            self.add_class_btn.clicked.connect(lambda: self._on_add_class_label(text=None))
            self.class_labels_layout.addWidget(self.add_class_btn, len(self.initial_labels)+1, 0, 1, 5)
            self.remove_class_btn = QPushButton('Remove class')
            self.remove_class_btn.clicked.connect(lambda: self._on_remove_class_label(del_annots=True))
            self.class_labels_layout.addWidget(self.remove_class_btn, len(self.initial_labels)+1, 5, 1, 5)
            # Reset to initial state
            self.reset_class_btn = QPushButton('Reset to default')
            self.reset_class_btn.clicked.connect(self._on_reset_class_labels)
            self.class_labels_layout.addWidget(self.reset_class_btn, len(self.initial_labels)+2, 0, 1, 10)
            self.btn_class_distribution_annot = QPushButton('Show class distribution (in annotation)')
            self.btn_class_distribution_annot.setToolTip('Show a diagram of the class distribution in the annotation layer')
            self.class_labels_layout.addWidget(self.btn_class_distribution_annot, len(self.initial_labels)+3, 0, 1, 10)

            # Create the class labels
            self._create_default_class_labels()

            # Add the widget to the tab
            self.class_labels_layout.setColumnStretch(1, 1)
            self.class_labels_layout.setColumnStretch(5, 1)
            self.tabs.add_named_tab('Class labels', self.class_labels_widget)

        # === ADVANCED TAB ===

        if 'Advanced' in self.tab_names:
            # Create group boxes
            self.advanced_note_group = VHGroup('Important note', orientation='G')
            self.advanced_labels_group = VHGroup('Layers handling', orientation='G')
            self.advanced_training_group = VHGroup('Training', orientation='G')
            # self.advanced_multifile_group = VHGroup('Multifile Training', orientation='G')
            self.advanced_prediction_group = VHGroup('Prediction', orientation='G')
            self.advanced_input_group = VHGroup('Input', orientation='G')
            self.advanced_output_group = VHGroup('Output', orientation='G')
            self.advanced_unsupervised_group = VHGroup('Unsupervised extraction (without annotations)', orientation='G')

            # Add groups to the tab
            self.tabs.add_named_tab('Advanced', self.advanced_note_group.gbox)
            self.tabs.add_named_tab('Advanced', self.advanced_labels_group.gbox)
            self.tabs.add_named_tab('Advanced', self.advanced_training_group.gbox)
            # self.tabs.add_named_tab('Advanced', self.advanced_multifile_group.gbox)$
            self.tabs.add_named_tab('Advanced', self.advanced_prediction_group.gbox)
            self.tabs.add_named_tab('Advanced', self.advanced_input_group.gbox)
            self.tabs.add_named_tab('Advanced', self.advanced_output_group.gbox)
            self.tabs.add_named_tab('Advanced', self.advanced_unsupervised_group.gbox)

            # Text to warn the user about their responsibility
            self.advanced_note = QLabel("Applying these options may lead to situations where the tool does not function as expected. " +
                                        "In particular, it is the user's responsibility that the dimensions of images and annotations are compatible. " +
                                        "Please refer to the documentation or contact the developers for assistance.")
            self.advanced_note.setStyleSheet(style_for_infos)
            self.advanced_note.setWordWrap(True)
            self.advanced_note_group.glayout.addWidget(self.advanced_note, 0, 0, 1, 2)

            # Checkbox to turn off automatic addition of annot/segmentation layers
            self.check_auto_add_layers = QCheckBox('Auto add annotations')
            self.check_auto_add_layers.setToolTip('Automatically add annotation layer when selecting images')
            self.check_auto_add_layers.setChecked(self.auto_add_layers)
            self.advanced_labels_group.glayout.addWidget(self.check_auto_add_layers, 1, 0, 1, 1)

            # Checkbox for keeping old layers
            self.check_keep_layers = QCheckBox('Keep old layers')
            self.check_keep_layers.setToolTip('Keep old annotation and output layers when creating new ones.')
            self.check_keep_layers.setChecked(self.keep_layers)
            self.advanced_labels_group.glayout.addWidget(self.check_keep_layers, 1, 1, 1, 1)

            # Button for adding annotation layers for selected images
            self.btn_add_all_annot_layers = QPushButton('Add for all selected')
            self.btn_add_all_annot_layers.setToolTip('Add annotation layers for all selected images in the layers list')
            self.advanced_labels_group.glayout.addWidget(self.btn_add_all_annot_layers, 2, 0, 1, 1)

            # Checkbox for auto-selecting annotation layers
            self.check_auto_select_annot = QCheckBox('Auto-select annotation layer')
            self.check_auto_select_annot.setToolTip('Automatically select annotation layers when selecting images')
            self.check_auto_select_annot.setChecked(self.auto_select_annot)
            self.advanced_labels_group.glayout.addWidget(self.check_auto_select_annot, 2, 1, 1, 1)

            # Textbox to define the prefix for the annotation layers; NOTE: DISABLED FOR NOW
            # self.text_annot_prefix = QtWidgets.QLineEdit()
            # self.text_annot_prefix.setText('annot_')
            # self.text_annot_prefix.setToolTip('Prefix for annotation layers to be used for training')
            # self.advanced_labels_group.glayout.addWidget(QLabel('Annotation prefix'), 2, 0, 1, 1)
            # self.advanced_labels_group.glayout.addWidget(self.text_annot_prefix, 2, 1, 1, 1)
            # Ensure both columns are stretched equally
            self.advanced_labels_group.glayout.setColumnStretch(0, 1)
            self.advanced_labels_group.glayout.setColumnStretch(1, 1)

            # Button for training on selected images
            self.btn_train_on_selected = QPushButton('Train on selected data')
            self.btn_train_on_selected.setToolTip("Train using layers selected in the viewer's layer list and beginning with 'annotations'")
            # self.advanced_training_group.glayout.addWidget(self.btn_train_on_selected, 1, 0, 1, 2)

            # Radio Buttons for continuous training
            self.button_group_cont_training = QButtonGroup()
            self.radio_img_training = QRadioButton('Image')
            self.radio_img_training.setToolTip('Keep features in memory, updating them only for new annotations in each training, as long as the image is not changed')
            self.radio_global_training = QRadioButton('Global')
            self.radio_global_training.setToolTip('Keep features in memory, updating them only for new annotations in each training, until reset manually')
            self.radio_single_training = QRadioButton('Off')
            self.radio_single_training.setToolTip('Extract all features freshly for each training')
            self.radio_img_training.setChecked(True)
            self.button_group_cont_training.addButton(self.radio_img_training, id=1)
            self.button_group_cont_training.addButton(self.radio_global_training, id=2)
            self.button_group_cont_training.addButton(self.radio_single_training, id=3)
            self.advanced_training_group.glayout.addWidget(QLabel('Continuous training:'), 2, 0, 1, 1)
            self.advanced_training_group.glayout.addWidget(self.radio_img_training, 2,1,1,1)
            self.advanced_training_group.glayout.addWidget(self.radio_global_training, 2,2,1,1)
            self.advanced_training_group.glayout.addWidget(self.radio_single_training, 2,3,1,1)

            # self.check_cont_training = QCheckBox('Continuous training')
            # self.check_cont_training.setToolTip('Save and use combined features in memory for training')
            # self.check_cont_training.setChecked(self.cont_training)
            # self.advanced_training_group.glayout.addWidget(self.check_cont_training, 2, 0, 1, 1)

            # Label for number of trainings performed
            self.label_training_count = QLabel('')
            self._update_training_counts()
            self.advanced_training_group.glayout.addWidget(self.label_training_count, 3, 0, 1, 2)

            # Button to display a diagram of class distribution
            self.btn_class_distribution_trained = QPushButton('Show class distr. (trained)')
            self.btn_class_distribution_trained.setToolTip('Show a diagram of the class distribution in the data saved in the model for training')
            self.advanced_training_group.glayout.addWidget(self.btn_class_distribution_trained, 3, 2, 1, 2)

            # Reset training button
            self.btn_reset_training = QPushButton('Reset continuous training')
            self.btn_reset_training.setToolTip('Clear training history and restart training counter')
            self.advanced_training_group.glayout.addWidget(self.btn_reset_training, 4, 0, 1, 4)

            # Dask option
            self.check_use_dask = QCheckBox('Use Dask when tiling image for segmentation')
            self.check_use_dask.setToolTip('Use Dask when using the option "Tile for segmentation"')
            self.check_use_dask.setChecked(self.use_dask)
            self.advanced_prediction_group.glayout.addWidget(self.check_use_dask, 0, 0, 1, 1)

            # Input channels option
            self.text_input_channels = QtWidgets.QLineEdit()
            self.text_input_channels.setPlaceholderText('e.g. 0,1,2 or 0,1')
            self.text_input_channels.setToolTip('Comma-separated list of channels to use for training and segmentation.\nLeave empty to use all channels.')
            self.advanced_input_group.glayout.addWidget(QLabel('Input channels (empty = all)'), 0, 0, 1, 2)
            self.advanced_input_group.glayout.addWidget(self.text_input_channels, 0, 2, 1, 2)

            # Button to switch first to axes
            self.btn_switch_axes = QPushButton('Switch channels axis')
            self.btn_switch_axes.setToolTip('Switch first two axes of the input image (to match the convention to have channels first)')
            self.advanced_input_group.glayout.addWidget(self.btn_switch_axes, 1, 0, 1, 2)

            # Checkbox for adding segmentation
            self.check_add_seg = QCheckBox('Segmentation')
            self.check_add_seg.setToolTip('Add a layer with the predicted segmentation as output (= highest class probability)')
            self.check_add_seg.setChecked(self.add_seg)
            self.advanced_output_group.glayout.addWidget(self.check_add_seg, 0, 0, 1, 1)

            # Checkbox for adding probabilities
            self.check_add_probas = QCheckBox('Probabilities')
            self.check_add_probas.setToolTip('Add a layer with class probabilities as output')
            self.check_add_probas.setChecked(self.add_probas)
            self.advanced_output_group.glayout.addWidget(self.check_add_probas, 0, 1, 1, 1)

            # Button to add features for the current plane
            self.btn_add_features = QPushButton('Get features image')
            self.btn_add_features.setToolTip('Add a layer with the features extracted for the current plane')
            self.advanced_unsupervised_group.glayout.addWidget(self.btn_add_features, 2, 0, 1, 2)
            # Button to add features for the whole stack
            self.btn_add_features_stack = QPushButton('Get features of stack')
            self.btn_add_features_stack.setToolTip('Add a layer with the features extracted for the whole stack')
            self.advanced_unsupervised_group.glayout.addWidget(self.btn_add_features_stack, 2, 2, 1, 2)

            # PCA option for the features
            self.text_features_pca = QtWidgets.QLineEdit()
            self.text_features_pca.setPlaceholderText('e.g. 3 or 5')
            self.text_features_pca.setToolTip('Number of PCA components to use for the features image.\nSet to 0 to disable PCA.')
            self.text_features_pca.setText(self.features_pca_components)
            self.advanced_unsupervised_group.glayout.addWidget(QLabel('PCA components (0 = off)'), 0, 0, 1, 2)
            self.advanced_unsupervised_group.glayout.addWidget(self.text_features_pca, 0, 2, 1, 2)
            # Kmeans option for the features
            self.text_features_kmeans = QtWidgets.QLineEdit()
            self.text_features_kmeans.setPlaceholderText('e.g. 3 or 5')
            self.text_features_kmeans.setToolTip('Number of Kmeans clusters to use for the features image.\nSet to 0 to disable Kmeans.')
            self.text_features_kmeans.setText(self.features_kmeans_clusters)
            self.advanced_unsupervised_group.glayout.addWidget(QLabel('Kmeans clusters (0 = off)'), 1, 0, 1, 2)
            self.advanced_unsupervised_group.glayout.addWidget(self.text_features_kmeans, 1, 2, 1, 2)

        # === FILES/PROJECT TAB (MULTIFILE) ===

        # Add files/project tab and widget
        if 'Files' in self.tab_names or 'Project' in self.tab_names:
            self._on_create_files_project()

        # === CONNECTIONS ===

        # Add connections and initialize by setting default model and params
        self._add_connections()
        if self.image_layer_selection_widget.value is not None:
            self._on_select_layer()
        self._reset_model()

        # === KEY BINDINGS ===

        # Add key bindings
        self.viewer.bind_key('Shift+a', self.toggle_annotation, overwrite=True)
        self.viewer.bind_key('Shift+s', self._on_train, overwrite=True)
        self.viewer.bind_key('Shift+d', self._on_predict, overwrite=True)
        self.viewer.bind_key('Shift+f', self.toggle_prediction, overwrite=True)
        self.viewer.bind_key('Shift+q', lambda event=None: self.set_annot_label_class(1, event), overwrite=True)
        self.viewer.bind_key('Shift+w', lambda event=None: self.set_annot_label_class(2, event), overwrite=True)
        self.viewer.bind_key('Shift+e', lambda event=None: self.set_annot_label_class(3, event), overwrite=True)
        self.viewer.bind_key('Shift+r', lambda event=None: self.set_annot_label_class(4, event), overwrite=True)

### Visibility toggles for key bindings

    def toggle_annotation(self, event=None):
        """Hide/unhide annotations layer."""
        annot_layer = self.annotation_layer_selection_widget.value
        if annot_layer is None:
            return
        if annot_layer.visible == False:
            annot_layer.visible = True
            self.viewer.layers.selection.active = None
            self.viewer.layers.selection.active = annot_layer
        else:
            annot_layer.visible = False

    def toggle_prediction(self, event=None):
        """Hide/unhide prediction layer."""
        if not self.seg_prefix in self.viewer.layers:
            return
        if self.viewer.layers[self.seg_prefix].visible == False:
            self.viewer.layers[self.seg_prefix].visible = True
            self.viewer.layers.selection.active = None
            self.viewer.layers.selection.active = self.viewer.layers[self.seg_prefix]
        else:
            self.viewer.layers[self.seg_prefix].visible = False

    def set_annot_label_class(self, x, event=None):
        """Set the label class of the annotation layer."""
        annot_layer = self.annotation_layer_selection_widget.value
        if annot_layer is not None:
            annot_layer.selected_label = x
            annot_layer.visible = True
            annot_layer.mode = 'paint'
            self.viewer.layers.selection.active = None
            self.viewer.layers.selection.active = annot_layer
        # Also change the selected label for all (not selected) annotation layers and segmentation layers
        labels_layers = self.annot_layers.union(self.seg_layers)
        for l in labels_layers:
            if l is not None and l.name in self.viewer.layers:
                self.viewer.layers[l.name].selected_label = x

### Define the connections between the widget elements

    def _add_connections(self):

        # Reset layer choices in dropdown when layers are renamed; also bind this behaviour to inserted layers
        for layer in self.viewer.layers:
            layer.events.name.connect(self._update_image_layers)
        self.viewer.layers.events.inserted.connect(self._on_insert_layer)

        # === HOME TAB ===

        # Reset layer choices in dropdowns when napari layers are added or removed
        for layer_widget_reset in [self._update_annotation_layers,
                                   self._update_image_layers]:
            self.viewer.layers.events.inserted.connect(layer_widget_reset)
            self.viewer.layers.events.removed.connect(layer_widget_reset)

        # Load/Save
        self.save_model_btn.clicked.connect(self._on_save_model)
        self.load_model_btn.clicked.connect(self._on_load_model)

        # Model
        self._reset_convpaint_btn.clicked.connect(self._on_reset_convpaint)
        
        # Change input layer selection when choosing a new image layer in dropdown
        self.image_layer_selection_widget.native.activated.connect(self._delayed_on_select_layer)
        self.image_layer_selection_widget.changed.connect(self._delayed_on_select_layer)
        self.annotation_layer_selection_widget.native.activated.connect(self._on_select_annot)
        self.annotation_layer_selection_widget.changed.connect(self._on_select_annot)
        self.add_layers_btn.clicked.connect(self._on_add_annot_layer)

        # Image Processing; only trigger from buttons that are activated (checked)
        self.radio_single_channel.toggled.connect(lambda checked: 
            checked and self._on_channel_mode_changed())
        self.radio_multi_channel.toggled.connect(lambda checked:
            checked and self._on_channel_mode_changed())
        self.radio_rgb.toggled.connect(lambda checked:
            checked and self._on_channel_mode_changed())
        self.radio_no_normalize.toggled.connect(lambda checked:
            checked and self._on_norm_changed())
        self.radio_normalize_over_stack.toggled.connect(lambda checked:
            checked and self._on_norm_changed())
        self.radio_normalize_by_image.toggled.connect(lambda checked:
            checked and self._on_norm_changed())

        # Train
        self.train_classifier_btn.clicked.connect(self._on_train)
        self.check_auto_seg.stateChanged.connect(lambda: setattr(
            self, 'auto_seg', self.check_auto_seg.isChecked()))
        # self.check_use_project.stateChanged.connect(self._on_create_files_project)
        # self.train_classifier_on_project_btn.clicked.connect(self._on_train_on_project)

        # Segment
        self.segment_btn.clicked.connect(self._on_predict)
        self.segment_all_btn.clicked.connect(self._on_predict_all)

        # Acceleration
        self.spin_downsample.valueChanged.connect(lambda:
            self.cp_model.set_param('image_downsample', self.spin_downsample.value(), ignore_warnings=True))
        self.spin_smoothen.valueChanged.connect(lambda:
            self.cp_model.set_param('seg_smoothening', self.spin_smoothen.value(), ignore_warnings=True))
        self.check_tile_annotations.stateChanged.connect(lambda: 
            self.cp_model.set_param('tile_annotations', self.check_tile_annotations.isChecked(), ignore_warnings=True))
        self.check_tile_image.stateChanged.connect(lambda: 
            self.cp_model.set_param('tile_image', self.check_tile_image.isChecked(), ignore_warnings=True))

        # === MODEL OPTIONS TAB ===

        # Feature extractor
        self.qcombo_fe_type.currentIndexChanged.connect(self._on_fe_selected)
        self.set_fe_btn.clicked.connect(self._on_set_fe_model)
        self.reset_default_fe_btn.clicked.connect(self._on_reset_default_fe)
        self.fe_layer_selection.itemSelectionChanged.connect(self._on_fe_layer_selection_changed)
        self.fe_scaling_factors.currentIndexChanged.connect(self._on_fe_scalings_changed)
        # NOTE: Changing interpolation_order, use_min_features and use_gpu of FE
        # shall only be applied when the user clicks the button to set the FE
        
        # Classifier
        self.spin_iterations.valueChanged.connect(lambda:
            self.cp_model.set_param('clf_iterations', self.spin_iterations.value(), ignore_warnings=True))
        self.spin_learning_rate.valueChanged.connect(lambda:
            self.cp_model.set_param('clf_learning_rate', self.spin_learning_rate.value(), ignore_warnings=True))
        self.spin_depth.valueChanged.connect(lambda:
            self.cp_model.set_param('clf_depth', self.spin_depth.value(), ignore_warnings=True))
        self.set_default_clf_btn.clicked.connect(self._on_reset_clf_params)

        # === CLASS LABELS TAB ===

        if 'Class labels' in self.tab_names:
            for class_label in self.class_labels:
                class_label.textChanged.connect(self._update_class_labels)
            if self.annotation_layer_selection_widget.value is not None:
                labels_layer = self.annotation_layer_selection_widget.value
                labels_layer.events.colormap.connect(self._on_change_annot_cmap)
            if self.seg_prefix in self.viewer.layers:
                seg_layer = self.viewer.layers[self.seg_prefix]
                seg_layer.events.colormap.connect(self._on_change_seg_cmap)
            self.btn_class_distribution_annot.clicked.connect(lambda: self._on_show_class_distribution(trained_data=False))
        
        # === ADVANCED TAB ===

        if 'Advanced' in self.tab_names:
            self.check_auto_add_layers.stateChanged.connect(lambda: setattr(
                self, 'auto_add_layers', self.check_auto_add_layers.isChecked()))
            self.check_keep_layers.stateChanged.connect(lambda: setattr(
                self, 'keep_layers', self.check_keep_layers.isChecked()))
            self.btn_add_all_annot_layers.clicked.connect(self._on_add_all_annot_layers)
            self.check_auto_select_annot.stateChanged.connect(lambda: setattr(
                self, 'auto_select_annot', self.check_auto_select_annot.isChecked()))

            self.btn_train_on_selected.clicked.connect(self._on_train_on_selected)
            self.radio_single_training.toggled.connect(lambda: setattr(
                self, 'cont_training', 'off'))
            self.radio_img_training.toggled.connect(lambda: setattr(
                self, 'cont_training', "image"))
            self.radio_global_training.toggled.connect(lambda: setattr(
                self, 'cont_training', "global"))
            # self.check_cont_training.stateChanged.connect(lambda: setattr(
            #     self, 'cont_training', self.check_cont_training.isChecked()))
            self.btn_class_distribution_trained.clicked.connect(lambda: self._on_show_class_distribution(trained_data=True))
            self.btn_reset_training.clicked.connect(self._reset_train_features)

            self.check_use_dask.stateChanged.connect(lambda: setattr(
                self, 'use_dask', self.check_use_dask.isChecked()))

            self.text_input_channels.textChanged.connect(lambda: setattr(
                self, 'input_channels', self.text_input_channels.text()))
            
            self.btn_switch_axes.clicked.connect(self._on_switch_axes)

            # Checkboxes for output layers
            self.check_add_seg.stateChanged.connect(lambda: setattr(
                self, 'add_seg', self.check_add_seg.isChecked()))
            self.check_add_probas.stateChanged.connect(lambda: setattr(
                self, 'add_probas', self.check_add_probas.isChecked()))

            # Textboxes for PCA and Kmeans
            self.text_features_pca.textChanged.connect(lambda: setattr(
                self, 'features_pca_components', self.text_features_pca.text()))
            self.text_features_kmeans.textChanged.connect(lambda: setattr(
                self, 'features_kmeans_clusters', self.text_features_kmeans.text()))
            # Button to add features for the current plane
            self.btn_add_features.clicked.connect(self._on_get_feature_image)
            # Button to add features for stack (all planes)
            self.btn_add_features_stack.clicked.connect(self._on_get_feature_image_all)

### Define the behaviour in the class labels tab

    # Cass Labels

    def _create_default_class_labels(self):
        """Create the default class labels and icons in the layout."""
        # Start with default class labels
        for label in self.initial_labels:
            self._on_add_class_label(text=label)
        # Add default annot and seg layers if they exist
        if self.annotation_layer_selection_widget.value is not None:
            self.annot_layers.add(self.annotation_layer_selection_widget.value)
        if self.seg_prefix in self.viewer.layers:
            self.seg_layers.add(self.viewer.layers[self.seg_prefix])
        # Update all class labels and icons
        self.update_all_labels_and_cmaps()

    def _on_reset_class_labels(self):
        """Reset the class labels to the default ones and update all annotation and segmentation layers."""

        # Remove and delete all class label widgets and icons
        for label in self.class_labels:
            self.class_labels_layout.removeWidget(label)
            label.deleteLater()
        for icon in self.class_icons:
            self.class_labels_layout.removeWidget(icon)
            icon.deleteLater()

        self.class_labels.clear()
        self.class_icons.clear()

        # Remove the buttons from the layout
        self.class_labels_layout.removeWidget(self.add_class_btn)
        self.class_labels_layout.removeWidget(self.remove_class_btn)
        self.class_labels_layout.removeWidget(self.reset_class_btn)
        self.class_labels_layout.removeWidget(self.btn_class_distribution_annot)

        # Recreate the default class labels and icons
        self._create_default_class_labels()

        # Re-add the buttons below the class labels
        self.class_labels_layout.addWidget(self.add_class_btn, len(self.class_labels)+1, 0, 1, 5)
        self.class_labels_layout.addWidget(self.remove_class_btn, len(self.class_labels)+1, 5, 1, 5)
        self.class_labels_layout.addWidget(self.reset_class_btn, len(self.class_labels)+2, 0, 1, 10)
        self.class_labels_layout.addWidget(self.btn_class_distribution_annot, len(self.class_labels)+3, 0, 1, 10)
    
    def _on_add_class_label(self, text=None):
        """Add a new class label and icon to the layout and update all annotation and segmentation layers."""
        
        # Create a new class label
        new_label = QtWidgets.QLineEdit()
        self.class_labels.append(new_label)
        class_num = len(self.class_labels)  # Class number is the length of the list
        # Add the new label to the layout
        self.class_labels_layout.addWidget(new_label, class_num, 1, 1, 9)
        # Set the text of the new label
        text_str = text if text is not None else f'Class {class_num}'
        new_label.setText(text_str)

        # Change "clear" button to the last label and connect it to deleting the entire entry (instead of only text)
        # new_label.setClearButtonEnabled(True)
        # new_label.textChanged.connect(self.remove_class_label)
        # self.class_labels[-2].setClearButtonEnabled(False)
        
        # Connect the new label to the update function
        new_label.textChanged.connect(self._update_class_labels)
        
        # Add a new icon
        new_icon = QtWidgets.QLabel()
        self.class_icons.append(new_icon)
        self.class_labels_layout.addWidget(new_icon, class_num, 0)
        new_icon.mousePressEvent = lambda event: self._set_all_labels_classes(class_num, event)
        
        # Update the icon with the color of the last label and all class labels
        self._update_class_icons(class_num)
        self._update_class_labels()

        # Move the add, remove and reset buttons one down
        self.class_labels_layout.removeWidget(self.add_class_btn)
        self.class_labels_layout.removeWidget(self.remove_class_btn)
        self.class_labels_layout.removeWidget(self.reset_class_btn)
        self.class_labels_layout.removeWidget(self.btn_class_distribution_annot)
        self.class_labels_layout.addWidget(self.add_class_btn, class_num+1, 0, 1, 5)
        self.class_labels_layout.addWidget(self.remove_class_btn, class_num+1, 5, 1, 5)
        self.class_labels_layout.addWidget(self.reset_class_btn, class_num+2, 0, 1, 10)
        self.class_labels_layout.addWidget(self.btn_class_distribution_annot, class_num+3, 0, 1, 10)

    def _on_remove_class_label(self, del_annots=True, event=None):
        """Remove the last class label and icon from the layout and update all annotation and segmentation layers."""
        last_label_idx = len(self.class_labels)
        if last_label_idx > 2:
            # Remove the annotations from all annotation layers (do NOT do it in segmentation layers, as this would leave holes)
            if del_annots:
                for layer in self.annot_layers:
                    if layer is not None and layer.name in self.viewer.layers:
                        # Get the annotations image and remove the last label from it
                        label_img = layer.data
                        label_img[label_img == last_label_idx] = 0
                        # Update the layer to show changes immediately
                        layer.refresh()
            # Remove the last label and icon from the layout
            self.class_labels[-1].deleteLater()
            self.class_icons[-1].deleteLater()
            self.class_labels.pop()
            self.class_icons.pop()
            # Move the buttons one up
            self.class_labels_layout.removeWidget(self.add_class_btn)
            self.class_labels_layout.removeWidget(self.remove_class_btn)
            self.class_labels_layout.removeWidget(self.reset_class_btn)
            self.class_labels_layout.removeWidget(self.btn_class_distribution_annot)
            self.class_labels_layout.addWidget(self.add_class_btn, len(self.class_labels)+1, 0, 1, 5)
            self.class_labels_layout.addWidget(self.remove_class_btn, len(self.class_labels)+1, 5, 1, 5)
            self.class_labels_layout.addWidget(self.reset_class_btn, len(self.class_labels)+2, 0, 1, 10)
            self.class_labels_layout.addWidget(self.btn_class_distribution_annot, len(self.class_labels)+3, 0, 1, 10)
            # Update the icons and class labels
            self._update_class_labels()
        else:
            show_info('You need at least two classes.')

    def _update_class_icons(self, class_num=None, event=None):
        """Update the class icons with the colors of the class labels.
        If class_num is given, only update the icon of that class."""

        if self.labels_cmap is None:
            return

        cmap = self.labels_cmap.copy()

        if class_num is not None:
            col = cmap.map(class_num)
            pixmap = self.get_pixmap(col)
            self.class_icons[class_num-1].setPixmap(pixmap)
            self.class_icons[class_num-1].mousePressEvent = lambda event: self._set_all_labels_classes(class_num, event)

        # Update all icons with the colors of the class labels
        else:
            for i, _ in enumerate(self.class_labels):
                cl = i+1
                col = cmap.map(cl)
                pixmap = self.get_pixmap(col)
                self.class_icons[i].setPixmap(pixmap)
                # Bind clicking on the icon to selecting the label
                # self.class_icons[i].mousePressEvent = lambda event, idx=i: self._set_all_labels_classes(idx+1, event)

    def _set_all_labels_classes(self, x, event=None):
        """Set the selected label of all annotation and segmentation layers to x."""
        # For all annotations and segmentation layers added previously, set the selected label to x
        labels_layers = self.annot_layers.union(self.seg_layers)
        for l in labels_layers:
            if l is not None and l.name in self.viewer.layers:
                self.viewer.layers[l.name].selected_label = x

    def _update_class_labels(self, event=None):
        """Update the class labels for all annotation and segmentation layers."""
        # For all annotation and segmentation layers, set the class labels (= layer property) to the ones defined in the widget
        label_names = ["No label"] + [label.text() for label in self.class_labels]
        props = {"Class": label_names}
        labels_layers = self.annot_layers.union(self.seg_layers)
        for l in labels_layers:
            if l is not None and l.name in self.viewer.layers:
                self.viewer.layers[l.name].properties = props

    @staticmethod
    def get_pixmap(color):
        """Convert a color (array or list) to a QPixmap for displaying as icon."""
        if color.ndim == 2:
            color = color[0]
        # If color is float, convert to uint8
        if color.dtype != np.uint8:
            color = (color * 255).astype(np.uint8)
        # Create a QColor from the color array
        qcol = QtGui.QColor(*color)
        # Create a QPixmap and fill it with the color
        pixmap = QtGui.QPixmap(20, 20)  # Create a QPixmap of size 20x20
        pixmap.fill(qcol)  # Fill the pixmap with the chosen color
        # icon = QtGui.QIcon(pixmap)  # Convert pixmap to QIcon
        return pixmap
    
    # Synchronization of colormaps

    def _update_cmaps(self, source_layer=None):
        """Update class icons and segmentation colormap when annotation colormap changes."""

        # Avoid infinite loop when changing colormaps
        if self.cmap_flag:
            return
        
        # If there is no cmap yet, define it from the first annotation layer if available
        if self.labels_cmap is None:
            labels_layers = self.annot_layers.union(self.seg_layers)
            if labels_layers:
                # Use the first layer's colormap as the default
                source_layer = next(iter(labels_layers))
            else:
                # If no layers are available, return
                return
        
        # Determine if a source layer is given and update the colormap if so
        if source_layer is not None:
            self.labels_cmap = source_layer.colormap.copy()

        # Make sure we are not creating a loop
        self.cmap_flag = True

        # Get the new colormap and apply to all other label layers
        new_cmap = self.labels_cmap.copy()
        labels_layers = self.annot_layers.union(self.seg_layers)
        for layer in labels_layers:
            if source_layer is None or layer != source_layer:
                layer.colormap = new_cmap

        self._update_class_icons()

        # Turn off the flag to allow for new events
        self.cmap_flag = False

    def _connect_all_cmaps(self):
        """Connect colormap changes for all annotation layers."""
        labels_layers = self.annot_layers.union(self.seg_layers)
        for l in labels_layers:
            l.events.colormap.connect(lambda event: self._update_cmaps(source_layer=event.source))

    def _on_change_annot_cmap(self, event=None):
        """Update class icons and segmentation colormap when annotation colormap changes."""
        if (self.cmap_flag or # Avoid infinite loop
            (event is not None and event.source != self.annotation_layer_selection_widget.value)): # Only triggers of annotation layer
            return
        # Make sure we are not in a loop
        self.cmap_flag = True
        # Update the colormap of all labels layers according to the changed annotation layer
        if self.annotation_layer_selection_widget is not None:
            self._update_cmaps(source_layer=self.annotation_layer_selection_widget.value)
        # Turn off the flag to allow for new events
        self.cmap_flag = False

    def _on_change_seg_cmap(self, event=None):
        """Update annotation colormap when segmentation colormap changes."""
        if (self.cmap_flag or # Avoid infinite loop
            (event is not None and event.source != self.viewer.layers[self.seg_prefix])): # Only triggers of segmentation layer
            return
        # Make sure we are not in a loop
        self.cmap_flag = True
        # Update the colormap of all labels layers according to the segmentation layer
        if self.seg_prefix in self.viewer.layers:
            self._update_cmaps(source_layer=self.viewer.layers[self.seg_prefix])
        # Turn off the flag to allow for new events
        self.cmap_flag = False

    def update_all_labels_and_cmaps(self):
        """Update class labels, icons and colormaps for all annotation and segmentation layers."""
        # Update class labels
        self._update_class_labels()
        # Sync all labels layers (annotations and segmentation) between each other
        self._connect_all_cmaps()
        # Sync the new labels' cmaps with the existing ones
        self._update_cmaps() # Also calls _update_class_icons()
        # Update the class icons
        # self._update_class_icons()

### Add the Files/Project tab

    def _on_create_files_project(self, event=None):
        """Add widget for multi-image project management if not already added."""

        from napari_annotation_project.project_widget import ProjectWidget
        self.project_widget = ProjectWidget(napari_viewer=self.viewer)

        tab_name = 'Files'

        # Add the project widget to a new tab
        # self.tabs.add_named_tab(tab_name, self.project_widget)
        self.tabs.add_named_tab(tab_name, self.project_widget.file_list)
        self.tabs.add_named_tab(tab_name, self.project_widget.btn_add_file)
        self.tabs.add_named_tab(tab_name, self.project_widget.btn_remove_file)
        self.tabs.add_named_tab(tab_name, self.project_widget.btn_save_annotation)
        self.tabs.add_named_tab(tab_name, self.project_widget.btn_load_project)

        # Add the train on project button to the tab
        self.train_classifier_on_project_btn = QPushButton('Train on Files')
        self.train_classifier_on_project_btn.setToolTip('Train on all images loaded in Files tab')
        self.tabs.add_named_tab(tab_name, self.train_classifier_on_project_btn)
        self.train_classifier_on_project_btn.clicked.connect(self._on_train_on_project)
            
        #     self.tabs.setTabEnabled(self.tabs.tab_names.index(tab_name), True)
        #     self.train_classifier_on_project_btn.setEnabled(True)
        # else:
        #     self.tabs.setTabEnabled(self.tabs.tab_names.index(tab_name), False)
        #     self.train_classifier_on_project_btn.setEnabled(False)


### Define the detailed behaviour of the widget

### HOME TAB

    # Handling of Napari layers

    def _on_insert_layer(self, event=None):
        """Bind the update of layer choices in dropdowns to the renaming of inserted layers."""
        layer = event.value
        layer.events.name.connect(self._update_image_layers)
        layer.events.name.connect(self._update_annotation_layers)
        layer.events.data.connect(self._on_select_layer)

    # Layer selection

    def _on_select_layer(self, newtext=None):
        """Assign the layer to segment and update data radio buttons accordingly"""

        # Check if it is the same layer
        was_same = ((self.selected_channel is not None) and
                    (self.selected_channel == self.image_layer_selection_widget.native.currentText()))

        # Update the selected channel and reset the radio buttons, but only if it is a new image
        initial_add_layers_flag = self.add_layers_flag # Save initial state of add_layers_flag
        self.selected_channel = self.image_layer_selection_widget.native.currentText()
        img = self._get_selected_img()

        if img is not None:
            # If it is a new image, set the according radio buttons, image stats etc.
            if self.data_shape is None or self.data_shape != img.data.shape or not was_same:
                # self.update_layer_flag = False
                self.add_layers_flag = False # Turn off layer creation, instead we do it manually below
                # Set radio buttons depending on selected image type
                self._reset_radio_channel_mode_choices()
                self._reset_radio_norm_choices()
                self._reset_predict_buttons()
                self._compute_image_stats(img)
                # Add empty layers and save old data tag for next addition
                if self.auto_add_layers and not was_same:
                    self._add_empty_annot()
                # Set flags that new outputs need to be generated (and not planes of the old ones populated)
                self.new_seg = True
                self.new_proba = True
                self.new_features = True
                # Activate button to add annotation and segmentation layers
                self.add_layers_btn.setEnabled(True)
                # Give info if image is very large to use "tile image"
                if self._check_large_image(img) and not self.cp_model.get_param('tile_image'):
                    show_info('Image is very large. Consider using tiling and/or downsampling.')
                # If we have continuous training within a single image, reset the training features
                if self.cont_training == "image" or self.cont_training == "off":
                    self._reset_train_features()
                self.update_layer_flag = True
                self.data_shape = img.data.shape
        else:
            self.add_layers_btn.setEnabled(False)

        # If the option is activated, select annotation layer according to the prefix and image name
        if self.auto_select_annot and not getattr(self, "_block_layer_select", False):
            self._auto_select_annot_layer()

        # Allow other methods again to add layers if that was the case before
        self.add_layers_flag = initial_add_layers_flag

        # Check if the selected annotations layer is suited, if one is selected
        labels_layer = self.annotation_layer_selection_widget.value
        if labels_layer is not None and not self._approve_annotation_layer_shape(labels_layer, img):
            warnings.warn('Annotation layer has wrong shape for the selected data')

    def _delayed_on_select_layer(self, event=None):
        """Delay the selection of the image layer to allow for napari operations to happen first."""
        print("Doing delayed on select layer")
        self._block_layer_select = False
        QTimer.singleShot(100, lambda: self._on_select_layer())
        # Only set the block flag again after some time, so auto selection is triggered, but only once
        QTimer.singleShot(200, lambda: setattr(self, "_block_layer_select", True))

    def _auto_select_annot_layer(self):
        """Automatically select an annotation layer according to the prefix and image name."""

        # Set the name of the annotations accordingly
        annot_name = f"{self.annot_prefix}_{self.selected_channel}"
        # Check if there are fitting labels layers present in the viewer
        found_layers = [layer.name for layer in self.viewer.layers # Take all layers
                        if isinstance(layer, napari.layers.Labels) # that are labels layers
                        and layer.name[:len(annot_name)] == annot_name] # and start with the prefix
        if self.annot_prefix in self.viewer.layers and isinstance(self.viewer.layers[self.annot_prefix], napari.layers.Labels):
            # If there is a layer with the name "annotations", set it as the annotation layer
            annot_name = self.annot_prefix
        elif len(found_layers) > 0:
            if len(found_layers) > 1:
                show_info('Multiple annotation layers found. The first one (alphabetically) is chosen.')
            # Sort and take first one
            found_layers.sort()
            annot_name = found_layers[0]
        else:
            show_info('No annotation layer found. Please create one.')
            return

        self.viewer.layers.selection.active = self.viewer.layers[annot_name]
        self.annotation_layer_selection_widget.value = self.viewer.layers[annot_name]
        self.viewer.layers[annot_name].mode = 'paint'
        self.viewer.layers[annot_name].brush_size = self.default_brush_size
        self._on_select_annot()

        # Hide all layers except the image and the annotation layer
        for layer in self.viewer.layers:
            if layer.name != self.selected_channel and layer.name != annot_name:
                layer.visible = False
            else:
                layer.visible = True

    def _on_select_annot(self, newtext=None):
        """Check if annotation layer dimensions are compatible with image, and raise warning if not."""
        
        labels_layer = self.annotation_layer_selection_widget.value
        
        # Handle labels classes
        self.annot_layers.add(self.annotation_layer_selection_widget.value)
        # labels_layer.events.colormap.connect(self._on_change_annot_cmap)
        self.update_all_labels_and_cmaps()
        
        # Check shape
        if self.image_layer_selection_widget.value is not None:
            img = self.image_layer_selection_widget.value
            if not self._approve_annotation_layer_shape(labels_layer, img):
                warnings.warn('Annotation layer has wrong shape for the selected data')

    def _on_add_annot_layer(self, event=None, force_add=True):
        """Add empty annotation and segmentation layers if not already present."""
        self._add_empty_annot(event, force_add)
        img = self._get_selected_img(check=True)
        if img is None:
            return
        labels_layer = self.annotation_layer_selection_widget.value
        labels_layer.events.colormap.connect(self._on_change_annot_cmap)
        self.update_all_labels_and_cmaps()

    # Train

    def _on_train(self, event=None):
        """Given a set of new annotations, update the CatBoost classifier."""

        # Get the data
        img = self._get_selected_img(check=True)
        annot = self.annotation_layer_selection_widget.value
        mem_mode = (self.cont_training == "image"
                    or self.cont_training == "global")

        # Check if annotations of at least 2 classes are present
        if annot is None:
            raise Exception('No annotation layer selected. Please create one.')
        unique_labels = np.unique(annot.data)
        unique_labels = unique_labels[unique_labels != 0]
        if len(unique_labels) < 2:
            if not mem_mode:
                raise Exception('You need annotations for at least foreground and background')
            if self.cp_model.num_trainings == 0:
                raise Exception('Model has not yet been trained. You need annotations for at least foreground and background')

        # Check if annotations layer has correct shape for the chosen data type
        if not self._approve_annotation_layer_shape(annot, img):
            raise Exception('Annotation layer has wrong shape for the chosen data')

        # Set the current model path to 'in training' and adjust the model description
        self.current_model_path = 'in training'
        self._set_model_description()

        # Get the image data and normalize it; also get the annotations
        image_stack_norm = self._get_data_channel_first_norm(img)
        annot = annot.data
        
        # Start training
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(True)

        with progress(total=0) as pbr:
            pbr.set_description(f"Training")
            img_name = self._get_selected_img().name
            in_channels = self._parse_in_channels(self.input_channels)
            # Train the model with the current image and annotations; skip normalization as it is done in the widget
            _ = self.cp_model.train(image_stack_norm, annot, memory_mode=mem_mode, img_ids=img_name, in_channels=in_channels, skip_norm=True)
            self._update_training_counts()
    
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(False)

        # Set the current model path to 'trained, unsaved' and adjust the model description
        self.current_model_path = 'trained, unsaved'
        self.trained = True
        self._reset_predict_buttons()
        self._set_model_description()

        # Automatically segment the image if the option is activated
        if self.auto_seg:
            self._on_predict()

    def _on_train_on_project(self):
        """Train classifier on all annotations in project.
        !!!! Need to double-check if normalization is done correctly for projects !!!!"""

        num_files = len(self.project_widget.params.file_paths)
        if num_files == 0:
            raise Exception('No files found')

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(True)
        
        self.viewer.layers.events.removed.disconnect(self._on_reset_convpaint)
        with progress(total=0) as pbr:
            pbr.set_description(f"Training")
            self.current_model_path = 'in training'
            # all_features, all_targets = [], []
            all_imgs, all_annots = [], []
            in_channels = self._parse_in_channels(self.input_channels)
            for ind in range(num_files):
                self.project_widget.file_list.setCurrentRow(ind)
                image_stack_norm = self._get_data_channel_first_norm()
                annots = self.annotation_layer_selection_widget.value.data
                all_imgs.append(image_stack_norm)
                all_annots.append(annots)
                # in_channels = self._parse_in_channels(self.input_channels)
                # features, targets = self.cp_model.get_features_current_layers(image_stack_norm, annots, in_channels=in_channels)
                # if features is None:
                    # continue
                # all_features.append(features)
                # all_targets.append(targets)
            
            self.cp_model.train(all_imgs, all_annots, in_channels=in_channels, skip_norm=True)
            # all_features = np.concatenate(all_features, axis=0)
            # all_targets = np.concatenate(all_targets, axis=0)

            # self.cp_model._clf_train(all_features, all_targets)

            self.current_model_path = 'trained, unsaved'
            self.trained = True
            self._reset_predict_buttons()
            # self.save_model_btn.setEnabled(True)
            self._set_model_description()

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(False)

        self.viewer.layers.events.removed.connect(self._on_reset_convpaint)

    # Predict

    def _on_predict(self, event=None):
        """Predict the segmentation of the currently viewed frame based
        on a classifier trained with annotations."""

        if not (self.add_seg or self.add_probas):
            warnings.warn('Neither segmentation nor probabilities output selected to be added. Nothing to do.')
            return

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(True)

        with progress(total=0) as pbr:
            pbr.set_description(f"Prediction")
            # Get the data
            image_plane = self._get_current_plane_norm()
            # Predict image (use backend function which returns probabilities and segmentation); skip norm as it is done above
            in_channels = self._parse_in_channels(self.input_channels)
            probas, segmentation = self.cp_model._predict(image_plane, add_seg=True, in_channels=in_channels, skip_norm=True, use_dask=self.use_dask)

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(False)

        # Get the current step in case of stacks
        data_dims = self._get_data_dims(self._get_selected_img())
        step = self.viewer.dims.current_step[-3] if data_dims in ['3D_single', '4D', '3D_RGB'] else None

        # Add segmentation layer if enabled
        if self.add_seg:
            # Check if we need to create a new segmentation layer
            self._check_create_segmentation_layer()
            # Set the flag to False, so we don't create a new layer every time
            self.new_seg = False

            # Update segmentation layer
            if data_dims in ['2D', '2D_RGB', '3D_multi']:
                self.viewer.layers[self.seg_prefix].data = segmentation
            else: # 3D_single, 4D, 3D_RGB
                # seg has no channel dim -> z is first
                self.viewer.layers[self.seg_prefix].data[step] = segmentation
            self.viewer.layers[self.seg_prefix].refresh()

        # Add probabilities if enabled
        if self.add_probas:
            # Check if we need to create a new probabilities layer
            num_classes = probas.shape[:1]
            self._check_create_probas_layer(num_classes)
            # Set the flag to False, so we don't create a new layer every time
            self.new_proba = False

            # Update probabilities layer
            if data_dims in ['2D', '2D_RGB', '3D_multi']: # No stack dim
                self.viewer.layers[self.proba_prefix].data = probas
            else: # 3D_single, 4D, 3D_RGB (stack dim is second, probas first)
                self.viewer.layers[self.proba_prefix].data[:, step] = probas
            self.viewer.layers[self.proba_prefix].refresh()

    def _on_get_feature_image(self, event=None):
        """Get the feature image for the currently viewed frame based
        on the current feature extractor and show it in a new layer."""

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(True)

        with progress(total=0) as pbr:
            pbr.set_description(f"Feature extraction") 

            # Get the data
            image_plane = self._get_current_plane_norm()
            in_channels = self._parse_in_channels(self.input_channels)

            # Check and parse PCA and Kmeans parameters
            pca, kmeans = self._check_parse_pca_kmeans()

            # Get feature image; skip norm as it is done above
            feature_image = self.cp_model.get_feature_image(image_plane, in_channels=in_channels, skip_norm=True,
                                                            pca_components=pca,
                                                            kmeans_clusters=kmeans)

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(False)

        # Update features layer
        # Check if we need to create a new features layer
        num_features = feature_image.shape[0] if not kmeans else 0
        self._check_create_features_layer(num_features)
        # Set the flag to False, so we don't create a new layer every time
        self.new_features = False

        data_dims = self._get_data_dims(self._get_selected_img())
        if data_dims in ['2D', '2D_RGB', '3D_multi']: # No stack dim
            self.viewer.layers[self.features_prefix].data = feature_image
        else: # 3D_single, 4D, 3D_RGB (stack dim is third last)
            step = self.viewer.dims.current_step[-3]
            self.viewer.layers[self.features_prefix].data[..., step, :, :] = feature_image
        self.viewer.layers[self.features_prefix].refresh()

    def _on_predict_all(self):
        """Predict the segmentation of all frames based 
        on a classifier model trained with annotations."""
        
        img = self._get_selected_img(check=True)
        data_dims = self._get_data_dims(img)
        if data_dims not in ['3D_single', '3D_RGB', '4D']:
            raise Exception(f'Image stack has wrong dimensionality {data_dims}')
        
        # Create the segmentation layer if it is not already present
        # (NOTE: probabilities layer is created in the prediction loop, as we need to know the number of classes)
        if self.add_seg:
            self._check_create_segmentation_layer()
            # Set the flag to False, so we don't create a new layer every time
            self.new_seg = False

        # Start prediction
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(True)

        # Get normalized stack data (entire stack, and stats prepared given the radio buttons)
        image_stack_norm = self._get_data_channel_first_norm(img) # Normalize the entire stack
        
        # Step through the stack and predict each image
        num_steps = image_stack_norm.shape[-3]
        for step in progress(range(num_steps)):

            # Take the slice of the 3rd last dimension (since images are C, Z, H, W or Z, H, W)
            image = image_stack_norm[..., step, :, :]

            # Predict the current step; skip normalization as it is done above
            in_channels = self._parse_in_channels(self.input_channels)
            # Use the backend function which returns probabilities and segmentation
            probas, seg = self.cp_model._predict(image, add_seg=True, in_channels=in_channels, skip_norm=True, use_dask=self.use_dask)

            # In the first iteration, check if we need to create a new probas layer
            # (we need the information about the number of classes)
            if step == 0 and self.add_probas:
                num_classes = probas.shape[0]
                # Check if we need to create a new probabilities layer
                self._check_create_probas_layer(num_classes)
                # Set the flag to False, so we don't create a new layer every time
                self.new_proba = False

            # Add the slices to the segmentation and probabilities layers
            if self.add_seg:
                self.viewer.layers[self.seg_prefix].data[step] = seg
                self.viewer.layers[self.seg_prefix].refresh()
            if self.add_probas:
                self.viewer.layers[self.proba_prefix].data[..., step, :, :] = probas
                self.viewer.layers[self.proba_prefix].refresh()

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(False)

    def _on_get_feature_image_all(self):
        """Get the feature image for all frames based
        on the current feature extractor and show it in a new layer."""

        # Get the data
        img = self._get_selected_img(check=True)
        data_dims = self._get_data_dims(img)
        if data_dims not in ['3D_single', '3D_RGB', '4D']:
            raise Exception(f'Image stack has wrong dimensionality {data_dims}')

        # Start feature extraction
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(True)

        # Get normalized stack data (entire stack, and stats prepared given the radio buttons)
        image_stack_norm = self._get_data_channel_first_norm(img) # Normalize the entire stack
        pca, kmeans = self._check_parse_pca_kmeans()
        in_channels = self._parse_in_channels(self.input_channels)

        if kmeans:
            # Get feature image for entire stack; skip norm as it is done above
            feature_image = self.cp_model.get_feature_image(image_stack_norm, in_channels=in_channels, skip_norm=True,
                                                                pca_components=pca,
                                                                kmeans_clusters=kmeans)

            # Check if we need to create a new features layer
            # num_features = feature_image.shape[0] if not kmeans else 0
            # self._check_create_features_layer(num_features)
            self._check_create_features_layer(0)
            # Set the flag to False, so we don't create a new layer every time
            self.new_features = False
            # Update features layer
            self.viewer.layers[self.features_prefix].data = feature_image

        else: # No kmeans, can do step-by-step to save memory (and show progress)
            # Step through the stack and predict each image
            num_steps = image_stack_norm.shape[-3]
            for step in progress(range(num_steps)):

                # Take the slice of the 3rd last dimension (since images are C, Z, H, W or Z, H, W)
                image = image_stack_norm[..., step, :, :]

                # Predict the current step; skip normalization as it is done above
                # Get feature image; skip norm as it is done above
                feature_image = self.cp_model.get_feature_image(image, in_channels=in_channels, skip_norm=True,
                                                                pca_components=pca,
                                                                kmeans_clusters=kmeans)

                # In the first iteration, check if we need to create a new features layer
                # (we need the information about the number of classes)
                if step == 0:
                    # Check if we need to create a new features layer
                    num_features = feature_image.shape[0] if not kmeans else 0
                    self._check_create_features_layer(num_features)
                    # Set the flag to False, so we don't create a new layer every time
                    self.new_features = False

                # Add the slices to the segmentation and probabilities layers
                # if kmeans:
                #     self.viewer.layers[self.features_prefix].data[step] = feature_image
                #     self.viewer.layers[self.features_prefix].refresh()
                # else:
                self.viewer.layers[self.features_prefix].data[..., step, :, :] = feature_image
                self.viewer.layers[self.features_prefix].refresh()

            with warnings.catch_warnings():
                warnings.simplefilter(action="ignore", category=FutureWarning)
                self.viewer.window._status_bar._toggle_activity_dock(False)


    # Load/Save

    def _on_save_model(self, event=None, save_file=None):
        """Select file where to save the classifier along with the model parameters."""

        # Get file path
        if save_file is None:
            dialog = QFileDialog()
            save_file, _ = dialog.getSaveFileName(self, "Save model", None, "PICKLE (*.pkl);;YAML (*.yml)")
        
        # If file selection is aborted, raise a warning (instead of an error)
        if save_file == '':
            warnings.warn('No file selected')
            return

        # Save model
        save_file = Path(save_file)
        suff = save_file.suffix
        pkl = suff == '.pkl'
        yml = suff == '.yml'
        save_string = str(save_file)
        self.cp_model.save(save_string, create_pkl=pkl, create_yml=yml)

        # Adjust the model description
        self.current_model_path = save_file.name
        self._set_model_description()

    def _on_load_model(self, event=None, save_file=None):
        """Select file to load classifier along with the model parameters."""

        # Get file path and open the data
        if save_file is None:
            dialog = QFileDialog()
            save_file, _ = dialog.getOpenFileName(self, "Choose model", None, "PICKLE (*.pkl);;YAML (*.yml)")

        # If file selection is aborted, raise a warning (instead of an error)
        if save_file == '':
            warnings.warn('No file selected')
            return
        
        save_file = Path(save_file)
        
        # Load the model parameters
        save_string = str(save_file)
        new_model = ConvpaintModel(model_path=save_string)
        new_param = new_model.get_params()
        
        # Check if the new multichannel setting is compatible with data
        data = self._get_selected_img()
        data_dims = self._get_data_dims(data)
        channel_mode = new_param.channel_mode
        if data_dims in ['2D'] and channel_mode in ['rgb', 'multi']:
            warnings.warn(f'The loaded model works with {channel_mode} data, but the data is {data_dims}. ' +
                            'This might cause problems.')
        # Check if the loaded normalization setting is compatible with data
        if data_dims in ['2D', '2D_RGB', '3D_multi'] and new_param.normalize == 2:
            warnings.warn(f'The loaded model normalizes over stack, but the data is {data_dims}. ' +
                            'This might cause problems.')

        # Update GUI with the new parameters
        self._update_gui_from_params(new_param)

        # Update GUI to show selectable layers of model chosen from drop-down
        self._update_gui_fe_layers_from_model(new_model)

        # Load the model (Note: done after updating GUI, since GUI updates might reset clf or change model)
        self.cp_model = new_model
        self.cp_model._param = new_param
        self.temp_fe_model = ConvpaintModel.create_fe(new_param.fe_name,
                                                      new_param.fe_use_gpu)

        # Adjust trained flag, save button, predict buttons etc., and update model description
        self.trained = save_file.suffix == '.pkl' and new_model.classifier is not None
        # self.save_model_btn.setEnabled(True)
        self._reset_predict_buttons()
        self.current_model_path = save_file.name
        self._set_model_description()
        self._update_training_counts()

    # Image Processing

    def _on_channel_mode_changed(self):
        """Set the image data dimensions based on radio buttons,
        reset classifier, and adjust normalization options."""
        channel_mode = "multi" if self.radio_multi_channel.isChecked() else \
                       "rgb" if self.radio_rgb.isChecked() else \
                       "single"
        self.cp_model.set_param("channel_mode", channel_mode, ignore_warnings=True)
        self._reset_clf()
        self._reset_radio_norm_choices()
        if (self.add_layers_flag # Add layers only if not triggered from layer selection
            and self.auto_add_layers): # and if auto_add_layers is checked
            self._add_empty_annot()
        # Reset the features for continuous training
        self._reset_train_features()
        # Set flags that new outputs need to be generated (and not planes of the old ones populated)
        self.new_seg = True
        self.new_proba = True
        self.new_features = True

    def _on_norm_changed(self):
        """Set the normalization options based on radio buttons,
        and update stats."""
        if self.radio_no_normalize.isChecked():
            self.cp_model.set_param("normalize", 1, ignore_warnings=True)
        elif self.radio_normalize_over_stack.isChecked():
            self.cp_model.set_param("normalize", 2, ignore_warnings=True)
        elif self.radio_normalize_by_image.isChecked():
            self.cp_model.set_param("normalize", 3, ignore_warnings=True)
        # self._compute_image_stats() # NOTE: Don't we want to set it right away?
        self.image_mean, self.image_std = None, None
        self._reset_clf()
        # Reset the features for continuous training
        self._reset_train_features()

    # Model reset

    def _on_reset_convpaint(self, event=None):
        """Reset model to default and update GUI."""

        # Turn off automatic layer creation for the model reset
        self.add_layers_flag = False

        # Remove class labels (note, resetting of class labels needs to be split because of the handling of the attributes)
        if 'Class labels' in self.tab_names:
            for label in self.class_labels:
                self.class_labels_layout.removeWidget(label)
                label.deleteLater()
            for icon in self.class_icons:
                self.class_labels_layout.removeWidget(icon)
                icon.deleteLater()

        # Reset the model to default
        self._reset_model()
        # Reset the widget attributes and the according buttons
        self._reset_attributes()
        # Adjust the buttons in the widget
        self.check_auto_seg.setChecked(self.auto_seg) 
        self.check_auto_add_layers.setChecked(self.auto_add_layers)
        self.check_keep_layers.setChecked(self.keep_layers)
        self.check_auto_select_annot.setChecked(self.auto_select_annot)
        {"off": lambda: self.radio_single_training.setChecked(True),
         "image": lambda: self.radio_img_training.setChecked(True),
         "global": lambda: self.radio_global_training.setChecked(True)}[self.cont_training]()
        # self.check_cont_training.setChecked(self.cont_training)
        self.check_use_dask.setChecked(self.use_dask)
        self.text_input_channels.setText(self.input_channels)
        self.check_add_seg.setChecked(self.add_seg)
        self.check_add_probas.setChecked(self.add_probas)
        self.text_features_pca.setText(self.features_pca_components)
        self.text_features_kmeans.setText(self.features_kmeans_clusters)
        # Reset the model description
        self._set_model_description()

        if 'Class labels' in self.tab_names:
            # Remove the buttons from the layout
            self.class_labels_layout.removeWidget(self.add_class_btn)
            self.class_labels_layout.removeWidget(self.remove_class_btn)
            self.class_labels_layout.removeWidget(self.reset_class_btn)
            self.class_labels_layout.removeWidget(self.btn_class_distribution_annot)

            # Recreate the default class labels and icons
            self._create_default_class_labels()

            # Re-add the buttons below the class labels
            self.class_labels_layout.addWidget(self.add_class_btn, len(self.class_labels)+1, 0, 1, 5)
            self.class_labels_layout.addWidget(self.remove_class_btn, len(self.class_labels)+1, 5, 1, 5)
            self.class_labels_layout.addWidget(self.reset_class_btn, len(self.class_labels)+2, 0, 1, 10)
            self.class_labels_layout.addWidget(self.btn_class_distribution_annot, len(self.class_labels)+3, 0, 1, 10)

        # Turn on layer creation again
        self.add_layers_flag = True

    def _reset_model(self):
        """Reset the model to default."""
        # self.save_model_btn.setEnabled(True)
        self._reset_fe_params() # Also resets the FE model
        self._reset_clf_params()
        self._reset_clf()
        self._reset_default_general_params()
        self._update_gui_from_params()
        # Set radio buttons depending on selected image type
        self._reset_radio_channel_mode_choices()
        self._reset_radio_norm_choices()
        self._reset_predict_buttons()
        selected_img = self._get_selected_img()
        self._compute_image_stats(selected_img)
        # Reset the features for continuous training
        self._reset_train_features()

    def _reset_attributes(self):
        """Reset all attributes of the widget."""
        # Set widget-specific attributes
        self.image_mean = None
        self.image_std = None
        self.trained = False
        self.project_widget = None
        # self.selected_channel = None
        # Plugin options and attributes
        self.auto_seg = True # Automatically segment after training
        self.old_annot_tag = "None" # Tag for the annotations, saved to be able to rename them later
        self.old_seg_tag = "None" # Tag for the segmentation, saved to be able to rename them later
        self.old_proba_tag = "None" # Tag for the probabilities, saved to be able to rename them later
        self.add_layers_flag = True # Flag to prevent adding layers twice on one trigger
        # self.update_layer_flag = True # Flag to prevent updating layers twice on one trigger
        # self.rgb_img = getattr(self, "rgb_img", None) or False # Tag to register if the image is RGB
        self.data_shape = None # Shape of the currently selected image data
        self.current_model_path = 'not trained' # Path to the current model (if saved)
        self.auto_add_layers = True # Automatically add layers when a new image is selected
        self.keep_layers = False # Keep old layers when adding new ones
        self.auto_select_annot = False # Automatically select annotation layer based on image name
        self.annot_prefix = 'annotations' # Prefix for the annotation layer names
        self.seg_prefix = 'segmentation' # Prefix for the segmentation layer names
        self.proba_prefix = 'probabilities' # Prefix for the class probabilities layer names
        self.features_prefix = 'features' # Prefix for the feature image layer name
        self.cont_training = "image" # Update features for subsequent training ("image" or "off" or "global")
        self.use_dask = False # Use Dask for parallel processing
        self.input_channels = "" # Input channels for the model (as txt, will be parsed)
        self.add_seg = True # Add a layer with segmentation
        self.add_probas = False # Add a layer with class probabilities
        self.new_seg = True # Flags to indicate if new outputs are created
        self.new_proba = True
        self.new_features = True
        self.features_pca_components = "0" # Number of PCA components for feature image (0 = no PCA)
        self.features_kmeans_clusters = "0" # Number of k-means clusters for feature image (0 = no k-means)
        self.initial_labels = ['Background', 'Foreground']
        self.annot_layers = set() # List of annotation layers
        self.seg_layers = set() # List of segmentation layers
        self.class_labels = [] # List of class labels
        self.class_icons = [] # List of class icons
        self.cmap_flag = False # Flag to prevent infinite loops when changing colormaps
        self.labels_cmap = None # Colormap for the labels (annotations and segmentation)
        self._block_layer_select = True # Flag to block layer selection events temporarily

### Model Tab

    def _on_fe_selected(self, index):
        """Update GUI to show selectable layers of model chosen from drop-down."""

        # Create a temporary model to get the layers (to display) and default parameters
        new_fe_type = self.qcombo_fe_type.currentText()
        current_gpu = self.cp_model.get_param("fe_use_gpu")
        self.temp_fe_model = ConvpaintModel.create_fe(new_fe_type,
                                                      current_gpu)

        # Update the GUI to show the FE layers of the temp model
        self._update_gui_fe_layer_choice_from_temp_model()

        # Get the default FE params for the temp model and update the GUI
        fe_defaults = self.temp_fe_model.get_default_params()

        # NOTE: only FE params are adjusted here, since the FE is not set yet
        self._update_gui_fe_scalings(fe_defaults.fe_scalings)
        val_to_setter = {
            "fe_name": self.qcombo_fe_type.setCurrentText,
            "fe_order": self.spin_interpolation_order.setValue,
            "fe_use_min_features": self.check_use_min_features.setChecked,
            "fe_use_gpu": self.check_use_gpu.setChecked
        }
        for attr, setter in val_to_setter.items():
            val = getattr(fe_defaults, attr, None)
            if val is not None:
                if isinstance(val, list): val = str(val)
                setter(val)
        self.FE_description.setText(self.temp_fe_model.get_description())

    def _on_fe_layer_selection_changed(self):
        """Enable the set button based on the model type."""
        if self.temp_fe_model.get_layer_keys() is not None:
            selected_layers = self._get_selected_layer_names()
            if len(selected_layers) == 0:
                self.set_fe_btn.setEnabled(False)
            else:
                self.set_fe_btn.setEnabled(True)
        else:
            self.set_fe_btn.setEnabled(True)

    def _on_fe_scalings_changed(self):
        """Update param object only when FE is set
        (i.e., do nothing when changing scalings before FE is set)."""
        return

    def _on_set_fe_model(self, event=None):
        """Create a neural network model that will be used for feature extraction and
        reset the classifier."""

        # Read FE parameters from the GUI
        new_param = self.cp_model.get_params()
        new_layers = self._get_selected_layer_names() # includes re-writing to keys
        new_param.set(fe_name = self.qcombo_fe_type.currentText(),
                      fe_layers = new_layers,
                      fe_use_gpu = self.check_use_gpu.isChecked(),
                      fe_scalings = self.fe_scaling_factors.currentData(),
                      fe_order = self.spin_interpolation_order.value(),
                      fe_use_min_features = self.check_use_min_features.isChecked())

        # Get default non-FE params from temp model and update the GUI (also setting the params)
        fe_defaults = self.temp_fe_model.get_default_params()
        adjusted_params = [] # List of adjusted parameters for raising a warning
        data = self._get_selected_img()
        data_dims = self._get_data_dims(data)
        # Multichannel
        if ((fe_defaults.channel_mode is not None) and
            (new_param.channel_mode != fe_defaults.channel_mode)):
            # Catch case where multichannel is adjusted on incompatible data
            if data_dims in ['2D'] and fe_defaults.channel_mode in ['multi', 'rgb']:
                warnings.warn(f'The feature extractor tried to set its default "{fe_defaults.channel_mode}" channel mode on {data_dims} data. ' +
                              'This is not supported and will be ignored.')
            elif data_dims in ['2D_RGB', '3D_RGB', '4D'] and fe_defaults.channel_mode == 'single':
                warnings.warn(f'The feature extractor tried to set its default single-channel mode on {data_dims} data. ' +
                              'This is not supported and will be ignored.')
            else: # If data is compatible, set the model's default multichannel setting
                adjusted_params.append('channel_mode')
                new_param.channel_mode = fe_defaults.channel_mode
                self._reset_radio_channel_mode_choices()
                self._reset_radio_norm_choices() # Update norm options, since channel_mode changed
            # else: # If data is compatible, set the model's default RGB setting
            #     adjusted_params.append('rgb_img')
            #     if fe_defaults.rgb_img: # If the default model is RGB, set it
            #         new_param.rgb_img = True
            #         self.radio_rgb.setChecked(True)
            #     else: # If the default model is non-RGB, reset data dims according to data
            #         new_param.rgb_img = False
            #         self._reset_radio_channel_mode_choices()
            #     self._reset_radio_norm_choices() # Update norm options, since rgb_img changed
        # Normalization
        if ((fe_defaults.normalize is not None) and
            (new_param.normalize != fe_defaults.normalize)):
            if data_dims in ['2D', '2D_RGB', '3D_multi'] and fe_defaults.normalize == 2:
                warnings.warn(f'The feature extractor tried to set its default normalization over stack on {data_dims} data. ' +
                              'This is not supported and will be ignored.')
            else:
                adjusted_params.append('normalize')
                self.button_group_normalize.button(fe_defaults.normalize).setChecked(True)
                setattr(new_param, 'normalize', fe_defaults.normalize)
        # Other params
        val_to_setter = {
            "image_downsample": self.spin_downsample.setValue,
            "seg_smoothening": self.spin_smoothen.setValue,
            "tile_annotations": self.check_tile_annotations.setChecked,
            "tile_image": self.check_tile_image.setChecked,
            "clf_iterations": self.spin_iterations.setValue,
            "clf_learning_rate": self.spin_learning_rate.setValue,
            "clf_depth": self.spin_depth.setValue
        }
        for attr, setter in val_to_setter.items():
            val = getattr(fe_defaults, attr, None)
            if val is not None and val != getattr(new_param, attr):
                adjusted_params.append(attr)
                if isinstance(val, list):
                    val = str(val)
                setter(val) # Set the default value in the GUI
                setattr(new_param, attr, val) # Set the default value in the param object
        if adjusted_params: show_info(f'The feature extractor adjusted the parameters {adjusted_params}')

        # Create a new model with the new FE
        self.cp_model = ConvpaintModel(param=new_param)
        self._reset_clf() # Call to take all actions needed after resetting the clf
        # Reset the features for continuous training
        self._reset_train_features()

    def _on_reset_default_fe(self, event=None):
        """Reset the feature extraction model to the default model."""
        self._reset_fe_params()

    def _on_reset_clf_params(self):
        """Reset the classifier parameters to the default values
        and discard the trained model."""
        self._reset_clf_params()
        self._reset_clf()
    
### Helper functions

    def _add_empty_annot(self, event=None, force_add=True):
        """Add annotation layer to viewer. If the layer already exists,
        remove it (or rename and keep it if specified in self.keep_layers) and add a new one.
        If the widget is used as third party (self.third_party=True), no layer is added if it didn't exist before,
        unless force_add=True (e.g. when the user clicks on the add layer button)"""

        img = self._get_selected_img(check=True)
        if img is None:
            warnings.warn('No image selected. No layers added.')
            return
        layer_shape = self._get_annot_shape(img)

        # Create a new annotation layer if it doesn't exist yet
        annotation_exists = self.annot_prefix in self.viewer.layers

        if (not self.third_party) | (self.third_party & annotation_exists) | (force_add):
            # Create a temp named annotation layer, so we can select it before renaming the old one
            # (and not trigger a problem with incompatible layer shapes)
            temp_name = self._get_unique_layer_name("temp_annotations")
            self.viewer.add_labels(
                data=np.zeros((layer_shape), dtype=np.uint8),
                name=temp_name
                )

            # Select the temp layer in the dropdown
            if self.viewer.layers[temp_name] in self.annotation_layer_selection_widget.choices:
                self.annotation_layer_selection_widget.value = self.viewer.layers[temp_name]
            else:
                warnings.warn(f'The temporary annotation layer {temp_name} could not be selected. ' +
                              'This might cause problems with the annotation layer creation.')
            # If we replace a current layer, we can backup the old one, and replace it
            if annotation_exists:
                # Backup the old annotation layer if keep_layers is set (and the layer exists)
                if self.keep_layers:
                    self._rename_annot_for_backup()
                # Otherwise, just remove the old annotation layer
                else:
                    self.viewer.layers.remove(self.annot_prefix)
            # Create a copy of the temp layer with the original name (so we can select it)
            self.viewer.add_labels(
                data=self.viewer.layers[temp_name].data,
                name=self.annot_prefix
                )
            # Select the new annotation layer in the dropdown
            if self.viewer.layers[self.annot_prefix] in self.annotation_layer_selection_widget.choices:
                self.annotation_layer_selection_widget.value = self.viewer.layers[self.annot_prefix]
            else:
                warnings.warn(f'The annotation layer {self.annot_prefix} could not be selected. ' +
                              'This might cause problems with the annotation layer creation.')
            # Remove the temp layer
            self.viewer.layers.remove(temp_name)

            # Save information about the annotation layer to be able to rename it later
            self._set_old_annot_tag()
            
            # Add it to the list of layers where class labels shall be updated
            self.annot_layers.add(self.viewer.layers[self.annot_prefix])

        # Activate the annotation layer, select it in the dropdown and activate paint mode
        if self.annot_prefix in self.viewer.layers:
            self.viewer.layers.selection.active = self.viewer.layers[self.annot_prefix]
            self.annotation_layer_selection_widget.value = self.viewer.layers[self.annot_prefix]
            self.viewer.layers[self.annot_prefix].mode = 'paint'
            self.viewer.layers[self.annot_prefix].brush_size = self.default_brush_size

        # Sync the class labels
        self.update_all_labels_and_cmaps()

    def _check_create_segmentation_layer(self):
        """Check if segmentation layer exists and create it if not."""
        
        img = self._get_selected_img(check=True)
        if img is None:
            warnings.warn('No image selected. No layers added.')
            return
        layer_shape = self._get_annot_shape(img)
    
        # Create a new segmentation layer if it doesn't exist yet or we need a new one
        seg_exists = self.seg_prefix in self.viewer.layers

        # If we replace a current layer, we can backup the old one, and remove it
        if self.new_seg & seg_exists:
            # Backup the old segmentation layer if keep_layers is set (and the layer exists)
            if self.keep_layers:
                self._rename_seg_for_backup()
            # Otherwise, just remove the old segmentation layer
            else:
                self.viewer.layers.remove(self.seg_prefix)

        # If there was no segmentation layer, or we need a new one, create it
        if (not seg_exists) or self.new_seg:
            self.viewer.add_labels(
                data=np.zeros((layer_shape), dtype=np.uint8),
                name=self.seg_prefix
                )
            # Save information about the segmentation layer to be able to rename it later
            self._set_old_seg_tag()
            
            # Add it to the list of layers where class labels shall be updated
            self.seg_layers.add(self.viewer.layers[self.seg_prefix])
            self.update_all_labels_and_cmaps()

    def _check_create_probas_layer(self, num_classes):
        """Check if class probabilities layer exists and create it if not."""

        img = self._get_selected_img(check=True)
        if img is None:
            warnings.warn('No image selected. No layers added.')
            return
        
        spatial_dims = self._get_annot_shape(img)
        if isinstance(num_classes, int):
            num_classes = (num_classes,)

        # Create a new probabilities layer if it doesn't exist yet or we need a new one
        proba_exists = self.proba_prefix in self.viewer.layers

        # If we replace a current layer, we can backup the old one, and remove it
        if proba_exists:
            same_num_classes = self.viewer.layers[self.proba_prefix].data.shape[0] == num_classes[0]
            if self.new_proba or not same_num_classes:
                # Backup the old probabilities layer if keep_layers is set (and the layer exists)
                if self.keep_layers:
                    self._rename_probas_for_backup()
                # Otherwise, just remove the old probabilities layer
                else:
                    self.viewer.layers.remove(self.proba_prefix)

        # If there was no probabilities layer, or we need a new one, create it
        if (not proba_exists) or (proba_exists and not same_num_classes) or self.new_proba:
            # Create a new probabilities layer
            self.viewer.add_image(
                data=np.zeros(num_classes+spatial_dims, dtype=np.float32),
                name=self.proba_prefix
                )
            # Change the colormap to one suited for probabilities
            self.viewer.layers[self.proba_prefix].colormap = "turbo"
            # Save information about the probabilities layer to be able to rename it later
            self._set_old_proba_tag()

    def _check_create_features_layer(self, num_features):
        """Check if feature image layer exists and create it if not."""

        img = self._get_selected_img(check=True)
        if img is None:
            warnings.warn('No image selected. No layers added.')
            return
        
        spatial_dims = self._get_annot_shape(img)

        # Create a new features layer if it doesn't exist yet or we need a new one
        features_exists = self.features_prefix in self.viewer.layers

        # If we replace a current layer, we can backup the old one, and remove it
        if features_exists:
            same_num_features = (self.viewer.layers[self.features_prefix].data.shape[0] == num_features) & (num_features != 0)
            both_kmeans = (self.viewer.layers[self.features_prefix].data.shape == spatial_dims) & (num_features == 0)
            same_dim = same_num_features or both_kmeans
            if self.new_features or not same_dim:
                # Backup the old features layer if keep_layers is set (and the layer exists)
                if self.keep_layers:
                    self._rename_features_for_backup()
                # Otherwise, just remove the old features layer
                else:
                    self.viewer.layers.remove(self.features_prefix)

        # If there was no features layer, or we need a new one, create it
        if (not features_exists) or (features_exists and not same_dim) or self.new_features:
            # Create a new features layer
            if not num_features == 0:
                self.viewer.add_image(
                    data=np.zeros((num_features,)+spatial_dims, dtype=np.float32),
                    name=self.features_prefix
                    )
                # Change the colormap to one suited for features
                self.viewer.layers[self.features_prefix].colormap = "viridis"
            else: # Kmeans feature image (2D or 3D)
                self.viewer.add_labels(
                    data=np.zeros(spatial_dims, dtype=np.uint8),
                    name=self.features_prefix
                    )
            # Save information about the features layer to be able to rename it later
            self._set_old_features_tag()

    def _rename_annot_for_backup(self):
        """Name the annotation with a unique name according to its image,
        so it can be kept when adding a new with the standard name."""
        # Rename the layer to avoid overwriting it
        if self.annot_prefix in self.viewer.layers:
            full_name = self._get_unique_layer_name(self.old_annot_tag)
            self.viewer.layers[self.annot_prefix].name = full_name
            # Add it to the layers where class labels shall be updated
            self.annot_layers.add(self.viewer.layers[full_name])

    def _rename_seg_for_backup(self):
        """Name the segmentation layer with a unique name according to its image,
        so it can be kept when adding a new with the standard name."""
        # Rename the layer to avoid overwriting it
        if self.seg_prefix in self.viewer.layers:
            full_name = self._get_unique_layer_name(self.old_seg_tag)
            self.viewer.layers[self.seg_prefix].name = full_name
            # Add it to the list of layers where class labels shall be updated
            self.seg_layers.add(self.viewer.layers[full_name])

    def _rename_probas_for_backup(self):
        """Name the class probabilities layer with a unique name according to its image,
        so it can be kept when adding a new with the standard name."""
        # Rename the layer to avoid overwriting it
        if self.proba_prefix in self.viewer.layers:
            full_name = self._get_unique_layer_name(self.old_proba_tag)
            self.viewer.layers[self.proba_prefix].name = full_name

    def _rename_features_for_backup(self):
        """Name the features layer with a unique name according to its image,
        so it can be kept when adding a new with the standard name."""
        # Rename the layer to avoid overwriting it
        if self.features_prefix in self.viewer.layers:
            full_name = self._get_unique_layer_name(self.old_features_tag)
            self.viewer.layers[self.features_prefix].name = full_name

    def _get_unique_layer_name(self, base_name):
        """Get a unique name for a new layer by checking existing layers."""

        # Check if the layer name already exists in the viewer
        existing_layers = [layer.name for layer in self.viewer.layers]
        new_name = base_name
        exists = new_name in existing_layers

        # If the layer already exists, find a unique name
        num = 1
        while exists:
            new_name = f"{base_name} [{num}]"
            num += 1
            exists = new_name in existing_layers
        
        return new_name

    def _get_old_data_tag(self):
        """Set the old data tag based on the current image layer and data dimensions.
        This is used to rename old layers when creating new ones."""
        image_name = self.image_layer_selection_widget.value.name
        channel_mode_str = (self.radio_rgb.isChecked()*"RGB" +
                    self.radio_multi_channel.isChecked()*"multiCh" +
                    self.radio_single_channel.isChecked()*"singleCh")
        return f"{image_name}_{channel_mode_str}"
    
    def _set_old_annot_tag(self):
        """Set the old annotation tag based on the current image layer and data dimensions.
        This is used to rename old annotation layers when creating new ones."""
        self.old_annot_tag = f"{self.annot_prefix}_{self._get_old_data_tag()}"

    def _set_old_seg_tag(self):
        """Set the old segmentation tag based on the current image layer and data dimensions.
        This is used to rename old segmentation layers when creating new ones."""
        self.old_seg_tag = f"{self.seg_prefix}_{self._get_old_data_tag()}"

    def _set_old_proba_tag(self):
        """Set the old probabilities tag based on the current image layer and data dimensions.
        This is used to rename old probabilities layers when creating new ones."""
        self.old_proba_tag = f"{self.proba_prefix}_{self._get_old_data_tag()}"

    def _set_old_features_tag(self):
        """Set the old features tag based on the current image layer and data dimensions.
        This is used to rename old features layers when creating new ones."""
        self.old_features_tag = f"{self.features_prefix}_{self._get_old_data_tag()}"

    def _reset_radio_channel_mode_choices(self):
        """Set radio buttons for channel mode active/inactive depending on selected image type.
        Note that this will also trigger _on_channel_mode_changed(), and adjust the channel_mode param, if any change occurs."""
        if self.image_layer_selection_widget.value is None:
            for x in self.channel_buttons: x.setEnabled(False)
            return
        rgb_img = self.image_layer_selection_widget.value.rgb if hasattr(self.image_layer_selection_widget.value, 'rgb') else False
        if rgb_img: # If the image is RGB (2D or 3D makes no difference), this is the only option
            self.radio_single_channel.setEnabled(False)
            self.radio_multi_channel.setEnabled(False)
            self.radio_rgb.setEnabled(True)
            self.radio_rgb.setChecked(True) # enforce rgb
        elif self.image_layer_selection_widget.value.ndim == 2: # If non-rgb 2D, only single channel
            self.radio_single_channel.setEnabled(True)
            self.radio_multi_channel.setEnabled(False)
            self.radio_rgb.setEnabled(False)
            self.radio_single_channel.setChecked(True) # enforce single
        elif self.image_layer_selection_widget.value.ndim == 3: # If non-rgb 3D, single or multi channel
            self.radio_single_channel.setEnabled(True)
            self.radio_multi_channel.setEnabled(True)
            self.radio_rgb.setEnabled(False)
            self.radio_single_channel.setChecked(self.cp_model.get_param("channel_mode") == "single")
            self.radio_multi_channel.setChecked(self.cp_model.get_param("channel_mode") != "single") # If it was multi or rgb, set multi
        elif self.image_layer_selection_widget.value.ndim == 4: # If non-rgb 4D, it must be multi channel
            self.radio_single_channel.setEnabled(False)
            self.radio_multi_channel.setEnabled(True)
            self.radio_rgb.setEnabled(False)
            self.radio_multi_channel.setChecked(True) # enforce multi

    def _reset_radio_norm_choices(self, event=None):
        """Set radio buttons for normalization active/inactive depending on selected image type.
        If current parameter is not valid, set a default."""
        # Reset the stats; not necessary, since they are recalculated anyway if changing the norm mode
        # self.image_mean, self.image_std = None, None

        if self.image_layer_selection_widget.value is None:
            for x in self.norm_buttons: x.setEnabled(False)
            return
        
        data = self._get_selected_img()
        data_dims = self._get_data_dims(data)
        norm_scope = self.cp_model.get_param("normalize")
        
        if data_dims in ['2D', '2D_RGB', '3D_multi']: # No z dim available -> no stack norm
            self.radio_no_normalize.setEnabled(True)
            self.radio_normalize_over_stack.setEnabled(False)
            self.radio_normalize_by_image.setEnabled(True)
            if norm_scope == 2: # If initially over stack, reset to by image
                self.radio_normalize_by_image.setChecked(True)
            else: # Otherwise, keep the current setting
                self.button_group_normalize.button(norm_scope).setChecked(True)
        else: # 3D_single, 4D, 3D_RGB -> with z dim available -> all options
            self.radio_no_normalize.setEnabled(True)
            self.radio_normalize_over_stack.setEnabled(True)
            self.radio_normalize_by_image.setEnabled(True)
            self.button_group_normalize.button(norm_scope).setChecked(True)

    def _reset_predict_buttons(self):
        """Enable or disable predict buttons based on the current state."""
        # An image layer needs to be selected
        if self.image_layer_selection_widget.value is not None:
            data = self._get_selected_img()
            data_dims = self._get_data_dims(data)
            is_stacked = data_dims in ['4D', '3D_single', '3D_RGB']
            # We need a trained model to enable segmentation
            if self.trained:
                self.segment_btn.setEnabled(True)
                self.segment_all_btn.setEnabled(is_stacked)
            # ... but not for getting features
            self.btn_add_features_stack.setEnabled(is_stacked)
        else: # No image selected
            self.segment_btn.setEnabled(False)
            self.segment_all_btn.setEnabled(False)
            self.btn_add_features_stack.setEnabled(False)

    def _update_gui_fe_layers_from_model(self, cp_model=None):
        """Update GUI FE layer selection based on the current (e.g. loaded) model."""
        if cp_model is None:
            cp_model = self.cp_model
        # Display layers based on the temp FE model
        all_fe_layer_keys = cp_model.get_fe_layer_keys()
        all_fe_layer_texts = self._layer_keys_to_texts(all_fe_layer_keys)
        if all_fe_layer_texts is not None:
            # Add layer index to layer names
            self.fe_layer_selection.clear()
            self.fe_layer_selection.addItems(all_fe_layer_texts)
            self.fe_layer_selection.setEnabled(len(all_fe_layer_texts) > 1) # Only need to enable if multiple layers available
            # Select layers that are in the param object
            layer_keys = cp_model.get_param("fe_layers")
            layer_texts = self._layer_keys_to_texts(layer_keys)
            for layer in layer_texts:
                items = self.fe_layer_selection.findItems(layer, Qt.MatchExactly)
                for item in items:
                    item.setSelected(True)
        # For non-hookmodels, disable the selection
        else:
            self.fe_layer_selection.clear()
            self.fe_layer_selection.setEnabled(False)
        self.set_fe_btn.setEnabled(True)

    def _update_gui_fe_layer_choice_from_temp_model(self):
        """Update GUI selectable FE layers based on the temporary FE model."""
        # Get selectable layers from the temp model and update the GUI
        temp_fe_layer_keys = self.temp_fe_model.get_layer_keys()
        temp_fe_layer_texts = self._layer_keys_to_texts(temp_fe_layer_keys)
        fe_default_layers = self.temp_fe_model.get_default_params().get("fe_layers")
        fe_default_layers = self._layer_keys_to_texts(fe_default_layers)
        if temp_fe_layer_texts is not None:
            self.fe_layer_selection.clear()
            self.fe_layer_selection.addItems(temp_fe_layer_texts)
            self.fe_layer_selection.setEnabled(len(temp_fe_layer_texts) > 1) # Only need to enable if multiple layers available
            for layer in fe_default_layers:
                items = self.fe_layer_selection.findItems(layer, Qt.MatchExactly)
                for item in items:
                    item.setSelected(True)
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

    def _update_gui_from_params(self, params=None):
        """Update GUI from parameters. Use after parameters have been changed outside the GUI (e.g. loading)."""

        # Reset data dim choices (e.g. enable and set "rgb" if the image is RGB)
        self._reset_radio_channel_mode_choices() # important: also adjusts the channel_mode param if data dims demand it
        # Set params from the model if not provided
        if params is None:
            params = self.cp_model.get_params()
        # Set radio buttons depending on param (possibly enforcing a choice)
        if params.channel_mode == "multi":
            self.radio_single_channel.setChecked(False)
            self.radio_multi_channel.setChecked(True)
            self.radio_rgb.setChecked(False)
        elif params.channel_mode == "rgb":
            self.radio_single_channel.setChecked(False)
            self.radio_multi_channel.setChecked(False)
            self.radio_rgb.setChecked(True)
        else: # single
            self.radio_single_channel.setChecked(True)
            self.radio_multi_channel.setChecked(False)
            self.radio_rgb.setChecked(False)
        self._reset_radio_norm_choices()

        self.button_group_normalize.button(params.normalize).setChecked(True)

        val_to_setter = {
            "image_downsample": self.spin_downsample.setValue,
            "seg_smoothening": self.spin_smoothen.setValue,
            "tile_annotations": self.check_tile_annotations.setChecked,
            "tile_image": self.check_tile_image.setChecked,
            "fe_name": self.qcombo_fe_type.setCurrentText,
            "fe_order": self.spin_interpolation_order.setValue,
            "fe_use_min_features": self.check_use_min_features.setChecked,
            "fe_use_gpu": self.check_use_gpu.setChecked,
            "clf_iterations": self.spin_iterations.setValue,
            "clf_learning_rate": self.spin_learning_rate.setValue,
            "clf_depth": self.spin_depth.setValue
        }
        for attr, setter in val_to_setter.items():
            val = getattr(params, attr, None)
            if val is not None:
                if isinstance(val, list): val = str(val)
                setter(val)

        self._update_gui_fe_scalings(params.fe_scalings)
    
    def _get_selected_img(self, check=False):
        """Get the image layer currently selected in Convpaint."""
        img = self.image_layer_selection_widget.value
        if img is None and check:
            warnings.warn('No image layer selected')
        return img

    def _get_annot_shape(self, img):
        """Get shape of annotations and segmentation layers to create."""
        data_dims = self._get_data_dims(img)
        img_shape = img.data.shape
        if data_dims in ['2D_RGB', '2D']:
            return img_shape[0:2]
        elif data_dims in ['3D_RGB', '3D_single']:
            return img_shape[0:3]
        else: # 3D_multi, 4D (-> channels first)
            return img_shape[1:]

    def _check_large_image(self, img):
        """Check if the image is very large and should be tiled."""
        data_dims = self._get_data_dims(img)
        img_shape = img.data.shape
        if data_dims in ['2D', '2D_RGB']:
            xy_plane = img_shape[0] * img_shape[1]
        elif data_dims in ['3D_single', '3D_RGB', '3D_multi']:
            xy_plane = img_shape[1] * img_shape[2]
        else: # 4D
            xy_plane = img_shape[2] * img_shape[3]
        return xy_plane > self.spatial_dim_info_thresh

    def _reset_clf(self):
        """Discard the trained classifier."""
        self.cp_model.reset_classifier()
        self.current_model_path = 'not trained'
        self.trained = False
        # self.save_model_btn.setEnabled(True)
        self._reset_predict_buttons()
        self._set_model_description()

    def _reset_clf_params(self):
        """Reset classifier parameters to default values."""
        # In the widget, which will also trigger to adjust the param object
        self.spin_iterations.setValue(self.default_cp_param.clf_iterations)
        self.spin_learning_rate.setValue(self.default_cp_param.clf_learning_rate)
        self.spin_depth.setValue(self.default_cp_param.clf_depth)
        # In the param object (not done through bindings if values in the widget are not changed)
        self.cp_model.set_params(clf_iterations = self.default_cp_param.clf_iterations,
                                 clf_learning_rate = self.default_cp_param.clf_learning_rate,
                                 clf_depth = self.default_cp_param.clf_depth,
                                 ignore_warnings=True)

    def _reset_fe_params(self):
        """Reset feature extraction parameters to default values."""
        # Reset the gui values for the FE, which will also trigger to adjust the param object
        self.qcombo_fe_type.setCurrentText(self.default_cp_param.fe_name)
        self.fe_layer_selection.clearSelection()
        default_layers = self._layer_keys_to_texts(self.default_cp_param.fe_layers)
        for layer in default_layers:
            items = self.fe_layer_selection.findItems(layer, Qt.MatchExactly)
            for item in items:
                item.setSelected(True)
        self.fe_scaling_factors.setCurrentText(str(self.default_cp_param.fe_scalings))
        self.spin_interpolation_order.setValue(self.default_cp_param.fe_order)
        self.check_use_min_features.setChecked(self.default_cp_param.fe_use_min_features)
        self.check_use_gpu.setChecked(self.default_cp_param.fe_use_gpu)
        # Set default values in param object (by mimicking a click on the "Set FE" button)
        self._on_set_fe_model()

    def _reset_default_general_params(self):
        """Set general parameters back to default."""
        # Set defaults in GUI
        rgb_img = self.image_layer_selection_widget.value.rgb if hasattr(self.image_layer_selection_widget.value, 'rgb') else False
        self.radio_single_channel.setChecked(self.default_cp_param.channel_mode == 'single'
                                             and not rgb_img)
        self.radio_multi_channel.setChecked(self.default_cp_param.channel_mode == 'multi'
                                            and not rgb_img)
        self.radio_rgb.setChecked(rgb_img) # Also put rgb even if the defaults would be different
        self.button_group_normalize.button(self.default_cp_param.normalize).setChecked(True)
        self.spin_downsample.setValue(self.default_cp_param.image_downsample)
        self.spin_smoothen.setValue(self.default_cp_param.seg_smoothening)
        self.check_tile_annotations.setChecked(self.default_cp_param.tile_annotations)
        self.check_tile_image.setChecked(self.default_cp_param.tile_image)
        # Set defaults in param object (not done through bindings if values in the widget are not changed)
        self.cp_model.set_params(channel_mode = "rgb" if rgb_img else self.default_cp_param.channel_mode,
                                 normalize = self.default_cp_param.normalize,
                                 image_downsample = self.default_cp_param.image_downsample,
                                 seg_smoothening = self.default_cp_param.seg_smoothening,
                                 tile_annotations = self.default_cp_param.tile_annotations,
                                 tile_image = self.default_cp_param.tile_image,
                                 ignore_warnings=True)

    def _set_model_description(self):
        """Set the model description text."""
        if self.cp_model.fe_model is None:
            descr = 'No model set'
            return
        name, layers, scalings = (getattr(self.cp_model.get_params(), x)
                                  for x in ['fe_name', 'fe_layers', 'fe_scalings'])
        fe_name = name if name is not None else 'None'
        num_layers = len(layers) if layers is not None else 0
        num_scalings = len(scalings) if scalings is not None else 0
        descr = (fe_name +
        f': {num_layers} layer' + ('' if num_layers == 1 else 's') +
        f', {num_scalings} scaling' + ('' if num_scalings == 1 else 's') + 
        f' ({self.current_model_path})')
        self.model_description1.setText(descr)
        self.model_description2.setText(descr)

    def _get_selected_layer_names(self):
        """Get names of selected FE layers."""
        selected_rows = self.fe_layer_selection.selectedItems()
        selected_layers_texts = [x.text() for x in selected_rows]
        selected_layers = self._layer_texts_to_keys(selected_layers_texts)
        return selected_layers

    @staticmethod
    def _layer_texts_to_keys(layer_texts):
        """Convert FE layer texts to layer keys for use in the model."""
        if layer_texts is None:
            return None
        layer_keys = [text.split(": ", 1)[1] for text in layer_texts]
        return layer_keys
    
    @staticmethod
    def _layer_keys_to_texts(layer_keys):
        """Convert FE layer keys to layer texts for display."""
        if layer_keys is None:
            return None
        layer_texts = [f'{i}: {layer}' for i, layer in enumerate(layer_keys)]
        return layer_texts

    def _get_data_channel_first(self, img):
        """Get data from selected channel. If RGB/RGBA, move channel axis to first
        position and strip alpha channel if present (keep only first 3 channels)."""
        if img is None:
            return None
        data_dims = self._get_data_dims(img)
        if data_dims in ['2D_RGB', '3D_RGB']:
            data = img.data[..., :3]  # Strip alpha channel if RGBA
            img = np.moveaxis(data, -1, 0)
        else:
            img = img.data
        return img

    def _get_data_channel_first_norm(self, img):
        """Get data from selected channel. Output has channel (if present) in 
        first position and is normalized."""

        # Get data from selected layer, with channels first if present
        image_stack = self._get_data_channel_first(img) # 2D, 3D or 4D array, with C first if present

        norm_scope = self.cp_model.get_param("normalize")
        if norm_scope == 1: # No normalization
            return image_stack

        # FE-SPECIFIC NORMALIZATION
        # If the FE demands imagenet or percentile normalization, do that if image is compatible
        fe_norm = self.cp_model.fe_model.norm_mode

        if fe_norm == 'imagenet':
            if self.cp_model.get_param("channel_mode") == 'rgb':  # Only if rgb image --> array with 3 or 4 dims, with C=3 first
                return normalize_image_imagenet(image_stack)
            else:
                print("FE model is designed for imagenet normalization, but image is not declared as 'rgb' (parameter channel_mode). " +
                "Using default normalization instead.")
                
        elif fe_norm == 'percentile':
            if norm_scope == 2: # normalize over stack
                data_dims = self._get_data_dims(img)
                if data_dims in ["4D", "3D_multi", "2D_RGB", "3D_RGB"]: # Channels dimension present
                    num_ignored_dims = 1 # ignore channels dimension, but norm over stack if present
                else: # 2D or 3D_single -> no channels dimension present
                    num_ignored_dims = 0 # norm over entire stack
            elif norm_scope == 3: # normalize by image --> keep channels and plane dimensions
                # also takes into account CXY case (3D multi-channel image) where first dim is dropped, and 2D where none is
                num_ignored_dims = image_stack.ndim-2 # number of channels and/or Z dimension (i.e. "non-spatial" dimensions)
            return normalize_image_percentile(image_stack, ignore_n_first_dims=num_ignored_dims)

        # DEFAULT NORMALIZATION
        
        # If stats are not set, compute them --> they depend on the normalization mode and data dimensions
        if self.image_mean is None or self.image_std is None:
            self._compute_image_stats(img) # Calls _get_data_channel_first() inside -> pass original img

        # Normalize image: (image - image_mean) / image_std
        return normalize_image(image=image_stack,
                               image_mean=self.image_mean,
                               image_std=self.image_std)
    
    def _get_current_plane_norm(self):
        """Get the current image plane to predict on, normalized according to the settings."""
        
        # Get image and the info needed about normalization
        img = self._get_selected_img(check=True)
        if img is None:
            raise ValueError("No image selected.")
        fe_norm = self.cp_model.fe_model.norm_mode
        use_default = (fe_norm == "default")
        if fe_norm == "imagenet" and self.cp_model.get_param("channel_mode") != 'rgb':
            print("FE model is designed for imagenet normalization, but image is not declared as 'rgb' (parameter channel_mode). " +
            "Using default normalization instead.")
            use_default = True
        norm_scope = self.cp_model.get_param("normalize")
        data_dims = self._get_data_dims(img)

        # If we use default normalization, compute image stats if not already done
        if use_default and (self.image_mean is None or self.image_std is None):
            self._compute_image_stats(img)

        # Get image data and stats depending on the data dim and norm mode
        if fe_norm != "percentile":
            # For default and imagenet norm, we want unnormalized data to apply normalization only on the current plane
            img = self._get_data_channel_first(img)
        else: # "percentile"
            # For percentile norm, we want already normalized data to avoid artifacts when normalizing only the current plane
            img = self._get_data_channel_first_norm(img)

        if data_dims in ['2D', '2D_RGB', '3D_multi']: # No stack dim
            image_plane = img
            # Get stats for default norm
            if norm_scope != 1 and use_default: # if we need to normalize and use default norm
                image_mean = self.image_mean
                image_std = self.image_std

        elif data_dims == '3D_single': # Stack dim is third last
            # NOTE: If we already have a layer with probas, the viewer has 4 dims; therefore take 3rd last not first
            step = self.viewer.dims.current_step[-3]
            image_plane = img[step]
            # Get stats for default norm
            if norm_scope == 2 and use_default: # over stack (only 1 value in stack dim; 1, 1, 1)
                image_mean = self.image_mean
                image_std = self.image_std
            if norm_scope == 3 and use_default: # by image (use values for current step; N, 1, 1)
                image_mean = self.image_mean[step]
                image_std = self.image_std[step]

        else: # ['4D', '3D_RGB'] --> Stack dim is third last
            step = self.viewer.dims.current_step[-3]
            image_plane = img[:, step]
            # Get stats for default norm
            if norm_scope == 2 and use_default: # over stack (only 1 value in stack dim; C, 1, 1, 1)
                image_mean = self.image_mean[:,0]
                image_std = self.image_std[:,0]
            if norm_scope == 3 and use_default: # by image (use values for current step; C, N, 1, 1)
                image_mean = self.image_mean[:,step]
                image_std = self.image_std[:,step]
        
        # Normalize image (for default: use the stats based on the radio buttons; for imagenet: stats are fixed)
        if norm_scope != 1:
            if use_default:
                image_plane = normalize_image(image=image_plane, image_mean=image_mean, image_std=image_std)
            elif fe_norm == "imagenet":
                # Only if rgb image --> array with 3 or 4 dims, with C=3 first
                if self.cp_model.get_param("channel_mode") == 'rgb': # Double-check (actually redundant due to check above)
                    image_plane = normalize_image_imagenet(image=image_plane)
                else:
                    print("WIDGET _ON_PREDICT(): THIS SHOULD NOT BE HAPPENING, AS WE CHECKED ABOVE FOR RGB")
                    image_plane = normalize_image(image=image_plane, image_mean=image_mean, image_std=image_std)
            # For percentile norm, image is already normalized above (before selecting the plane)
        
        return image_plane

    def _compute_image_stats(self, img):
        """Get image stats depending on the normalization settings and data dimensions.
        Takes image as given in the viewer (not channel first)."""

        # If no image is selected, set stats to None
        if img is None:
            self.image_mean, self.image_std = None, None
            return

        # Assure to have channels dimension first, get the data_dims and normalization mode
        data = self._get_data_channel_first(img)
        data_dims = self._get_data_dims(img)
        norm_scope = self.cp_model.get_param("normalize")

        # Compute image stats depending on the normalization mode

        if norm_scope == 2: # normalize over stack --> only keep channels dimension
            if data_dims in ["4D", "3D_multi", "2D_RGB", "3D_RGB"]: # Channels dimension present
                # 3D multi/2D_RGB or 4D/3D_RGB --> (C,1,1) or (C,1,1,1)
                self.image_mean, self.image_std = compute_image_stats(
                    image=data,
                    ignore_n_first_dims=1) # ignore channels dimension, but norm over stack if present
            else: # No channels dimension present
                # 2D or 3D_single --> (1,1) or (1,1,1)
                self.image_mean, self.image_std = compute_image_stats(
                    image=data,
                    ignore_n_first_dims=None) # norm over entire stack

        elif norm_scope == 3: # normalize by image --> keep channels and plane dimensions
            # also takes into account CXY case (3D multi-channel image) where first dim is dropped, and 2D where none is
            c_z_channels = data.ndim-2 # number of channels and/or Z dimension (i.e. "non-spatial" dimensions)
            self.image_mean, self.image_std = compute_image_stats(
                image=data, ignore_n_first_dims=c_z_channels) # ignore all but the plane/spatial dimensions
            # --> 2D (1,1) or 3D (single, multi -> Z/C,1,1) or 2D_RGB (C,1,1) or 4D/3D_RGB (C,Z,1,1)

    def _get_data_dims(self, img):
        """Get data dimensions. Also perform checks on the data dimensions.
        Returns '2D', '2D_RGB', '3D_RGB', '3D_multi', '3D_single' or '4D'."""

        if img is None:
            return None
        num_dims = img.ndim
        # Sanity check for number of dimensions
        if (num_dims == 1 or num_dims > 4):
            raise Exception('Image has wrong number of dimensions')
        
        # 2D can be either single channel or RGB/RGBA (in which case the underlying data is in fact 3D with 3-4 channels as last dim)
        if num_dims == 2:
            if self.cp_model.get_param("channel_mode") == "rgb":
                if img.data.shape[-1] in (3, 4):
                    return '2D_RGB'
                else:
                    warnings.warn('Image is 2D, but does not have 3 or 4 channels as last dimension. Setting channel_mode to "single".')
                    self.cp_model.set_param("channel_mode", "single")
                    return '2D'
            elif self.cp_model.get_param("channel_mode") == "single":
                return '2D'
            else: # multi channel not possible in 2D
                warnings.warn('Image is 2D, but channel_mode is "multi". Setting channel_mode to "single".')
                self.cp_model.set_param("channel_mode", "single")
                return '2D'

        # 3D can be single channel, multi channel or RGB/RGBA (in which case the underlying data is in fact 4D with 3-4 channels as last dim)
        if num_dims == 3:
            if self.cp_model.get_param("channel_mode") == "rgb":
                if img.data.shape[-1] in (3, 4):
                    return '3D_RGB'
                else:
                    warnings.warn('Image is 3D, but does not have 3 or 4 channels as last dimension. Setting channel_mode to "multi".')
                    self.cp_model.set_param("channel_mode", "multi")
                    return '3D_multi'
            if self.cp_model.get_param("channel_mode") == "multi":
                return '3D_multi'
            else: # single
                return '3D_single'

        # 4D can only be multi channel
        if num_dims == 4:
            if self.cp_model.get_param("channel_mode") == "single":
                warnings.warn('Image has 4 dimensions, but channel_mode is "single". Setting channel_mode to "multi".')
                self.cp_model.set_param("channel_mode", "multi")
            return '4D'
        
    def _approve_annotation_layer_shape(self, annot, img):
        """Check if the annotation layer has the same shape as the image layer."""
        problem = (annot is None or img is None
                   or annot.data.shape != self._get_annot_shape(img))
        return not problem

    def _update_image_layers(self):
        """"Update the image layers in the viewer sorted by their names."""
        # Update the choices in the image layer selection widget
        self.image_layer_selection_widget.reset_choices()
        # self.image_layer_selection_widget.refresh_choices()

        # Sort them
        # img_layer_list = [layer for layer in self.viewer.layers
        #                   if isinstance(layer, napari.layers.Image)]
        # img_layer_list = [layer for layer in self.image_layer_selection_widget.choices
        #                   if isinstance(layer, napari.layers.Image)]
        # img_layer_list.sort(key=lambda x: x.name)
        # self.image_layer_selection_widget.choices = [] # reset; seems necessary to allow sorting
        # self.image_layer_selection_widget.choices = [(layer.name, layer) for layer in img_layer_list]
        # return img_layer_list

    def _update_annotation_layers(self):
        """"Update the annotation layers in the viewer sorted by their names."""
        # Update the choices in the annotation layer selection widget
        self.annotation_layer_selection_widget.reset_choices()

        # Sort them
        # annot_layer_list = [layer for layer in self.viewer.layers
        #                     if isinstance(layer, napari.layers.Labels)]
        # annot_layer_list = [layer for layer in self.annotation_layer_selection_widget.choices
        #                     if isinstance(layer, napari.layers.Labels)]
        # annot_layer_list.sort(key=lambda x: x.name)
        # self.annotation_layer_selection_widget.choices = [] # reset; seems necessary to allow sorting
        # self.annotation_layer_selection_widget.choices = [(layer.name, layer) for layer in annot_layer_list]
        # return annot_layer_list

    
### ADVANCED TAB
    
    def _update_training_counts(self):
        """Update the training counts (used with continuous_training/memory_mode) in the GUI."""
        if self.cp_model is None:
            return
        pix = len(self.cp_model.table)
        imgs = len(np.unique(self.cp_model.table['img_id']))
        lbls = len(np.unique(self.cp_model.table['label']))
        self.label_training_count.setText(f'{pix} pixels, {imgs} image{"s"*(imgs>1)}, {lbls} labels')

    def _reset_train_features(self):
        """Reset the training features used with continuous_training/memory_mode."""
        self.cp_model.reset_training()
        self._update_training_counts()
        # Save image_layer names and their corresponding annotation layers, to allow only extracting new features
        self.features_annots = {}

    def _on_show_class_distribution(self, trained_data=False):
        """Show the class distribution of the data used with continuous_training/memory_mode (saved in self.cp_model.table)
        in a pie chart using the according cmaps.
        
        trained_data: If True, show the distribution of the data in the training table (i.e. used for training).
                      If False, show the distribution of the data in the currently selected annotation layer.
        """

        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn('matplotlib is not installed. Cannot show class distribution.')
            return

        # Get the class distribution from the training table
        if trained_data:
            # If trained_data, get the labels from the training table (memory mode)
            if self.cp_model is None or self.cp_model.table is None:
                warnings.warn('No training data available. Cannot show class distribution.')
                return
            labels = self.cp_model.table['label'].values
        else:
            # Otherwise get the labels from the annotations layer selected in the layers widget (excluding unlabeled pixels)
            labels = self.annotation_layer_selection_widget.value.data.flatten()
            labels = labels[labels != 0]
        if len(labels) == 0:
            warnings.warn('No labels available. Cannot show class distribution.')
            return
        classes = np.unique(labels)
        counts = np.array([np.sum(labels == c) for c in classes])
        percs = counts / np.sum(counts) * 100

        # Get class display names from a list, assuming class numbers start at 1
        if self.class_labels is not None and self.class_labels:
            class_names = [self.class_labels[c - 1].text() if 1 <= c <= len(self.class_labels) else str(c) for c in classes]

        # Create label strings for the pie chart
        pie_labels = [f'{count} ({perc:.1f}%)' for count, perc in zip(counts, percs)]

        # Create a donut chart
        class_list = [c for c in range(1, classes.max()+1)]
        fig, ax = plt.subplots(figsize=(8, 4), subplot_kw=dict(aspect="equal"))
        if self.labels_cmap is not None:
            colors = [self.labels_cmap.map(i) for i in class_list if i in classes]
        else:
            cmap = plt.get_cmap('tab20', len(class_list))
            colors = [cmap(i) for i in class_list if i in classes]
        
        # Use custom labels with count and percentage
        wedges, _ = ax.pie(
            counts,
            labels=pie_labels,            # Your custom count + % labels
            colors=colors,
            startangle=90,
            wedgeprops=dict(width=0.5),   # Donut shape
            textprops=dict(color="black") # Label color
        )

        # Add a legend with clean class names (not repeated on pie)
        ax.legend(
            wedges,
            class_names,
            title="Classes",
            loc="center left",
            bbox_to_anchor=(1, 1)
        )

        # Final touches
        ax.axis('equal')  # Keep chart circular
        plt.title('Class Distribution')
        plt.tight_layout()
        plt.show()

    def _on_add_all_annot_layers(self):
        """Add annotation layers for all image layers selected in the layers widget (napari)."""

        # Get the selected image layers in the order they are in the widget
        img_layer_list = [layer for layer in self.viewer.layers
                          if layer in self.viewer.layers.selection]
        chosen_in_dropdwon = self._get_selected_img()
        # If the layer chosen in the dropdown is selected, move it first, so its annotation is added first
        if chosen_in_dropdwon is not None and chosen_in_dropdwon.name in img_layer_list:
            img_layer_list.remove(chosen_in_dropdwon)
            img_layer_list.insert(0, chosen_in_dropdwon)

        for img in img_layer_list:
            layer_shape = self._get_annot_shape(img)
            channel_mode_str = (self.radio_rgb.isChecked()*"RGB" +
                    self.radio_multi_channel.isChecked()*"multiCh" +
                    self.radio_single_channel.isChecked()*"singleCh")
            layer_name = f'{self.annot_prefix}_{img.name}_{channel_mode_str}'
            layer_name = self._get_unique_layer_name(layer_name)
            data = np.zeros((layer_shape), dtype=np.uint8)
            # Create a new annotation layer with the unique name
            self.viewer.add_labels(data=data, name=layer_name)
            labels_layer = self.viewer.layers[layer_name]
            # Set the annotation layer to paint mode
            labels_layer.mode = 'paint'
            labels_layer.brush_size = self.default_brush_size
            # Connect colormap events in new layers
            labels_layer.events.colormap.connect(self._on_change_annot_cmap)
            # Add it to the list of layers where class labels shall be updated
            self.annot_layers.add(labels_layer)
        
        # sync class labels and cmaps
        self.update_all_labels_and_cmaps()

    def _on_train_on_selected(self):
        """Train the model on the image and annotation layers currently selected in the layers widget (napari)."""

        # Get selected layers (arbitrary order) and sort them by their names
        layer_list = list(self.viewer.layers.selection)
        layer_list.sort(key=lambda x: x.name)
        
        # Get the image and annotation layers based on the prefix
        prefix_len = len(self.annot_prefix)
        annot_list = [layer for layer in layer_list
                      if layer.name[:prefix_len] == self.annot_prefix
                      and isinstance(layer, napari.layers.Labels)]
        img_list = [layer for layer in layer_list
                    if not layer.name[:prefix_len] == self.annot_prefix
                    and isinstance(layer, napari.layers.Image)]

        # NOTE: Checks are technically not necessary here, as it is done in the CPModel
        if len(annot_list) == 0 or len(img_list) == 0 or len(annot_list) != len(img_list):
            warnings.warn('Please select images and corresponding annotation layers')
            return
        
        # Create lists of images and annotations
        id_list = [img.name for img in img_list]
        img_list = [self._get_data_channel_first(img) for img in img_list]
        annot_list = [annot.data for annot in annot_list]

        # Start training
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(True)        

        with progress(total=0) as pbr:
            pbr.set_description(f"Training")
            mem_mode = self.cont_training == "global"
            # Train; in this case, normalization is not skipped (but done in the ConvpaintModel)
            in_channels = self._parse_in_channels(self.in_channels.text())
            _ = self.cp_model.train(img_list, annot_list, memory_mode=mem_mode, img_ids=id_list, in_channels=in_channels, skip_norm=False, progress=pbr)
            self._update_training_counts()
    
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(False)

        # Set the current model path to 'trained, unsaved' and adjust the model description
        self.current_model_path = 'trained, unsaved'
        self.trained = True
        self._reset_predict_buttons()
        # self.save_model_btn.setEnabled(True)
        self._set_model_description()

    def _on_switch_axes(self):
        """Switch the first two axes of the input image."""
        in_img = self._get_selected_img(check=True)
        data_dims = self._get_data_dims(in_img)
        if in_img is None:
            warnings.warn('No image layer selected')
            return
        elif data_dims != "4D":
            warnings.warn('Switching axes is only supported for 4D images')
            return
        # Get the data from the selected image layer
        img_data = in_img.data
        in_img.data = np.swapaxes(img_data, 0, 1)  # Swap the first two axes

    def _check_parse_pca_kmeans(self):
        """Check and parse the PCA and KMeans settings for continuous training/memory_mode."""
        # Check if pca and kmeans are numbers, warn and set to 0 if not; make integers from strings
        if not self.features_pca_components.isdigit():
            if self.features_pca_components: # If it's not empty
                warnings.warn('PCA components must be an integer. Turning off PCA.')
            self.features_pca_components = '0'
        if not self.features_kmeans_clusters.isdigit():
            if self.features_kmeans_clusters: # If it's not empty
                warnings.warn('KMeans clusters must be an integer. Turning off KMeans.')
            self.features_kmeans_clusters = '0'
        pca, kmeans = int(self.features_pca_components), int(self.features_kmeans_clusters)
        return pca, kmeans

    @staticmethod
    def _annot_to_sparse(annot):
        """Convert annotations np array to sparse format {coord: label}."""
        # Get the coordinates of the non-zero elements in the annotation array
        coords = np.argwhere(annot > 0)
        # Create a sparse representation of the annotations
        sparse_annot = {tuple(coord): annot[tuple(coord)] for coord in coords}
        return sparse_annot
        
    @staticmethod
    def _sparse_to_annot(sparse_annot, shape):
        """Convert sparse annotations to np array."""
        annot = np.zeros(shape, dtype=np.uint8)
        for coord, label in sparse_annot.items():
            annot[coord] = label
        return annot
    
    @staticmethod
    def _remove_duplicates_from_annot(new_annot, existing_annot):
        """Remove duplicates from new annotations based on existing ones, each given as sparse labels."""
        filtered_annot = new_annot.copy()
        for coord in new_annot.keys():
            if coord in existing_annot.keys():
                # If the coordinate is already in the existing annotations, remove it from the new ones
                del filtered_annot[coord]
        return filtered_annot
    
    @staticmethod
    def _parse_in_channels(channels_text):
        """Parse input channels from text."""
        if not channels_text:
            return None
        try:
            channels = list(map(int, channels_text.split(',')))
            return channels
        except ValueError:
            return None