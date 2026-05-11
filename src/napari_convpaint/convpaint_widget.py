from qtpy.QtWidgets import (QWidget, QPushButton,QVBoxLayout,
                            QLabel, QComboBox,QFileDialog, QListWidget,
                            QCheckBox, QAbstractItemView, QGridLayout, QSpinBox, QButtonGroup,
                            QRadioButton,QDoubleSpinBox, QTableWidget, QTableWidgetItem, QHeaderView,
                            QMessageBox)
from qtpy import QtWidgets, QtGui
from qtpy.QtCore import Qt, QTimer, QUrl
from magicgui.widgets import create_widget
import napari
from napari.utils import progress
from napari.utils.notifications import show_info
from napari_guitils.gui_structures import VHGroup, TabSet
from pathlib import Path
import numpy as np
import warnings
import tifffile
import imageio.v2 as imageio
from collections import defaultdict

# Imported inline to avoid heavy memory usage when the functions are not used:
# import torch
# from .utils import normalize_image, compute_image_stats, normalize_image_percentile, normalize_image_imagenet, get_fe_device
# from .convpaint_model import ConvpaintModel

class ConvpaintWidget(QWidget):
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
    parent : QWidget or None
        parent widget, if any
    third_party : bool
        if True, widget is used as third party and will not add layers to the viewer
        by default.
    """

### Define the basic structure of the widget

    def __init__(self, napari_viewer, parent=None, third_party=False):

        ### Initialize the widget state
        super().__init__(parent=parent)
        self.viewer = napari_viewer

        # Set settings/values that never change
        self.third_party = third_party
        self.selected_channel = None
        self.spatial_dim_info_thresh = 1000000
        self.default_brush_size = 3
        # suffixes for opening images in multifile, to exclude potential annotation/segmentation files
        self.multifile_annot_suffixes = ["annot", "annotation", "annotations", "scribble", "scribbles", "scrib"]
        self.multifile_seg_suffixes = ["seg", "segmentation", "segmentations", "prediction", "predictions", "pred", "preds", "mask", "masks"]
        # Set initial values for attributes that change and can be reset to defaults
        self._reset_attributes()

        ### Build the widget
        style_for_infos = "font-size: 12px; color: rgba(120, 120, 120, 80%); font-style: italic"
        style_for_shortcut_info = "font-size: 11px; color: rgba(120, 120, 120, 80%); font-style: italic"

        # Create main layout
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        # Create and add tabs
        self.tab_names = ['Home', 'Models']
        self.tab_names += ['Classes']
        self.tab_names += ['Advanced']
        self.tab_names += ['Multifile']
        tab_layouts = [None if name not in ['Models', 'Multifile'] else QGridLayout() for name in self.tab_names]
        self.tabs = TabSet(self.tab_names, tab_layouts=tab_layouts) # [None, None, QGridLayout()])
        tab_bar = self.tabs.tabBar()
        tab_bar.setSizePolicy(tab_bar.sizePolicy().horizontalPolicy(), tab_bar.sizePolicy().verticalPolicy())

        # Create docs button
        docs_button = QtWidgets.QToolButton()
        docs_button.setText("Documentation")
        docs_button.setStyleSheet("QToolButton {color: #999; text-decoration: underline; margin-left: 4px; margin-right: 8px}")
        docs_button.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(QUrl("https://guiwitz.github.io/napari-convpaint/book/Landing.html")))
        docs_button.setToolTip("Open the documentation in your default browser.")

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
        
        # Align rows in some tabs on top
        for tab_name in ['Home', 'Models', 'Advanced']:
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
        shortcuts_text2 = 'Shift+q: Set annotations label 1\nShift+w: Set annotations label 2\nShift+e: Set annotations label 3\nShift+r: Set annotations label 4'
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
        self.save_model_btn.setEnabled(True)
        self.model_group.glayout.addWidget(self.save_model_btn, 1,0,1,1)
        self.load_model_btn = QPushButton('Load model')
        self.model_group.glayout.addWidget(self.load_model_btn, 1,1,1,1)
        # Reset model button
        self._reset_convpaint_btn = QPushButton('Reset Convpaint')
        self.model_group.glayout.addWidget(self._reset_convpaint_btn, 2,0,1,2)

        # Add elements to "Layer selection" group
        # Image layer (widget for selecting the layer to segment)
        self.image_layer_selection_widget = create_widget(annotation=napari.layers.Image, label='Pick image')
        self._update_image_layers()
        # annotations layer
        self.annotations_layer_selection_widget = create_widget(annotation=napari.layers.Labels, label='Pick annotation')
        self._update_annotations_layers()
        # Add widgets to layout
        self.image_layer_label = QLabel('Image layer')
        self.layer_selection_group.glayout.addWidget(self.image_layer_label, 0,0,1,1)
        self.layer_selection_group.glayout.addWidget(self.image_layer_selection_widget.native, 0,1,1,1)
        self.annotations_layer_label = QLabel('annotations layer')
        self.layer_selection_group.glayout.addWidget(self.annotations_layer_label, 1,0,1,1)
        self.layer_selection_group.glayout.addWidget(self.annotations_layer_selection_widget.native, 1,1,1,1)

        # Add button for adding annotation/segmentation layers
        self.add_layers_btn = QPushButton('Add annotations layer')
        self.add_layers_btn.setEnabled(True)
        self.layer_selection_group.glayout.addWidget(self.add_layers_btn, 2,0,1,2)

        # Add buttons for "Image Processing" group
        # Radio buttons for "Data Dimensions"
        self.button_group_channels = QButtonGroup()
        self.radio_single_channel = QRadioButton('Single channel image')
        self.radio_multi_channel = QRadioButton('Multichannel image')
        self.radio_rgb = QRadioButton('RGB image')
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
        self.radio_normalize_over_stack = QRadioButton('Normalize over stack')
        self.radio_normalize_by_image = QRadioButton('Normalized by plane')
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
        self.train_group.glayout.addWidget(self.train_classifier_btn, 0,0,1,1)
        self.check_auto_seg = QCheckBox('Auto segment')
        self.check_auto_seg.setChecked(self.auto_seg)
        self.train_group.glayout.addWidget(self.check_auto_seg, 0,1,1,1)
        self.segment_btn = QPushButton('Segment image')
        self.segment_btn.setEnabled(False)
        self.train_group.glayout.addWidget(self.segment_btn, 1,0,1,1)
        self.segment_all_btn = QPushButton('Segment stack')
        self.segment_all_btn.setEnabled(False)
        self.train_group.glayout.addWidget(self.segment_all_btn, 1,1,1,1)

        # Add elements to "Acceleration" group
        # "Tile annotations" checkbox
        self.check_tile_annotations = QCheckBox('Tile annotations for training')
        self.check_tile_annotations.setChecked(False)
        self.acceleration_group.glayout.addWidget(self.check_tile_annotations, 0,0,1,1)
        # "Tile image" checkbox
        self.check_tile_image = QCheckBox('Tile image for segmentation')
        self.check_tile_image.setChecked(False)
        self.acceleration_group.glayout.addWidget(self.check_tile_image, 0,1,1,1)
        # Use Device/GPU dropdown
        self.device_options_default = ['auto', 'gpu', 'cpu']
        self.device_options_gpu_only_clf = ['auto', 'gpu (only classifier)', 'cpu']
        self.device_dropdown = QComboBox()
        self.device_dropdown.addItems(self.device_options_default)
        self.device_label = QLabel('Device (GPU/CPU)')
        self.acceleration_group.glayout.addWidget(self.device_label, 1,0,1,1)
        self.acceleration_group.glayout.addWidget(self.device_dropdown, 1,1,1,1)
        # "Downsample" spinbox
        self.spin_downsample = QSpinBox()
        self.spin_downsample.setMinimum(-20)
        self.spin_downsample.setMaximum(20)
        self.spin_downsample.setValue(1)
        self.downsample_label = QLabel('Downsample input')
        self.acceleration_group.glayout.addWidget(self.downsample_label, 2,0,1,1)
        self.acceleration_group.glayout.addWidget(self.spin_downsample, 2,1,1,1)
        # "Smoothen output" spinbox
        self.spin_smoothen = QSpinBox()
        self.spin_smoothen.setMinimum(1)
        self.spin_smoothen.setMaximum(20)
        self.spin_smoothen.setValue(1)
        self.smoothen_label = QLabel('Smoothen output')
        self.acceleration_group.glayout.addWidget(self.smoothen_label, 3,0,1,1)
        self.acceleration_group.glayout.addWidget(self.spin_smoothen, 3,1,1,1)

        # === MODEL TAB ===

        # Create three groups
        self.current_model_group = VHGroup('Current model', orientation='G')
        self.fe_group = VHGroup('Feature extractor', orientation='G')
        self.classifier_params_group = VHGroup('Classifier (CatBoost)', orientation='G')

        # Add groups to the tab
        self.tabs.add_named_tab('Models', self.current_model_group.gbox, [0, 0, 1, 2])
        self.tabs.add_named_tab('Models', self.fe_group.gbox, [2, 0, 8, 2])
        self.tabs.add_named_tab('Models', self.classifier_params_group.gbox, [10, 0, 3, 2])
        self.tabs.setTabEnabled(self.tabs.tab_names.index('Models'), True)
        
        # Current model
        self.model_description2 = QLabel('None')
        self.current_model_group.glayout.addWidget(self.model_description2, 0, 0, 1, 2)

        # Add "FE architecture" combo box to FE group
        self.qcombo_fe_type = QComboBox()
        self.fe_group.glayout.addWidget(self.qcombo_fe_type, 1, 0, 1, 2)

        # Add "FE description" label to FE group
        self.FE_description = QLabel('None')
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

        # Create and add scalings selection to FE group
        self.fe_scaling_factors = QComboBox()
        self.fe_scaling_factors.setEditable(True)
        self.scalings_label = QLabel('Pyramid scaling factors')
        self.fe_group.glayout.addWidget(self.scalings_label, 4, 0, 1, 1)
        self.fe_group.glayout.addWidget(self.fe_scaling_factors, 4, 1, 1, 1)

        # Add interpolation order spinbox to FE group
        self.spin_interpolation_order = QSpinBox()
        self.spin_interpolation_order.setMinimum(0)
        self.spin_interpolation_order.setMaximum(5)
        self.spin_interpolation_order.setValue(1)
        self.interpolation_label = QLabel('Interpolation order')
        self.fe_group.glayout.addWidget(self.interpolation_label, 5, 0, 1, 1)
        self.fe_group.glayout.addWidget(self.spin_interpolation_order, 5, 1, 1, 1)

        # Add min features checkbox to FE group
        self.check_use_min_features = QCheckBox('Use min features')
        self.check_use_min_features.setChecked(False)
        # self.fe_group.glayout.addWidget(self.check_use_min_features, 6, 0, 1, 1)

        # # Add use gpu checkbox to FE group
        # self.check_use_gpu = QCheckBox('Use GPU')
        # self.check_use_gpu.setChecked(False)
        # self.fe_group.glayout.addWidget(self.check_use_gpu, 6, 1, 1, 1)

        # Add "set" buttons to FE group
        self.set_fe_btn = QPushButton('Set feature extractor')
        self.fe_group.glayout.addWidget(self.set_fe_btn, 6, 0, 1, 2)
        # And reset button
        self.reset_default_fe_btn = QPushButton('Reset to default')
        self.fe_group.glayout.addWidget(self.reset_default_fe_btn, 7, 0, 1, 2)

        # Add classifier parameters
        self.spin_iterations = QSpinBox()
        self.spin_iterations.setMinimum(1)
        self.spin_iterations.setMaximum(1000)
        self.iterations_label = QLabel('Iterations')
        self.classifier_params_group.glayout.addWidget(self.iterations_label, 0, 0, 1, 1)
        self.classifier_params_group.glayout.addWidget(self.spin_iterations, 0, 1, 1, 1)

        self.spin_learning_rate = QDoubleSpinBox()
        self.spin_learning_rate.setMinimum(0.001)
        self.spin_learning_rate.setMaximum(1.0)
        self.spin_learning_rate.setSingleStep(0.01)
        self.learning_rate_label = QLabel('Learning Rate')
        self.classifier_params_group.glayout.addWidget(self.learning_rate_label, 1, 0, 1, 1)
        self.classifier_params_group.glayout.addWidget(self.spin_learning_rate, 1, 1, 1, 1)

        self.spin_depth = QSpinBox()
        self.spin_depth.setMinimum(1)
        self.spin_depth.setMaximum(20)
        self.depth_label = QLabel('Depth')
        self.classifier_params_group.glayout.addWidget(self.depth_label, 2, 0, 1, 1)
        self.classifier_params_group.glayout.addWidget(self.spin_depth, 2, 1, 1, 1)

        self.set_clf_btn = QPushButton('Set classifier parameters')
        self.classifier_params_group.glayout.addWidget(self.set_clf_btn, 3, 0, 1, 2)

        self.set_default_clf_btn = QPushButton('Reset to defaults')
        self.set_default_clf_btn.setEnabled(True)
        self.classifier_params_group.glayout.addWidget(self.set_default_clf_btn, 4, 0, 1, 2)

        # === CLASSES TAB ===

        if 'Classes' in self.tab_names:
            # Create the main layout
            self.classes_layout = QGridLayout()
            self.classes_layout.setAlignment(Qt.AlignTop)
            self.classes_widget = QWidget()
            self.classes_widget.setLayout(self.classes_layout)
             
            # Add text to instruct the user (note that it is optional to use)
            class_names_text = QLabel('Set the names of the classes (optional):')
            class_names_text.setWordWrap(True)
            # class_names_text.setStyleSheet("font-size: 11px; color: rgba(120, 120, 120, 70%)")#; font-style: italic")
            self.classes_layout.addWidget(class_names_text, 0, 0, 1, 10)

            # Add buttons ("add class", "remove class" and reset)
            self.add_class_btn = QPushButton('Add class')
            self.classes_layout.addWidget(self.add_class_btn, len(self.initial_names)+1, 0, 1, 5)
            self.remove_class_btn = QPushButton('Remove class')
            self.classes_layout.addWidget(self.remove_class_btn, len(self.initial_names)+1, 5, 1, 5)
            # Minimal import/export buttons (CSV)
            self.export_class_names_btn = QPushButton('Export class names (csv)')
            self.classes_layout.addWidget(self.export_class_names_btn, len(self.initial_names)+2, 0, 1, 5)
            self.import_class_names_btn = QPushButton('Import class names (csv/txt)')
            self.classes_layout.addWidget(self.import_class_names_btn, len(self.initial_names)+2, 5, 1, 5)
            # Reset to initial state
            self.reset_class_names_btn = QPushButton('Reset to default')
            self.classes_layout.addWidget(self.reset_class_names_btn, len(self.initial_names)+3, 0, 1, 10)
            self.btn_class_distribution_annot = QPushButton('Show class distribution (in annotation)')
            self.classes_layout.addWidget(self.btn_class_distribution_annot, len(self.initial_names)+4, 0, 1, 10)

            # Create the class names
            self._create_default_class_names()

            # Add the widget to the tab
            self.classes_layout.setColumnStretch(1, 1)
            self.classes_layout.setColumnStretch(5, 1)
            self.tabs.add_named_tab('Classes', self.classes_widget)

        # === ADVANCED TAB ===

        if 'Advanced' in self.tab_names:
            # Create group boxes
            self.advanced_note_group = VHGroup('Important note', orientation='G')
            self.advanced_appearance_group = VHGroup('Appearance', orientation='G')
            self.advanced_labels_group = VHGroup('Layers handling', orientation='G')
            self.advanced_training_group = VHGroup('Training', orientation='G')
            # self.advanced_multifile_group = VHGroup('Multifile Training', orientation='G')
            self.advanced_prediction_group = VHGroup('Prediction', orientation='G')
            self.advanced_input_group = VHGroup('Input', orientation='G')
            self.advanced_output_group = VHGroup('Output', orientation='G')
            self.advanced_unsupervised_group = VHGroup('Unsupervised extraction (without annotations)', orientation='G')

            # Add groups to the tab
            self.tabs.add_named_tab('Advanced', self.advanced_note_group.gbox)
            self.tabs.add_named_tab('Advanced', self.advanced_appearance_group.gbox)
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

            # Appearance: show/hide tooltips
            self.check_show_tooltips = QCheckBox('Show tooltips')
            self.check_show_tooltips.setChecked(True)
            self.advanced_appearance_group.glayout.addWidget(self.check_show_tooltips, 0, 0, 1, 1)
            # Wire the checkbox to toggle the promoted widgets' tooltips
            self.check_show_tooltips.toggled.connect(lambda checked: self._setup_init_tooltips() if checked else self._remove_init_tooltips())

            # Checkbox to turn off automatic addition of annot/segmentation layers
            self.check_auto_add_layers = QCheckBox('Auto add annotations')
            self.check_auto_add_layers.setChecked(self.auto_add_layers)
            self.advanced_labels_group.glayout.addWidget(self.check_auto_add_layers, 1, 0, 1, 1)

            # Checkbox for keeping old layers
            self.check_keep_layers = QCheckBox('Keep old layers')
            self.check_keep_layers.setChecked(self.keep_layers)
            self.advanced_labels_group.glayout.addWidget(self.check_keep_layers, 1, 1, 1, 1)

            # Button for adding annotations layers for selected images
            self.btn_add_all_annot_layers = QPushButton('Add for all selected')
            self.advanced_labels_group.glayout.addWidget(self.btn_add_all_annot_layers, 2, 0, 1, 1)

            # Checkbox for auto-selecting annotations layers
            self.check_auto_select_annot = QCheckBox('Auto-select annotations layer')
            self.check_auto_select_annot.setChecked(self.auto_select_annot)
            self.advanced_labels_group.glayout.addWidget(self.check_auto_select_annot, 2, 1, 1, 1)

            # Textbox to define the prefix for the annotations layers; NOTE: DISABLED FOR NOW
            # self.text_annot_prefix = QtWidgets.QLineEdit()
            # self.text_annot_prefix.setText('annot_')
            # self.advanced_labels_group.glayout.addWidget(QLabel('annotations prefix'), 2, 0, 1, 1)
            # self.advanced_labels_group.glayout.addWidget(self.text_annot_prefix, 2, 1, 1, 1)
            # Ensure both columns are stretched equally
            self.advanced_labels_group.glayout.setColumnStretch(0, 1)
            self.advanced_labels_group.glayout.setColumnStretch(1, 1)

            # Button for training on selected images
            self.btn_train_on_selected = QPushButton('Train on selected data')
            # self.advanced_training_group.glayout.addWidget(self.btn_train_on_selected, 1, 0, 1, 2)

            # Radio Buttons for continuous training
            self.button_group_cont_training = QButtonGroup()
            self.radio_img_training = QRadioButton('Image')
            self.radio_global_training = QRadioButton('Global')
            self.radio_single_training = QRadioButton('Off')
            self.radio_img_training.setChecked(True)
            self.button_group_cont_training.addButton(self.radio_img_training, id=1)
            self.button_group_cont_training.addButton(self.radio_global_training, id=2)
            self.button_group_cont_training.addButton(self.radio_single_training, id=3)
            self.advanced_training_group.glayout.addWidget(QLabel('Continuous training:'), 2, 0, 1, 1)
            self.advanced_training_group.glayout.addWidget(self.radio_img_training, 2,1,1,1)
            self.advanced_training_group.glayout.addWidget(self.radio_global_training, 2,2,1,1)
            self.advanced_training_group.glayout.addWidget(self.radio_single_training, 2,3,1,1)

            # self.check_cont_training = QCheckBox('Continuous training')
            # self.check_cont_training.setChecked(self.cont_training)
            # self.advanced_training_group.glayout.addWidget(self.check_cont_training, 2, 0, 1, 1)

            # Label for number of trainings performed
            self.label_training_count = QLabel('')
            self.advanced_training_group.glayout.addWidget(self.label_training_count, 3, 0, 1, 2)

            # Button to display a diagram of class distribution
            self.btn_class_distribution_trained = QPushButton('Show class distr. (trained)')
            self.advanced_training_group.glayout.addWidget(self.btn_class_distribution_trained, 3, 2, 1, 2)

            # Reset training button
            self.btn_reset_training = QPushButton('Reset continuous training')
            self.advanced_training_group.glayout.addWidget(self.btn_reset_training, 4, 0, 1, 4)

            # Dask option
            self.check_use_dask = QCheckBox('Use Dask when tiling image for segmentation')
            self.check_use_dask.setChecked(self.use_dask)
            self.advanced_prediction_group.glayout.addWidget(self.check_use_dask, 0, 0, 1, 1)

            # Input channels option
            self.text_input_channels = QtWidgets.QLineEdit()
            self.text_input_channels.setStyleSheet("font-size: 12px;")
            self.text_input_channels.setPlaceholderText('e.g. 0,1,2 or 0,1')
            self.channels_label = QLabel('Input channels (empty = all)')
            self.advanced_input_group.glayout.addWidget(self.channels_label, 0, 0, 1, 2)
            self.advanced_input_group.glayout.addWidget(self.text_input_channels, 0, 2, 1, 2)

            # Button to switch first to axes
            self.btn_switch_axes = QPushButton('Switch channels axis')
            self.advanced_input_group.glayout.addWidget(self.btn_switch_axes, 1, 0, 1, 2)

            # Checkbox for adding segmentation
            self.check_add_seg = QCheckBox('Segmentation')
            self.check_add_seg.setChecked(self.add_seg)
            self.advanced_output_group.glayout.addWidget(self.check_add_seg, 0, 0, 1, 1)

            # Checkbox for adding probabilities
            self.check_add_probas = QCheckBox('Probabilities')
            self.check_add_probas.setChecked(self.add_probas)
            self.advanced_output_group.glayout.addWidget(self.check_add_probas, 0, 1, 1, 1)

            # Button to add features for the current plane
            self.btn_add_features = QPushButton('Get features image')
            self.advanced_unsupervised_group.glayout.addWidget(self.btn_add_features, 2, 0, 1, 2)
            # Button to add features for the whole stack
            self.btn_add_features_stack = QPushButton('Get features of stack')
            self.advanced_unsupervised_group.glayout.addWidget(self.btn_add_features_stack, 2, 2, 1, 2)

            # PCA option for the features
            self.text_features_pca = QtWidgets.QLineEdit()
            self.text_features_pca.setStyleSheet("font-size: 12px;")
            self.text_features_pca.setPlaceholderText('e.g. 3 or 5')
            self.text_features_pca.setText(self.features_pca_components)
            self.pca_label = QLabel('PCA components (0 = off)')
            self.advanced_unsupervised_group.glayout.addWidget(self.pca_label, 0, 0, 1, 2)
            self.advanced_unsupervised_group.glayout.addWidget(self.text_features_pca, 0, 2, 1, 2)
            # Kmeans option for the features
            self.text_features_kmeans = QtWidgets.QLineEdit()
            self.text_features_kmeans.setStyleSheet("font-size: 12px;")
            self.text_features_kmeans.setPlaceholderText('e.g. 3 or 5')
            self.text_features_kmeans.setText(self.features_kmeans_clusters)
            self.kmeans_label = QLabel('Kmeans clusters (0 = off)')
            self.advanced_unsupervised_group.glayout.addWidget(self.kmeans_label, 1, 0, 1, 2)
            self.advanced_unsupervised_group.glayout.addWidget(self.text_features_kmeans, 1, 2, 1, 2)

        # === MULTIFILE TAB ===

        if 'Multifile' in self.tab_names:
            # Create three groups for the Multifile tab to match other tabs' style
            self.multifile_files_group = VHGroup('Files', orientation='G')
            self.multifile_train_group = VHGroup('Train/Segment', orientation='G')
            self.multifile_reset_group = VHGroup('Clear/Close', orientation='G')
            self.multifile_export_import_group = VHGroup('Export/Import', orientation='G')
            self.multifile_settings_group = VHGroup('Preferences', orientation='G')

            # Add groups to the Multifile tab
            self.tabs.add_named_tab('Multifile', self.multifile_files_group.gbox, [0, 0, 8, 2])
            self.tabs.add_named_tab('Multifile', self.multifile_reset_group.gbox, [8, 0, 1, 2])
            self.tabs.add_named_tab('Multifile', self.multifile_train_group.gbox, [9, 0, 1, 2])
            self.tabs.add_named_tab('Multifile', self.multifile_export_import_group.gbox, [10, 0, 1, 2])
            self.tabs.add_named_tab('Multifile', self.multifile_settings_group.gbox, [12, 0, 1, 2])

            # Align on top
            self.tabs.widget(self.tabs.tab_names.index('Multifile')).layout().setAlignment(Qt.AlignTop)

            # --- Files group: folder selector + file list
            lbl_folder = QLabel('Folder:')
            self.multifile_path_edit = QtWidgets.QLineEdit()
            # Make path read-only; folder is selected via the button only
            self.multifile_path_edit.setReadOnly(True)
            self.multifile_select_btn = QPushButton('Open image folder')
            self.multifile_files_group.glayout.addWidget(lbl_folder, 0, 0, 1, 1)
            self.multifile_files_group.glayout.addWidget(self.multifile_path_edit, 0, 1, 1, 1)
            self.multifile_files_group.glayout.addWidget(self.multifile_select_btn, 0, 2, 1, 1)

            self.multifile_list = QTableWidget()
            self.multifile_list.setColumnCount(3)
            self.multifile_list.setHorizontalHeaderLabels(['Annot.', 'Image Filename', 'Segm.'])
            # Align the 'Image Filename' header label to the left for readability
            try:
                header_item = self.multifile_list.horizontalHeaderItem(1)
                header_item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            except Exception:
                pass
            self.multifile_list.setSelectionBehavior(QAbstractItemView.SelectRows)
            self.multifile_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
            # Allow sorting by clicking the header
            self.multifile_list.setSortingEnabled(True)
            self.multifile_list.horizontalHeader().setSectionsClickable(True)
            # Make rows a bit thinner by setting the default row height
            try:
                self.multifile_list.verticalHeader().setDefaultSectionSize(24)
            except Exception:
                pass
            # Keep an explicit maximum height for the widget area
            self.multifile_list.setFixedHeight(340)
            # Make filename column stretch and annotated column autosize
            self.multifile_list.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
            self.multifile_list.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
            self.multifile_list.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
            self.multifile_files_group.glayout.addWidget(self.multifile_list, 1, 0, 1, 3)

            self.multifile_clear_annotations_btn = QPushButton('Clear selected annot.')
            self.multifile_reset_group.glayout.addWidget(self.multifile_clear_annotations_btn, 1, 0, 1, 1)
            self.multifile_reset_folder_btn = QPushButton('Close folder')
            self.multifile_reset_group.glayout.addWidget(self.multifile_reset_folder_btn, 1, 1, 1, 1)
            self.multifile_clear_segmentations_btn = QPushButton('Clear selected segm.')
            self.multifile_reset_group.glayout.addWidget(self.multifile_clear_segmentations_btn, 1, 2, 1, 1)

            # --- Train/Segment group: action buttons (placeholders for now)
            self.multifile_train_all_annot_btn = QPushButton('Train on annotated')
            self.multifile_preview_btn = QPushButton('Preview segmentation')
            self.multifile_segment_selected_btn = QPushButton('Segment selected')

            self.multifile_train_group.glayout.addWidget(self.multifile_train_all_annot_btn, 0, 0, 1, 1)
            self.multifile_train_group.glayout.addWidget(self.multifile_preview_btn, 0, 1, 1, 1)
            self.multifile_train_group.glayout.addWidget(self.multifile_segment_selected_btn, 0, 2, 1, 1)

            # --- Import/Export group: action buttons (placeholders for now)
            self.multifile_export_annot_btn = QPushButton('Export annotations')
            self.multifile_import_annot_and_seg_btn = QPushButton('Import annot./segm.')

            self.multifile_export_import_group.glayout.addWidget(self.multifile_export_annot_btn, 0, 0, 1, 1)
            self.multifile_export_import_group.glayout.addWidget(self.multifile_import_annot_and_seg_btn, 0, 1, 1, 1)

            # --- Settings group: checkboxes
            self.lbl_import_open_labels = QLabel('Import and auto-open:')
            self.multifile_settings_group.glayout.addWidget(self.lbl_import_open_labels, 0, 0, 1, 1)
            self.check_open_import_annotations = QCheckBox('Annotations')
            self.check_open_import_segmentations = QCheckBox('Segmentation')
            self.check_open_import_annotations.setChecked(True)
            self.check_open_import_segmentations.setChecked(True)
            self.multifile_settings_group.glayout.addWidget(self.check_open_import_annotations, 0, 1, 1, 1)
            self.multifile_settings_group.glayout.addWidget(self.check_open_import_segmentations, 0, 2, 1, 1)

            self.lbl_multifile_suffixes = QLabel('Suffixes (annot. | segm.):')
            self.multifile_settings_group.glayout.addWidget(self.lbl_multifile_suffixes, 1, 0, 1, 1)
            self.multifile_annotations_suffix_txt = QtWidgets.QLineEdit()
            self.multifile_annotations_suffix_txt.setStyleSheet("font-size: 12px;")
            self.multifile_segmentation_suffix_txt = QtWidgets.QLineEdit()
            self.multifile_segmentation_suffix_txt.setStyleSheet("font-size: 12px;")
            self.multifile_annotations_suffix_txt.setText('annotations')
            self.multifile_segmentation_suffix_txt.setText('segmentation')
            self.multifile_settings_group.glayout.addWidget(self.multifile_annotations_suffix_txt, 1, 1, 1, 1)
            self.multifile_settings_group.glayout.addWidget(self.multifile_segmentation_suffix_txt, 1, 2, 1, 1)
        
        # === Show tooltips by default ===

        self._setup_init_tooltips()
        # Set device dropdown tooltip separately, as we want to show these dynamically and permanently, even when the "Show tooltips" checkbox is unchecked
        self.device_dropdown.setToolTip('Select device policy for feature extraction and classifier.')

    def _setup_init_tooltips(self):

        # Set tooltip for the tabs
        self.tabs.setTabToolTip(self.tabs.tab_names.index('Home'), 'Main controls for Convpaint: select layers, train model and segment images. Adjust image processing and acceleration options.')
        self.tabs.setTabToolTip(self.tabs.tab_names.index('Models'), 'Select the feature extraction model and adjust classifier parameters.')
        if 'Classes' in self.tab_names:
            self.tabs.setTabToolTip(self.tabs.tab_names.index('Classes'), 'Set the names of the classes for the annotations and segmentation layer. Display class distribution.' +
                                    'This is optional but can help with keeping track of the classes, especially when using more than 3 classes.')
        if 'Advanced' in self.tab_names:
            self.tabs.setTabToolTip(self.tabs.tab_names.index('Advanced'), 'Advanced options for power users. Use with caution, as these options may lead to situations where the tool does not function as expected. ' +
                                    'Adjust layer handling, training accross images, Dask usage, form of input and output, and get features as a displayable image.')

        # Home tab
        self.save_model_btn.setToolTip('Save model as *.pkl (incl. classifier) or *.yml (parameters only) file.')
        self.load_model_btn.setToolTip('Select *.pkl or *.yml file to load.')
        self._reset_convpaint_btn.setToolTip('Discard current model and create new default model.')
        for w in [self.image_layer_label, self.image_layer_selection_widget.native]:
            w.setToolTip('Select the image for training and/or segmentation.')
        for w in [self.annotations_layer_label, self.annotations_layer_selection_widget.native]:
            w.setToolTip('Select the annotations layer for training.')
        self.add_layers_btn.setToolTip('Add annotations layer with the correct dimensions to the viewer.')
        self.radio_single_channel.setToolTip('2D images or 3D images where additional dimension is NOT channels.')
        self.radio_multi_channel.setToolTip('Images with an additional channel dimension.')
        self.radio_rgb.setToolTip('This option is used with images displayed as RGB.')
        self.radio_no_normalize.setToolTip('No normalization is applied.')
        self.radio_normalize_over_stack.setToolTip('Normalize over complete z or t stack.')
        self.radio_normalize_by_image.setToolTip('Normalize each plane individually.')
        self.train_classifier_btn.setToolTip('Train model on annotations.')
        self.check_auto_seg.setToolTip('Automatically segment image after training.')
        self.segment_btn.setToolTip('Segment 2D image or current slice/frame of 3D image/movie.')
        self.segment_all_btn.setToolTip('Segment all slices/frames of 3D image/movie.')
        self.check_tile_annotations.setToolTip('Crop around annotated regions to speed up training.\n' +
                                               'Disable for models that extract long range features (e.g. DINO).')
        self.check_tile_image.setToolTip('Tile image to reduce memory usage.\n' +
                                         'Use with care when using models that extract long range features (e.g. DINO).')
        # Do not toggle device dropdown, as we want to show tooltips dynamically and permanently
        # for w in [self.device_label, self.device_dropdown]:
        #     w.setToolTip('Select device policy for feature extraction and classifier.')
        self.device_label.setToolTip('Select device policy for feature extraction and classifier.')
        for w in [self.downsample_label, self.spin_downsample]:
             w.setToolTip('Reduce image size, e.g. for faster computing (output is rescaled to original size).\n' +
                            'Negative values will instead upscale the image by the absolute value.')
        for w in [self.smoothen_label, self.spin_smoothen]:
             w.setToolTip('Smoothen output with a filter of this size.\n' +
                            'Increasing this value can reduce noise in the output and make details less prominent.')

        # Models tab
        self.qcombo_fe_type.setToolTip('Select architecture of feature extraction model.')
        self.fe_layer_selection.setToolTip('Select the layers of the feature extraction model (if applicable) to use for training and segmentation.\n' +
                                           'Select a range by dragging or using shift, choose multiple with ctrl/cmd.\n' +
                                           'If multiple layers are selected, features from these layers will be concatenated and used together for training the classifier.\n')
        for w in [self.scalings_label, self.fe_scaling_factors]:
            w.setToolTip('Set the scaling factors for the pyramid features. Image will be scaled by these factors,\n' +
                         'the features extracted and rescaled to original size, and, finally, concatenated for all scales.')
        for w in [self.interpolation_label, self.spin_interpolation_order]:
            w.setToolTip('Interpolation order for rescaling of the extracted features.')
        self.check_use_min_features.setToolTip('Use same number of features from each layer. Otherwise use all features from each layer.')
        # self.check_use_gpu.setToolTip('Use GPU for training and segmentation')
        self.set_fe_btn.setToolTip('Set the feature extraction model.')
        self.reset_default_fe_btn.setToolTip('Set the feature extractor back to the default model.')
        for w in [self.iterations_label, self.spin_iterations]:
             w.setToolTip('Set the number of iterations for the classifier.')
        for w in [self.learning_rate_label, self.spin_learning_rate]:
            w.setToolTip('Set the learning rate for the classifier.')
        for w in [self.depth_label, self.spin_depth]:
            w.setToolTip('Set the depth of the trees for the classifier.')
        self.set_clf_btn.setToolTip('Apply classifier parameters to the current model.')
        self.set_default_clf_btn.setToolTip('Reset classifier parameters to default values.')

        # Classes tab
        if 'Classes' in self.tab_names:
            self.add_class_btn.setToolTip('Add a class name to the list.')
            self.remove_class_btn.setToolTip('Remove a class from the list. Note that this will also delete the corresponding annotations from the annotations layer, if they exist.')
            self.export_class_names_btn.setToolTip('Export class names as a csv file.')
            self.import_class_names_btn.setToolTip('Import class names from a csv or txt file.')
            self.reset_class_names_btn.setToolTip('Reset the list of class names to "Background" and "Foreground".')
            self.btn_class_distribution_annot.setToolTip('Show a diagram of the class distribution in the annotations layer.')

        # Advanced tab
        if 'Advanced' in self.tab_names:
            self.check_show_tooltips.setToolTip('Toggle display of inline tooltips for various controls.')
            self.check_auto_add_layers.setToolTip('Automatically add annotations layer when selecting images.')
            self.check_keep_layers.setToolTip('Keep old annotations and output layers when creating new ones.')
            self.btn_add_all_annot_layers.setToolTip('Add annotations layers for all selected images in the layers list.')
            self.check_auto_select_annot.setToolTip('Automatically select annotations layers when selecting images.')
            # self.text_annot_prefix.setToolTip('Prefix for annotations layers to be used for training')
            self.btn_train_on_selected.setToolTip("Train using layers selected in the viewer's layer list and beginning with 'annotations'.")
            self.radio_img_training.setToolTip('Keep features in memory, updating them only for new annotations in each training, as long as the image is not changed.')
            self.radio_global_training.setToolTip('Keep features in memory, updating them only for new annotations in each training, until reset manually.')
            self.radio_single_training.setToolTip('Extract all features freshly for each training.')
            # self.check_cont_training.setToolTip('Save and use combined features in memory for training')
            self.btn_class_distribution_trained.setToolTip('Show a diagram of the class distribution in the data saved in the model for training.')
            self.btn_reset_training.setToolTip('Clear training history and restart training counter.')
            self.check_use_dask.setToolTip('Use Dask when using the option "Tile for segmentation".')
            for w in [self.channels_label, self.text_input_channels]:
                w.setToolTip('Comma-separated list of channels to use for training and segmentation.\n' +
                             'Leave empty to use all channels.')
            self.btn_switch_axes.setToolTip('Switch first two axes of a 4D input image (to match the convention to have channels first).')
            self.check_add_seg.setToolTip('Add a layer with the predicted segmentation as output (= highest class probability).')
            self.check_add_probas.setToolTip('Add a layer with class probabilities as output.')
            self.btn_add_features.setToolTip('Add a layer with the features extracted for the current plane.')
            self.btn_add_features_stack.setToolTip('Add a layer with the features extracted for the whole stack.')
            for w in [self.pca_label, self.text_features_pca]:
                 w.setToolTip('Number of PCA components to use for the features image.\nSet to 0 to disable PCA.')
            for w in [self.kmeans_label, self.text_features_kmeans]:
                w.setToolTip('Number of Kmeans clusters to use for the features image.\nSet to 0 to disable Kmeans.')

        if 'Multifile' in self.tab_names:
            self.multifile_select_btn.setToolTip('Select the folder containing the images to segment.\n' +
                                                 'annotations and segmentation files with a corresponding suffix will be ignored in that process.')
            self.multifile_list.setToolTip('List of files in the selected folder. The "Annot." column indicates whether an annotations layer is present in memory (yellow check in brackets), '+
                                           'saved to file (green checkmark) or absent (red cross).\n' +
                                           'Click on the column header to sort by it. Double-click a file to open the image, annotations and segmentation (if present).')
            self.multifile_clear_annotations_btn.setToolTip('Clear annotations for the selected images (if opened, add a new empty annotations layer).')
            self.multifile_reset_folder_btn.setToolTip('Close all images from the selected folder and clear the file list.')
            self.multifile_clear_segmentations_btn.setToolTip('Clear segmentations for the selected images (if opened, remove the segmentation layer).')
            self.multifile_train_all_annot_btn.setToolTip('Train a model using all annotated images in the file list.')
            self.multifile_preview_btn.setToolTip('Preview the segmentation on the opened image in the viewer without saving it to files.')
            self.multifile_segment_selected_btn.setToolTip('Segment all selected images in the file list. Choose a folder to save the segmentations to files.')
            self.multifile_export_annot_btn.setToolTip('Export annotations of all opened images as files.')
            self.multifile_import_annot_and_seg_btn.setToolTip('Import annotations and/or segmentations from files for all images.\n' +
                                                               'Will scan the folder to detect annotations and segmentations belonging to the images and import both (unless specified in settings).')
            for w in [self.lbl_import_open_labels, self.check_open_import_annotations, self.check_open_import_segmentations]:
                w.setToolTip('When importing labels, include annotations/segmentations. Also, when opening an image, automatically open the corresponding layer(s).')
            for w in [self.lbl_multifile_suffixes, self.multifile_annotations_suffix_txt, self.multifile_segmentation_suffix_txt]:
                w.setToolTip('Suffixes to identify annotations and segmentation files in the folder. Will be used for export and for recognizing annotations and segmentations at imports.\n' +
                             'Important convention: the filename must be in the form [image name]_[suffix].[ext], where [suffix] is the specified suffix and [ext] is the file extension (e.g. .tif).')

    def _remove_init_tooltips(self):
        # Remove tooltips for the tabs
        for tab_name in self.tabs.tab_names:
            self.tabs.setTabToolTip(self.tabs.tab_names.index(tab_name), '')

        # Home tab
        for w in [self.save_model_btn, self.load_model_btn, self._reset_convpaint_btn,
                  self.image_layer_label, self.image_layer_selection_widget.native,
                  self.annotations_layer_label, self.annotations_layer_selection_widget.native,
                  self.add_layers_btn, self.radio_single_channel, self.radio_multi_channel, self.radio_rgb,
                  self.radio_no_normalize, self.radio_normalize_over_stack, self.radio_normalize_by_image,
                  self.train_classifier_btn, self.check_auto_seg, self.segment_btn, self.segment_all_btn,
                  self.check_tile_annotations, self.check_tile_image, self.device_label, #self.device_dropdown,
                  self.downsample_label, self.spin_downsample, self.smoothen_label, self.spin_smoothen]:
            w.setToolTip('')

        # Models tab
        for w in [self.qcombo_fe_type, self.fe_layer_selection, self.scalings_label, self.fe_scaling_factors,
                  self.interpolation_label, self.spin_interpolation_order, self.check_use_min_features,
                  # self.check_use_gpu,
                  self.set_fe_btn, self.reset_default_fe_btn,
                  self.iterations_label, self.spin_iterations,
                  self.learning_rate_label, self.spin_learning_rate,
                  self.depth_label, self.spin_depth,
                  self.set_clf_btn, 	self.set_default_clf_btn]:
            w.setToolTip('')

        # Classes tab
        if 'Classes' in self.tab_names:
            for w in [self.add_class_btn, self.remove_class_btn,
                      self.export_class_names_btn, self.import_class_names_btn,
                      self.reset_class_names_btn, self.btn_class_distribution_annot]:
                w.setToolTip('')

        # Advanced tab
        if 'Advanced' in self.tab_names:
            for w in [self.check_show_tooltips, self.check_auto_add_layers,
                      self.check_keep_layers, self.btn_add_all_annot_layers,
                      self.check_auto_select_annot, # 	self.text_annot_prefix,
                      self.btn_train_on_selected, self.radio_img_training, self.radio_global_training, self.radio_single_training, # self.check_cont_training,
                      self.btn_class_distribution_trained, self.btn_reset_training, self.check_use_dask, self.channels_label,
                      self.text_input_channels, self.btn_switch_axes, self.check_add_seg, self.check_add_probas, self.btn_add_features, self.btn_add_features_stack,
                      self.pca_label, self.text_features_pca, self.kmeans_label, self.text_features_kmeans]:
                w.setToolTip('')

        if 'Multifile' in self.tab_names:
            for w in [self.multifile_select_btn, self.multifile_list, self.multifile_clear_annotations_btn,
                      self.multifile_clear_segmentations_btn, self.multifile_reset_folder_btn, self.multifile_train_all_annot_btn,
                      self.multifile_preview_btn, self.multifile_segment_selected_btn, self.multifile_export_annot_btn,
                      self.multifile_import_annot_and_seg_btn, self.lbl_import_open_labels,
                      self.check_open_import_annotations, self.check_open_import_segmentations,
                      self.lbl_multifile_suffixes, self.multifile_annotations_suffix_txt, self.multifile_segmentation_suffix_txt]:
                w.setToolTip('')

### ConvpaintModel instatiation and default population & calling connections, resetting model and key bindings

    def showEvent(self, event):
        """Override the showEvent to populate the model defaults and set up connections AFTER the GUI is shown."""
        super().showEvent(event)

        # Run only once
        if hasattr(self, "_post_init_done") and self._post_init_done:
            return

        self._post_init_done = True

        # Defer slightly to let Qt finish rendering
        QTimer.singleShot(0, self._late_init)

    def ensure_init(self):
        """Run deferred model initialization synchronously if it hasn't run yet.
        Useful for tests and non-GUI contexts where showEvent is not triggered."""
        if hasattr(self, "_post_init_done") and self._post_init_done:
            return
        self._post_init_done = True
        self._late_init()

    def _import_convpaint_model_class(self):
        if not hasattr(self, "_cpm_class"):
            from .convpaint_model import ConvpaintModel
            self._cpm_class = ConvpaintModel

    def _late_init(self):
        """Populate UI widgets with defaults from ConvpaintModel, set up connections, and reset model.
        This is called after the GUI is shown to ensure that all components are properly initialized."""

        # === MODEL DEFAULTS & WIDGET POPULATION ===
        self._import_convpaint_model_class()
        self.cp_model = self._cpm_class()
        # Get default parameters to set in widget
        self.default_cp_param = self._cpm_class.get_default_params()
        # Use variables of main model as temp variables for the Models tab, as it is the one model used at that time
        self.temp_fe_description = self.cp_model.get_fe_description()
        lks = self.cp_model.get_fe_layer_keys()
        self.temp_fe_layer_keys = lks.copy() if lks is not None else None
        pps = self.cp_model.get_fe_proposed_scalings()
        self.temp_fe_proposed_scalings = pps.copy() if pps is not None else None

        self.spin_iterations.setValue(self.default_cp_param.clf_iterations)
        self.spin_learning_rate.setValue(self.default_cp_param.clf_learning_rate)
        self.spin_depth.setValue(self.default_cp_param.clf_depth)
        if 'Advanced' in self.tab_names:
            self._update_training_counts()
        self.FE_description.setText(self.temp_fe_description)
        self.qcombo_fe_type.addItems(sorted(self._cpm_class.get_fe_models_types().keys()))
        num_items = self.qcombo_fe_type.count()
        self.qcombo_fe_type.setMaxVisibleItems(num_items) # Make sure the dropdown shows all items

        # === CONNECTIONS ===
        # Add connections and initialize by setting default model and params
        self._add_connections()
        if self.image_layer_selection_widget.value is not None:
            try: # This should technically not be necessary, as we are not raising errors, but is added as a precaution
                self._on_select_layer()
            except Exception:
                warnings.warn(
                    f'Could not initialize with the selected image '
                    f'(ndim={self.image_layer_selection_widget.value.ndim}). '
                    f'Please select a compatible image (2D-4D).'
                )
        self._reset_model()

        # === KEY BINDINGS ===
        self.viewer.bind_key('Shift+a', self.toggle_annotation, overwrite=True)
        self.viewer.bind_key('Shift+s', self._on_train, overwrite=True)
        self.viewer.bind_key('Shift+d', self._on_predict, overwrite=True)
        self.viewer.bind_key('Shift+f', self.toggle_prediction, overwrite=True)
        self.viewer.bind_key('Shift+q', lambda event=None: self.set_annot_label_class(1, event), overwrite=True)
        self.viewer.bind_key('Shift+w', lambda event=None: self.set_annot_label_class(2, event), overwrite=True)
        self.viewer.bind_key('Shift+e', lambda event=None: self.set_annot_label_class(3, event), overwrite=True)
        self.viewer.bind_key('Shift+r', lambda event=None: self.set_annot_label_class(4, event), overwrite=True)


### Define the connections between the widget elements

    def _add_connections(self):

        # Reset layer choices in dropdown when layers are renamed; also bind this behaviour to inserted layers
        for layer in self.viewer.layers:
            layer.events.name.connect(self._update_image_layers)
        self.viewer.layers.events.inserted.connect(self._on_insert_layer)
        self.viewer.layers.events.removed.connect(self._on_layer_removed)

        # === HOME TAB ===

        # Reset layer choices in dropdowns when napari layers are added or removed
        for layer_widget_reset in [self._update_annotations_layers,
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
        self.annotations_layer_selection_widget.native.activated.connect(self._on_select_annot)
        self.annotations_layer_selection_widget.changed.connect(self._on_select_annot)
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

        # Segment
        self.segment_btn.clicked.connect(self._on_predict)
        self.segment_all_btn.clicked.connect(self._on_predict_all)

        # Acceleration
        self.spin_downsample.valueChanged.connect(lambda:
            self.cp_model.set_param('image_downsample', self.spin_downsample.value(), ignore_warnings=True))
        self.spin_smoothen.valueChanged.connect(lambda:
            self.cp_model.set_param('seg_smoothening', self.spin_smoothen.value(), ignore_warnings=True))
        # set self.use_device according to the drop down value
        self.device_dropdown.currentIndexChanged.connect(self._on_change_device)
        self.check_tile_annotations.stateChanged.connect(lambda:
            self.cp_model.set_param('tile_annotations', self.check_tile_annotations.isChecked(), ignore_warnings=True))
        self.check_tile_image.stateChanged.connect(lambda:
            self.cp_model.set_param('tile_image', self.check_tile_image.isChecked(), ignore_warnings=True))

        # === MODELS TAB ===

        # Feature extractor
        self.qcombo_fe_type.currentIndexChanged.connect(self._on_fe_selected)
        self.set_fe_btn.clicked.connect(self._on_set_fe_model)
        self.reset_default_fe_btn.clicked.connect(self._on_reset_default_fe)
        self.fe_layer_selection.itemSelectionChanged.connect(self._on_fe_layer_selection_changed)
        self.fe_scaling_factors.currentIndexChanged.connect(self.flag_fe_as_temp)
        self.fe_scaling_factors.lineEdit().textChanged.connect(self.flag_fe_as_temp)
        # NOTE: Changing interpolation_order, use_min_features and use_gpu of FE
        # shall only be applied when the user clicks the button to set the FE
        # But we still want to flag the FE model as temporary when changing these parameters
        self.spin_interpolation_order.valueChanged.connect(self.flag_fe_as_temp)
        self.check_use_min_features.stateChanged.connect(self.flag_fe_as_temp)
        
        # Classifier
        # self.spin_iterations.valueChanged.connect(lambda:
        #     self.cp_model.set_param('clf_iterations', self.spin_iterations.value(), ignore_warnings=True))
        # self.spin_learning_rate.valueChanged.connect(lambda:
        #     self.cp_model.set_param('clf_learning_rate', self.spin_learning_rate.value(), ignore_warnings=True))
        # self.spin_depth.valueChanged.connect(lambda:
        #     self.cp_model.set_param('clf_depth', self.spin_depth.value(), ignore_warnings=True))
        self.spin_iterations.valueChanged.connect(self.flag_clf_as_temp)
        self.spin_learning_rate.valueChanged.connect(self.flag_clf_as_temp)
        self.spin_depth.valueChanged.connect(self.flag_clf_as_temp)
        self.set_clf_btn.clicked.connect(self._on_set_clf_params)
        self.set_default_clf_btn.clicked.connect(self._on_reset_clf_params)

        # === CLASSES TAB ===

        if 'Classes' in self.tab_names:
            self.add_class_btn.clicked.connect(lambda: self._on_add_class(text=None))
            self.remove_class_btn.clicked.connect(lambda: self._on_remove_class(del_annots=True))
            self.export_class_names_btn.clicked.connect(lambda: self._export_class_names_dialog())
            self.import_class_names_btn.clicked.connect(lambda: self._import_class_names_dialog())
            self.reset_class_names_btn.clicked.connect(self._on_reset_class_names)

            for class_name in self.class_names:
                class_name.textChanged.connect(self._update_class_names)
            if self.annotations_layer_selection_widget.value is not None:
                labels_layer = self.annotations_layer_selection_widget.value
                labels_layer.events.colormap.connect(self._on_change_annot_cmap)
            if self.seg_tag in self.viewer.layers:
                seg_layer = self.viewer.layers[self.seg_tag]
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
            self.radio_single_training.toggled.connect(lambda checked:
                checked and setattr(self, 'cont_training', 'Off'))
            self.radio_img_training.toggled.connect(lambda checked:
                checked and setattr(self, 'cont_training', "Image"))
            self.radio_global_training.toggled.connect(lambda checked:
                checked and setattr(self, 'cont_training', "Global"))
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

        # === Multifile tab ===
        if 'Multifile' in self.tab_names:
            self.multifile_select_btn.clicked.connect(self._select_multifile_img_folder)
            self.multifile_list.cellDoubleClicked.connect(self._on_multifile_open_file)
            self.multifile_clear_annotations_btn.clicked.connect(self._multifile_clear_annot)
            self.multifile_reset_folder_btn.clicked.connect(self._reset_multifile_folder)
            self.multifile_clear_segmentations_btn.clicked.connect(self._multifile_clear_seg)
            self.multifile_train_all_annot_btn.clicked.connect(self._on_train_on_multifile)
            self.multifile_preview_btn.clicked.connect(self._on_predict) # Use normal prediciton method as preview in Multifile ...
            self.multifile_segment_selected_btn.clicked.connect(self._on_segment_selected_multifile)
            self.multifile_import_annot_and_seg_btn.clicked.connect(self._import_annot_and_seg)
            self.multifile_export_annot_btn.clicked.connect(self._export_annotations)
            self.check_open_import_annotations.stateChanged.connect(lambda: setattr(
                self, 'multifile_import_open_annotations', self.check_open_import_annotations.isChecked()))
            self.check_open_import_segmentations.stateChanged.connect(lambda: setattr(
                self, 'multifile_import_open_segmentations', self.check_open_import_segmentations.isChecked()))
            self.multifile_annotations_suffix_txt.textChanged.connect(lambda: setattr(
                self, 'multifile_annot_suffix', self.multifile_annotations_suffix_txt.text()))
            self.multifile_segmentation_suffix_txt.textChanged.connect(lambda: setattr(
                self, 'multifile_seg_suffix', self.multifile_segmentation_suffix_txt.text()))

            self.viewer.layers.events.removed.connect(self._on_annotations_changed)
            # self.viewer.window.qt_viewer.canvas.events.mouse_release.connect(self._on_annotations_changed)
            self.viewer.mouse_drag_callbacks.append(self._on_mouse_event)

    def _on_mouse_event(self, viewer, event):
        yield
        while event.type == "mouse_press":
            yield
        if self.annot_tag in self.viewer.layers:
            layer = self.viewer.layers.selection.active
            if layer is not None and layer.name == self.annot_tag:
                self._on_annotations_changed(None)

### Visibility toggles for key bindings

    def toggle_annotation(self, event=None):
        """Hide/unhide annotations layer."""
        annot_layer = self.annotations_layer_selection_widget.value
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
        if not self.seg_tag in self.viewer.layers:
            return
        if self.viewer.layers[self.seg_tag].visible == False:
            self.viewer.layers[self.seg_tag].visible = True
            self.viewer.layers.selection.active = None
            self.viewer.layers.selection.active = self.viewer.layers[self.seg_tag]
        else:
            self.viewer.layers[self.seg_tag].visible = False

    def set_annot_label_class(self, x, event=None):
        """Set the label class of the annotations layer."""
        annot_layer = self.annotations_layer_selection_widget.value
        if annot_layer is not None:
            annot_layer.selected_label = x
            annot_layer.visible = True
            annot_layer.mode = 'paint'
            self.viewer.layers.selection.active = None
            self.viewer.layers.selection.active = annot_layer
        # Also change the selected label for all (not selected) annotations layers and segmentation layers
        labels_layers = self.annot_layers.union(self.seg_layers)
        for l in labels_layers:
            if l is not None and l.name in self.viewer.layers:
                self.viewer.layers[l.name].selected_label = x


### ============== Define the detailed behaviour in the classes tab ==============

    # Classes

    def _create_default_class_names(self):
        """Create the default class names and icons in the layout."""
        # Start with default class names
        for name in self.initial_names:
            self._on_add_class(text=name)
        # Add default annot and seg layers if they exist
        if self.annotations_layer_selection_widget.value is not None:
            self.annot_layers.add(self.annotations_layer_selection_widget.value)
        if self.seg_tag in self.viewer.layers:
            self.seg_layers.add(self.viewer.layers[self.seg_tag])
        # Update all class names and icons
        self.update_all_class_names_and_cmaps()

    def _on_reset_class_names(self):
        """Reset the class names to the default ones and update all annotations and segmentation layers."""

        # Remove and delete all class name widgets and icons
        for name in self.class_names:
            self.classes_layout.removeWidget(name)
            name.deleteLater()
        for icon in self.class_icons:
            self.classes_layout.removeWidget(icon)
            icon.deleteLater()

        self.class_names.clear()
        self.class_icons.clear()

        # Remove the buttons from the layout
        self.classes_layout.removeWidget(self.add_class_btn)
        self.classes_layout.removeWidget(self.remove_class_btn)
        self.classes_layout.removeWidget(self.reset_class_names_btn)
        self.classes_layout.removeWidget(self.btn_class_distribution_annot)

        # Recreate the default class names and icons
        self._create_default_class_names()

        # Re-add the buttons below the class names
        self.classes_layout.addWidget(self.add_class_btn, len(self.class_names)+1, 0, 1, 5)
        self.classes_layout.addWidget(self.remove_class_btn, len(self.class_names)+1, 5, 1, 5)
        self.classes_layout.addWidget(self.export_class_names_btn, len(self.class_names)+2, 0, 1, 5)
        self.classes_layout.addWidget(self.import_class_names_btn, len(self.class_names)+2, 5, 1, 5)
        self.classes_layout.addWidget(self.reset_class_names_btn, len(self.class_names)+3, 0, 1, 10)
        self.classes_layout.addWidget(self.btn_class_distribution_annot, len(self.class_names)+4, 0, 1, 10)

    def _on_add_class(self, text=None):
        """Add a new class name and icon to the layout and update all annotations and segmentation layers."""

        # Create a new class name
        new_name = QtWidgets.QLineEdit()
        new_name.setStyleSheet("font-size: 12px;")
        self.class_names.append(new_name)
        class_num = len(self.class_names)  # Class number is the length of the list
        # Add the new name to the layout
        self.classes_layout.addWidget(new_name, class_num, 1, 1, 9)
        # Set the text of the new name
        text_str = text if text is not None else f'Class {class_num}'
        new_name.setText(text_str)

        # Change "clear" button to the last name and connect it to deleting the entire entry (instead of only text)
        # new_name.setClearButtonEnabled(True)
        # new_name.textChanged.connect(self.remove_class_name)
        # self.class_names[-2].setClearButtonEnabled(False)

        # Connect the new name to the update function
        new_name.textChanged.connect(self._update_class_names)
        
        # Add a new icon
        new_icon = QtWidgets.QLabel()
        self.class_icons.append(new_icon)
        self.classes_layout.addWidget(new_icon, class_num, 0)
        new_icon.mousePressEvent = lambda event: self._set_all_labels_classes(class_num, event)
        
        # Update the icon with the color of the last label and all class names
        self._update_class_icons(class_num)
        self._update_class_names()

        # Move the add, remove and reset buttons one down
        self.classes_layout.removeWidget(self.add_class_btn)
        self.classes_layout.removeWidget(self.remove_class_btn)
        self.classes_layout.removeWidget(self.reset_class_names_btn)
        self.classes_layout.removeWidget(self.btn_class_distribution_annot)
        self.classes_layout.addWidget(self.add_class_btn, class_num+1, 0, 1, 5)
        self.classes_layout.addWidget(self.remove_class_btn, class_num+1, 5, 1, 5)
        self.classes_layout.addWidget(self.export_class_names_btn, class_num+2, 0, 1, 5)
        self.classes_layout.addWidget(self.import_class_names_btn, class_num+2, 5, 1, 5)
        self.classes_layout.addWidget(self.reset_class_names_btn, class_num+3, 0, 1, 10)
        self.classes_layout.addWidget(self.btn_class_distribution_annot, class_num+4, 0, 1, 10)

    def _on_remove_class(self, del_annots=True, event=None):
        """Remove the last class name and icon from the layout and update all annotations and segmentation layers."""
        last_name_idx = len(self.class_names)
        if last_name_idx > 2:
            # Remove the annotations from all annotations layers (do NOT do it in segmentation layers, as this would leave holes)
            if del_annots:
                for layer in self.annot_layers:
                    if layer is not None and layer.name in self.viewer.layers:
                        # Get the annotations image and remove the last label from it
                        label_img = layer.data
                        label_img[label_img == last_name_idx] = 0
                        # Update the layer to show changes immediately
                        layer.refresh()
            # Remove the last label and icon from the layout
            self.class_names[-1].deleteLater()
            self.class_icons[-1].deleteLater()
            self.class_names.pop()
            self.class_icons.pop()
            # Move the buttons one up
            self.classes_layout.removeWidget(self.add_class_btn)
            self.classes_layout.removeWidget(self.remove_class_btn)
            self.classes_layout.removeWidget(self.reset_class_names_btn)
            self.classes_layout.removeWidget(self.btn_class_distribution_annot)
            self.classes_layout.addWidget(self.add_class_btn, len(self.class_names)+1, 0, 1, 5)
            self.classes_layout.addWidget(self.remove_class_btn, len(self.class_names)+1, 5, 1, 5)
            self.classes_layout.addWidget(self.export_class_names_btn, len(self.class_names)+2, 0, 1, 5)
            self.classes_layout.addWidget(self.import_class_names_btn, len(self.class_names)+2, 5, 1, 5)
            self.classes_layout.addWidget(self.reset_class_names_btn, len(self.class_names)+3, 0, 1, 10)
            self.classes_layout.addWidget(self.btn_class_distribution_annot, len(self.class_names)+4, 0, 1, 10)
            # Update the icons and class names
            self._update_class_names()
        else:
            show_info('You need at least two classes.')

    def _update_class_icons(self, class_num=None, event=None):
        """Update the class icons with the colors of the class names.
        If class_num is given, only update the icon of that class."""

        if self.labels_cmap is None:
            return

        cmap = self.labels_cmap.copy()

        if class_num is not None:
            col = cmap.map(class_num)
            pixmap = self.get_pixmap(col)
            self.class_icons[class_num-1].setPixmap(pixmap)
            self.class_icons[class_num-1].mousePressEvent = lambda event: self._set_all_labels_classes(class_num, event)

        # Update all icons with the colors of the class names
        else:
            for i, _ in enumerate(self.class_names):
                cl = i+1
                col = cmap.map(cl)
                pixmap = self.get_pixmap(col)
                self.class_icons[i].setPixmap(pixmap)
                # Bind clicking on the icon to selecting the label
                # self.class_icons[i].mousePressEvent = lambda event, idx=i: self._set_all_labels_classes(idx+1, event)

    def _set_all_labels_classes(self, x, event=None):
        """Set the selected label of all annotations and segmentation layers to x."""
        # For all annotations and segmentation layers added previously, set the selected label to x
        labels_layers = self.annot_layers.union(self.seg_layers)
        for l in labels_layers:
            if l is not None and l.name in self.viewer.layers:
                self.viewer.layers[l.name].selected_label = x

    def _update_class_names(self, event=None):
        """Update the class names for all annotations and segmentation layers."""
        # For all annotations and segmentation layers, set the class names (= layer property) to the ones defined in the widget
        class_names = ["No label"] + [label.text() for label in self.class_names]
        props = {"Class": class_names}
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
        """Update class icons and segmentation colormap when annotations colormap changes."""

        # Avoid infinite loop when changing colormaps
        if self.cmap_flag:
            return
        
        # If there is no cmap yet, define it from the first annotations layer if available
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
        """Connect colormap changes for all annotations layers."""
        labels_layers = self.annot_layers.union(self.seg_layers)
        for l in labels_layers:
            l.events.colormap.connect(lambda event: self._update_cmaps(source_layer=event.source))

    def _on_change_annot_cmap(self, event=None):
        """Update class icons and segmentation colormap when annotations colormap changes."""
        if (self.cmap_flag or # Avoid infinite loop
            (event is not None and event.source != self.annotations_layer_selection_widget.value)): # Only triggers of annotations layer
            return
        # Make sure we are not in a loop
        self.cmap_flag = True
        # Update the colormap of all labels layers according to the changed annotations layer
        if self.annotations_layer_selection_widget is not None:
            self._update_cmaps(source_layer=self.annotations_layer_selection_widget.value)
        # Turn off the flag to allow for new events
        self.cmap_flag = False

    def _on_change_seg_cmap(self, event=None):
        """Update annotations colormap when segmentation colormap changes."""
        if self.cmap_flag:
            return
        if event is not None:
            if self.seg_tag not in self.viewer.layers or event.source != self.viewer.layers[self.seg_tag]:
                return
        # Make sure we are not in a loop
        self.cmap_flag = True
        # Update the colormap of all labels layers according to the segmentation layer
        if self.seg_tag in self.viewer.layers:
            self._update_cmaps(source_layer=self.viewer.layers[self.seg_tag])
        # Turn off the flag to allow for new events
        self.cmap_flag = False

    def update_all_class_names_and_cmaps(self):
        """Update class names, icons and colormaps for all annotations and segmentation layers."""
        # Update class names
        self._update_class_names()
        # Sync all labels layers (annotations and segmentation) between each other
        self._connect_all_cmaps()
        # Sync the new labels' cmaps with the existing ones
        self._update_cmaps() # Also calls _update_class_icons()
        # Update the class icons
        # self._update_class_icons()

    def export_class_names_csv(self, file_path):
        """
        Save current class names to a CSV file.
        CSV columns: index,name
        Index 0 is 'No name', subsequent indices correspond to widget names order.
        """
        if file_path is None:
            raise ValueError("file_path must be provided")

        import csv
        # Build list of label names: include "No label" at index 0
        class_names = ["No label"] + [w.text() for w in self.class_names]

        with open(file_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["index", "name"])
            for idx, name in enumerate(class_names):
                writer.writerow([idx, name])

    def import_class_names_csv(self, file_path):
        """
        Load class names from a CSV produced by `export_class_names_csv`.

        Behavior:
        - Expects rows with columns `index,label` (header optional).
        - Ignores the index==0 row ("No label").
        - Resets widget to defaults (2 labels), then adds extra labels if CSV contains more than 2 labels.
        - If CSV provides fewer than 2 labels, remaining labels are set to "Class N" (e.g. "Class 2").
        """
        if file_path is None:
            raise ValueError("file_path must be provided")

        import csv

        parsed = []
        with open(file_path, newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            rows = list(reader)

        if not rows:
            # empty file -> don't do anything
            return

        # Special case: single-row file with multiple comma-separated values
        # Interpret as names list: e.g. "A,B,C" -> ['A','B','C']
        if len(rows) == 1 and len(rows[0]) > 1:
            parsed = [c.strip() for c in rows[0] if c.strip() != ""]
            # proceed to reset/add using parsed
            rows = []  # skip normal parsing below

        # Skip header if present
        if rows:
            first = rows[0]
            if len(first) >= 2 and first[0].strip().lower() == 'index' and first[1].strip().lower() == 'name':
                rows = rows[1:]
            elif len(first) >= 1 and first[0].strip().lower() == 'name' and len(first) == 1:
                rows = rows[1:]

        num_appended = 0
        for row in rows:
            if not row:
                continue
            # If two columns, treat as index,name
            if len(row) >= 2:
                # try parse index; if non-numeric, treat first column as name
                try:
                    idx = int(row[0])
                except Exception:
                    parsed.append(row[0].strip())
                    num_appended += 1
                    continue
                name = row[1].strip()
                if idx == 0:
                    # skip "No label"
                    continue
                if idx != num_appended + 1: # + 1 because index 0 is "No label"
                    warnings.warn(f"Row {num_appended + 1} of named classes has index {idx}, meaning it is not increasing sequentially. Using the row number as index instead.")
                parsed.append(name)
                num_appended += 1
            # Single column: treat as names
            else:
                val = row[0].strip()
                # ignore empty rows
                if val != "":
                    parsed.append(val)
                    num_appended += 1

        # Reset to defaults (this will create 2 labels)
        self._on_reset_class_names()

        # Add extra labels if CSV contains more than 2
        n_parsed = len(parsed)
        if n_parsed > 2:
            for i in range(n_parsed - 2):
                # add extra empty labels; we'll set texts in the unified loop below
                self._on_add_class()
        # If there are fewer than 2 in the import, rename the 2 base labels to "pad" the imported ones
        elif n_parsed < 2:
            # Rename the 2 base labels to "pad" the imported ones
            self.class_names[0].setText(f'Class 1')
            self.class_names[1].setText(f'Class 2')

        # Overwrite as many label texts as available; pad missing up to 2
        for i in range(0, n_parsed):
            self.class_names[i].setText(parsed[i])

        # Sync labels to layers
        self._update_class_names()
        # Keep icons/cmaps in sync (no color data is read or written)
        try:
            self.update_all_class_names_and_cmaps()
        except Exception:
            # don't fail on cmap sync problems
            pass

    def _export_class_names_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Export class names', filter='CSV/Text Files (*.csv *.txt);;All Files (*)')
        if not path:
            return
        try:
            self.export_class_names_csv(path)
        except Exception as e:
            show_info(f'Failed to export class names: {e}')

    def _import_class_names_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Import class names', filter='CSV/Text Files (*.csv *.txt);;All Files (*)')
        if not path:
            return
        try:
            self.import_class_names_csv(path)
        except Exception as e:
            show_info(f'Failed to import class names: {e}')


### ============== Define the detailed behaviour in the rest of the widget ==============

### HOME TAB

    # Handling of Napari layers

    def _on_insert_layer(self, event=None):
        """Bind the update of layer choices in dropdowns to the renaming of inserted layers."""
        layer = event.value
        layer.events.name.connect(self._update_image_layers)
        layer.events.name.connect(self._update_annotations_layers)
        layer.events.data.connect(self._on_select_layer)

    def _on_layer_removed(self, event=None):
        """When a layer is removed, remove it from the sets of annotations and segmentation layers if it is in there."""
        # keep only layers that still exist in viewer
        self.annot_layers = {l for l in self.annot_layers if l is None or l.name in self.viewer.layers}
        self.seg_layers = {l for l in self.seg_layers if l is None or l.name in self.viewer.layers}

    # Layer selection

    def _on_select_layer(self, newtext=None):
        """Assign the layer to segment and update data radio buttons accordingly"""

        # Check if the selected image has compatible dimensions
        img = self._get_selected_img()
        data_dims = self._get_data_dims(img.data, img.ndim) if img is not None else None
        if data_dims not in self.supported_data_dims:
            warnings.warn(
                f'Non-supported image dimensions {data_dims}. Only '
                f'2D-4D images are supported. Please select a compatible image.'
            )
            self.add_layers_btn.setEnabled(False)
            return

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
                # Activate button to add annotations and segmentation layers
                self.add_layers_btn.setEnabled(True)
                # Give info if image is very large to use "tile image"
                if self._check_large_image(img) and not self.cp_model.get_param('tile_image'):
                    show_info('Image is very large. Consider using tiling and/or downsampling.')
                # If we have continuous training within a single image, reset the training features
                if self.cont_training == "Image" or self.cont_training == "Off":
                    self._reset_train_features()
                self.update_layer_flag = True
                self.data_shape = img.data.shape
        else:
            self.add_layers_btn.setEnabled(False)

        # If the option is activated, select annotations layer according to the prefix and image name
        if self.auto_select_annot and not getattr(self, "_block_layer_select", False):
            self._auto_select_annot_layer()

        # Allow other methods again to add layers if that was the case before
        self.add_layers_flag = initial_add_layers_flag

        # Check if the selected annotations layer is suited, if one is selected
        labels_layer = self.annotations_layer_selection_widget.value
        if labels_layer is not None and not self._approve_annotations_layer_shape(labels_layer, img):
            warnings.warn('annotations layer has wrong shape for the selected data')

    def _delayed_on_select_layer(self, event=None):
        """Delay the selection of the image layer to allow for napari operations to happen first."""
        self._block_layer_select = False
        QTimer.singleShot(100, lambda: self._on_select_layer())
        # Only set the block flag again after some time, so auto selection is triggered, but only once
        QTimer.singleShot(200, lambda: setattr(self, "_block_layer_select", True))

    def _auto_select_annot_layer(self):
        """Automatically select an annotations layer according to the prefix and image name."""

        # Set the name of the annotations accordingly
        annot_name = f"{self.annot_tag}_{self.selected_channel}"
        # Check if there are fitting labels layers present in the viewer
        found_layers = [layer.name for layer in self.viewer.layers # Take all layers
                        if isinstance(layer, napari.layers.Labels) # that are labels layers
                        and layer.name[:len(annot_name)] == annot_name] # and start with the prefix
        if self.annot_tag in self.viewer.layers and isinstance(self.viewer.layers[self.annot_tag], napari.layers.Labels):
            # If there is a layer with the name "annotations", set it as the annotations layer
            annot_name = self.annot_tag
        elif len(found_layers) > 0:
            if len(found_layers) > 1:
                show_info('Multiple annotations layers found. The first one (alphabetically) is chosen.')
            # Sort and take first one
            found_layers.sort()
            annot_name = found_layers[0]
        else:
            show_info('No annotations layer found. Please create one.')
            return

        self.viewer.layers.selection.active = self.viewer.layers[annot_name]
        self.annotations_layer_selection_widget.value = self.viewer.layers[annot_name]
        self.viewer.layers[annot_name].mode = 'paint'
        self.viewer.layers[annot_name].brush_size = self.default_brush_size
        self._on_select_annot()

        # Hide all layers except the image and the annotations layer
        for layer in self.viewer.layers:
            if layer.name != self.selected_channel and layer.name != annot_name:
                layer.visible = False
            else:
                layer.visible = True

    def _on_select_annot(self, newtext=None):
        """Check if annotations layer dimensions are compatible with image, and raise warning if not."""
        
        labels_layer = self.annotations_layer_selection_widget.value
        
        # Handle labels classes
        self.annot_layers.add(self.annotations_layer_selection_widget.value)
        # labels_layer.events.colormap.connect(self._on_change_annot_cmap)
        self.update_all_class_names_and_cmaps()
        
        # Check shape
        if self.image_layer_selection_widget.value is not None:
            img = self.image_layer_selection_widget.value
            if not self._approve_annotations_layer_shape(labels_layer, img):
                warnings.warn('annotations layer has wrong shape for the selected data')

    def _on_add_annot_layer(self, event=None, force_add=True):
        """Add empty annotations and segmentation layers if not already present."""
        self._add_empty_annot(event, force_add)
        img = self._get_selected_img(check=True)
        if img is None:
            return
        labels_layer = self.annotations_layer_selection_widget.value
        labels_layer.events.colormap.connect(self._on_change_annot_cmap)
        self.update_all_class_names_and_cmaps()

    # Train

    def _on_train(self, event=None):
        """Given a set of new annotations, update the CatBoost classifier."""

        # Get the data
        img = self._get_selected_img(check=True)
        annot = self.annotations_layer_selection_widget.value
        mem_mode = (self.cont_training == "Image"
                    or self.cont_training == "Global")

        # Check if annotations of at least 2 classes are present
        if annot is None:
            raise Exception('No annotations layer selected. Please create/select one.')
        unique_labels = np.unique(annot.data)
        unique_labels = unique_labels[unique_labels != 0]
        if len(unique_labels) < 2:
            if not mem_mode:
                raise Exception('You need annotations for at least foreground and background')
            if self.cp_model.num_trainings == 0:
                raise Exception('Model has not yet been trained. You need annotations for at least foreground and background')

        # Check if annotations layer has correct shape for the chosen data type
        if not self._approve_annotations_layer_shape(annot, img):
            raise Exception('annotations layer has wrong shape for the chosen data')

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
            _ = self.cp_model.train(image_stack_norm, annot, memory_mode=mem_mode, img_ids=img_name,
                                    in_channels=in_channels, skip_norm=False,
                                    fe_use_device=self.fe_device, clf_use_device=self.clf_device)
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
            
            # Check dimensionality
            img = self._get_selected_img()
            data_dims = self._get_data_dims(img.data, img.ndim) if img is not None else None
            if data_dims not in self.supported_data_dims:
                warnings.warn(f'Non-supported image dimensions {data_dims}. Prediction not performed.')
                return
            
            # Get the data
            image_plane = self._get_current_plane_norm()
            in_channels = self._parse_in_channels(self.input_channels)

            # Predict image (use backend function which returns probabilities and segmentation); skip norm as it is done above
            probas, segmentation = self.cp_model._predict(image_plane, add_seg=True, in_channels=in_channels, skip_norm=True,
                                                          use_dask=self.use_dask, fe_use_device=self.fe_device)

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(False)

        # Get the current step in case of stacks
        step = self.viewer.dims.current_step[-3] if data_dims in ['3D_single', '4D', '3D_RGB'] else None

        # Add segmentation layer if enabled
        if self.add_seg:
            # Check if we need to create a new segmentation layer
            self._check_create_segmentation_layer()
            # Set the flag to False, so we don't create a new layer every time
            self.new_seg = False

            # Update segmentation layer
            if data_dims in ['2D', '2D_RGB', '3D_multi']:
                self.viewer.layers[self.seg_tag].data = segmentation
            elif data_dims in ['3D_single', '4D', '3D_RGB']: # seg has no channel dim -> z is first
                self.viewer.layers[self.seg_tag].data[step] = segmentation
            # Case `data_dims is None` and other invalid cases are already caught above, so we don't need an else statement here
            self.viewer.layers[self.seg_tag].refresh()

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
            elif data_dims in ['3D_single', '4D', '3D_RGB']: # (stack dim is second, probas first)
                self.viewer.layers[self.proba_prefix].data[:, step] = probas
            # Case `data_dims is None` and other invalid cases are already caught above, so we don't need an else statement here
            self.viewer.layers[self.proba_prefix].refresh()

    def _on_get_feature_image(self, event=None):
        """Get the feature image for the currently viewed frame based
        on the current feature extractor and show it in a new layer."""

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(True)

        with progress(total=0) as pbr:
            pbr.set_description(f"Feature extraction")

            # Check dimensionality
            img = self._get_selected_img()
            data_dims = self._get_data_dims(img.data, img.ndim) if img is not None else None
            if data_dims not in self.supported_data_dims:
                warnings.warn(f'Non-supported image dimensions {data_dims}. Feature extraction not performed.')
                return

            # Get the data
            image_plane = self._get_current_plane_norm()
            in_channels = self._parse_in_channels(self.input_channels)

            # Check and parse PCA and Kmeans parameters
            pca, kmeans = self._check_parse_pca_kmeans()

            # Get feature image; skip norm as it is done above
            feature_image = self.cp_model.get_feature_image(image_plane, in_channels=in_channels, skip_norm=True,
                                                            pca_components=pca, kmeans_clusters=kmeans,
                                                            use_device=self.fe_device)

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(False)

        # Check if we need to create a new features layer
        num_features = feature_image.shape[0] if not kmeans else 0
        self._check_create_features_layer(num_features)
        # Set the flag to False, so we don't create a new layer every time
        self.new_features = False

        # Update features layer
        if data_dims in ['2D', '2D_RGB', '3D_multi']: # No stack dim
            self.viewer.layers[self.features_prefix].data = feature_image
        elif data_dims in ['3D_single', '4D', '3D_RGB']: # stack dim is third last
            step = self.viewer.dims.current_step[-3]
            self.viewer.layers[self.features_prefix].data[..., step, :, :] = feature_image
        # Case `data_dims is None` and other invalid cases are already caught above, so we don't need an else statement here
        self.viewer.layers[self.features_prefix].refresh()

    def _on_predict_all(self):
        """Predict the segmentation of all frames based 
        on a classifier model trained with annotations."""
        
        # Get the data
        img = self._get_selected_img(check=True)

        # Check dimensionality
        data_dims = self._get_data_dims(img.data, img.ndim) if img is not None else None
        if data_dims not in ['3D_single', '3D_RGB', '4D']:
            warnings.warn(f'Image stack has wrong dimensionality ({data_dims}) for predicting stacks. Prediction not performed.')
            return
        
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
            probas, seg = self.cp_model._predict(image, add_seg=True, in_channels=in_channels, skip_norm=True,
                                                 use_dask=self.use_dask, fe_use_device=self.fe_device)

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
                self.viewer.layers[self.seg_tag].data[step] = seg
                self.viewer.layers[self.seg_tag].refresh()
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

        # Check dimensionality
        data_dims = self._get_data_dims(img.data, img.ndim) if img is not None else None
        if data_dims not in ['3D_single', '3D_RGB', '4D']:
            warnings.warn(f'Image stack has wrong dimensionality ({data_dims}) for processing stacks. Feature extraction not performed.')
            return
        
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
                                                                pca_components=pca, kmeans_clusters=kmeans,
                                                                use_device=self.fe_device)

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
                                                                pca_components=pca, kmeans_clusters=kmeans,
                                                                use_device=self.fe_device)

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
            # DontUseNativeDialog: ensures extension is appended on all platforms
            dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
            save_file, selected_filter = dialog.getSaveFileName(self, "Save model", None, "PICKLE (*.pkl);;YAML (*.yml)")
        
        # If file selection is aborted, raise a warning (instead of an error)
        if save_file == '':
            warnings.warn('No file selected')
            return

        # Determine path/name and suffix
        save_file = Path(save_file)
        suff = save_file.suffix

        # Ensure we have a valid extension
        # Note: this is originally a fix for Ubuntu/GTK where native dialog does not auto-append the extension
        # (together with switching off the native dialog), but we keep it for all platforms to ensure the correct extension is always appended
        if suff not in ('.pkl', '.yml') and selected_filter is not None:
            if 'pkl' in selected_filter:
                save_file = save_file.with_suffix('.pkl')
            else:
                save_file = save_file.with_suffix('.yml')
            suff = save_file.suffix

        # Save model
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
        new_model = self._cpm_class(model_path=save_string)
        new_param = new_model.get_params()
        
        # Check if the new multichannel setting is incompatible with data
        img = self._get_selected_img()
        data_dims = self._get_data_dims(img.data, img.ndim) if img is not None else None
        channel_mode = new_param.channel_mode
        if data_dims in ['2D'] and channel_mode in ['rgb', 'multi']:
            warnings.warn(f'The loaded model works with {channel_mode} data, but the data is {data_dims}. ' +
                            'This might cause problems.')
        # Check if the loaded normalization setting is compatible with data
        if data_dims in ['2D', '2D_RGB', '3D_multi'] and new_param.normalize == 2:
            warnings.warn(f'The loaded model normalizes over stack, but the data is {data_dims}. ' +
                            'This might cause problems.')

        # Update GUI to show selectable layers and scalings of model chosen from drop-down
        # self._update_gui_fe_layer_keys(new_model.get_fe_layer_keys())
        # self._update_gui_fe_proposed_scalings(new_model.get_fe_proposed_scalings())

        # Update GUI with the new parameters
        self._update_gui_from_params(new_param) # This triggers _on_fe_selected()
        self._update_gui_fe_layers(layers=new_param.fe_layers) # Override defaults with saved values
        self._update_gui_fe_scalings(scalings=new_param.fe_scalings) # Override defaults with saved values

        # Load the model (Note: done after updating GUI, since GUI updates might reset clf or change model)
        self.cp_model = new_model
        self.cp_model._param = new_param
        temp_fe_model = self._cpm_class.create_fe(new_param.fe_name)
        self.temp_fe_description = temp_fe_model.get_description()
        lks = temp_fe_model.get_layer_keys()
        self.temp_fe_layer_keys = lks.copy() if lks is not None else None
        pps = temp_fe_model.get_proposed_scalings()
        self.temp_fe_proposed_scalings = pps.copy() if pps is not None else None

        # Adjust trained flag, save button, predict buttons etc., and update model description
        self.trained = save_file.suffix == '.pkl' and new_model.classifier is not None
        # self.save_model_btn.setEnabled(True)
        self._reset_predict_buttons()
        self.current_model_path = save_file.name
        self._set_model_description()
        self._update_training_counts()
        self._reset_device_options()
        self.flag_fe_as_set()
        self.flag_clf_as_set()

    # Image Processing

    def _on_channel_mode_changed(self):
        """Set the image data dimensions based on radio buttons,
        reset classifier, and adjust normalization options."""
        old_channel_mode = self.cp_model.get_param("channel_mode")
        new_channel_mode = "multi" if self.radio_multi_channel.isChecked() else \
                       "rgb" if self.radio_rgb.isChecked() else \
                       "single"
        self.cp_model.set_param("channel_mode", new_channel_mode, ignore_warnings=True)
        if old_channel_mode != new_channel_mode:
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

        # Remove class names (note, resetting of class names needs to be split because of the handling of the attributes)
        if 'Classes' in self.tab_names:
            for name in self.class_names:
                self.classes_layout.removeWidget(name)
                name.deleteLater()
            for icon in self.class_icons:
                self.classes_layout.removeWidget(icon)
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
        {"Off": lambda: self.radio_single_training.setChecked(True),
         "Image": lambda: self.radio_img_training.setChecked(True),
         "Global": lambda: self.radio_global_training.setChecked(True)}[self.cont_training]()
        # self.check_cont_training.setChecked(self.cont_training)
        self.check_use_dask.setChecked(self.use_dask)
        self.text_input_channels.setText(self.input_channels)
        self.check_add_seg.setChecked(self.add_seg)
        self.check_add_probas.setChecked(self.add_probas)
        self.text_features_pca.setText(self.features_pca_components)
        self.text_features_kmeans.setText(self.features_kmeans_clusters)
        # Reset the model description
        self._set_model_description()

        if 'Classes' in self.tab_names:
            # Remove the buttons from the layout
            self.classes_layout.removeWidget(self.add_class_btn)
            self.classes_layout.removeWidget(self.remove_class_btn)
            self.classes_layout.removeWidget(self.reset_class_names_btn)
            self.classes_layout.removeWidget(self.btn_class_distribution_annot)

            # Recreate the default class names and icons
            self._create_default_class_names()

            # Re-add the buttons below the class names
            self.classes_layout.addWidget(self.add_class_btn, len(self.class_names)+1, 0, 1, 5)
            self.classes_layout.addWidget(self.remove_class_btn, len(self.class_names)+1, 5, 1, 5)
            self.classes_layout.addWidget(self.export_class_names_btn, len(self.class_names)+2, 0, 1, 5)
            self.classes_layout.addWidget(self.import_class_names_btn, len(self.class_names)+2, 5, 1, 5)
            self.classes_layout.addWidget(self.reset_class_names_btn, len(self.class_names)+3, 0, 1, 10)
            self.classes_layout.addWidget(self.btn_class_distribution_annot, len(self.class_names)+4, 0, 1, 10)
        
        if 'Multifile' in self.tab_names:
            self._reset_multifile_folder()
            self.check_open_import_annotations.setChecked(self.multifile_import_open_annotations)
            self.check_open_import_segmentations.setChecked(self.multifile_import_open_segmentations)
            self.multifile_annotations_suffix_txt.setText(self.multifile_annot_suffix)
            self.multifile_segmentation_suffix_txt.setText(self.multifile_seg_suffix)

        # Re-apply device dropdown state after attributes reset potentially changed policies.
        self._reset_device_options()

        # Turn on layer creation again
        self.add_layers_flag = True

    def _reset_model(self):
        """Reset the model to default."""
        # self.save_model_btn.setEnabled(True)
        self._reset_fe_params() # Also resets the FE model
        self._reset_clf_params()
        self._reset_clf()
        self._reset_default_general_params()
        # self._update_gui_fe_layer_keys(self.cp_model.fe_model.get_layer_keys())
        # self._update_gui_fe_scalings(self.cp_model.get_fe_proposed_scalings()())
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
        self.supported_data_dims = ('2D', '2D_RGB', '3D_RGB', '3D_multi', '3D_single', '4D') # Supported data dimensionalities (for checks and warnings)
        self.current_model_path = 'not trained' # Path to the current model (if saved)
        self.auto_add_layers = True # Automatically add layers when a new image is selected
        self.keep_layers = False # Keep old layers when adding new ones
        self.auto_select_annot = False # Automatically select annotations layer based on image name
        self.annot_tag = 'annotations' # Prefix for the annotations layer names
        self.seg_tag = 'segmentation' # Prefix for the segmentation layer names
        self.proba_prefix = 'probabilities' # Prefix for the class probabilities layer names
        self.features_prefix = 'features' # Prefix for the feature image layer name
        self.cont_training = "Image" # Update features for subsequent training ("Image" or "Off" or "Global")
        self.use_dask = False # Use Dask for parallel processing
        self.fe_device = 'auto' # Device to use for the FE (if applicable); 'auto' will use GPU if available, otherwise CPU
        self.clf_device = 'auto' # Device to use for the classifier (if applicable); 'auto' will use GPU if available, otherwise CPU
        self.input_channels = "" # Input channels for the model (as txt, will be parsed)
        self.add_seg = True # Add a layer with segmentation
        self.add_probas = False # Add a layer with class probabilities
        self.new_seg = True # Flags to indicate if new outputs are created
        self.new_proba = True
        self.new_features = True
        self.features_pca_components = "0" # Number of PCA components for feature image (0 = no PCA)
        self.features_kmeans_clusters = "0" # Number of k-means clusters for feature image (0 = no k-means)
        self.initial_names = ['Background', 'Foreground']
        self.annot_layers = set() # List of annotations layers
        self.seg_layers = set() # List of segmentation layers
        self.class_names = [] # List of class names
        self.class_icons = [] # List of class icons
        self.cmap_flag = False # Flag to prevent infinite loops when changing colormaps
        self.labels_cmap = None # Colormap for the labels (annotations and segmentation)
        self._block_layer_select = True # Flag to block layer selection events temporarily
        # Multifile attributes
        self._multifile_warned = False # Whether the user has already seen the "remove existing layers" warning (for Multifile)
        self._multifile_annotations_store = {} # Store for in-memory and saved annotations keyed by filename
        self._multifile_segmentation_store = {} # Store for saved segmentations keyed by filename
        self._multifile_last_folder = '' # Last used folder for images/annotations dialogs (used as default in folder pickers)
        self._multifile_update_annot_tick = True # Flag to adjust updating tick when opening images
        self._multifile_last_annot = None
        # self.multifile_import_warned = False # Whether the user has already seen the "import annotations" warning (for Multifile)
        self._current_multifile_filename = None # Current multifile-opened filename (when opened via Multifile UI)
        self.store_annot = False # Flag whether to store annotations in the in-memory store (for Multifile)
        self.multifile_import_open_annotations = True # Whether to open annotations when opening images via Multifile
        self.multifile_import_open_segmentations = True # Whether to open segmentations when opening images via Multifile
        self.multifile_annot_suffix = 'annotations' # Suffix for annotations files in Multifile
        self.multifile_seg_suffix = 'segmentation' # Suffix for segmentation files in

### Model Tab

    # FE selection and parameters

    def flag_fe_as_temp(self):
        """Whenever we change the temp FE, we want to flag that the changes are not yet applied.
        We do this by writing the "Set feature extractor*" in red."""
        self.set_fe_btn.setText('Set feature extractor *')
        # temp_css_color ="rgb(185, 85, 75)" # Light red
        # temp_css_weight = "bold" # Bold...
        self.set_fe_btn.setStyleSheet("font-weight: bold")
        # self.model_description1.setStyleSheet(f"color: {css_color}")
        # self.model_description2.setStyleSheet(f"color: {temp_css_color}")
        self.fe_group.gbox.setTitle("Feature extractor (unsaved changes) *")

    def flag_fe_as_set(self):
        """Whenever we set the FE, we want to flag that the changes are applied.
        We do this by writing the "Set feature extractor" with the default theme color."""
        self.set_fe_btn.setText('Set feature extractor')
        self.set_fe_btn.setStyleSheet("color: gray")
        # self.model_description1.setStyleSheet("")
        # self.model_description2.setStyleSheet("")
        self.fe_group.gbox.setTitle("Feature extractor")

    def _on_fe_selected(self, event=None):
        """Update GUI to show selectable layers of model chosen from drop-down."""

        # Create a temporary model to get the layers (to display) and default parameters
        new_fe_type = self.qcombo_fe_type.currentText()
        temp_fe_model = self._cpm_class.create_fe(new_fe_type)
        self.temp_fe_description = temp_fe_model.get_description()
        lks = temp_fe_model.get_layer_keys()
        self.temp_fe_layer_keys = lks.copy() if lks is not None else None
        pps = temp_fe_model.get_proposed_scalings()
        self.temp_fe_proposed_scalings = pps.copy() if pps is not None else None

        # Get the default FE params for the temp model and update the GUI
        self.temp_fe_defaults = temp_fe_model.get_default_params()
        fe_defaults = self.temp_fe_defaults

        # Update the GUI to show the FE layers of the temp model
        self._update_gui_fe_layer_keys(self.temp_fe_layer_keys)
        self._update_gui_fe_layers(fe_defaults.fe_layers)
        # Same for scalings
        self._update_gui_fe_proposed_scalings(self.temp_fe_proposed_scalings)
        self._update_gui_fe_scalings(fe_defaults.fe_scalings)

        # NOTE: only FE params are adjusted here, since the FE is not set yet
        val_to_setter = {
            "fe_name": self.qcombo_fe_type.setCurrentText,
            "fe_order": self.spin_interpolation_order.setValue,
            "fe_use_min_features": self.check_use_min_features.setChecked,
        }
        for attr, setter in val_to_setter.items():
            val = getattr(fe_defaults, attr, None)
            if val is not None:
                if isinstance(val, list): val = str(val)
                setter(val)
        self.FE_description.setText(self.temp_fe_description)

        self.flag_fe_as_temp() # Flag that the FE is not yet set (since we just changed the temp model)

    def _on_fe_layer_selection_changed(self):
        """Enable the set button based on the model type."""
        if self.temp_fe_layer_keys is not None:
            selected_layers = self._get_selected_layer_names()
            if len(selected_layers) == 0:
                self.set_fe_btn.setEnabled(False)
            else:
                self.set_fe_btn.setEnabled(True)
        else:
            self.set_fe_btn.setEnabled(True)

        self.flag_fe_as_temp() # Flag that the FE is not yet set (since we just changed the temp model)

    def _on_set_fe_model(self, event=None):
        """Create a neural network model that will be used for feature extraction and
        reset the classifier."""

        # Read FE parameters from the GUI
        new_param = self.cp_model.get_params() # Take the current and adjust FE-specific ones
        new_layers = self._get_selected_layer_names() # includes re-writing to keys
        scalings = self._get_selected_scaling_factors()
        if scalings is None:
            warnings.warn('FE scalings could not be parsed. Setting to [1].')
            scalings = [1]
            self._update_gui_fe_scalings(scalings)
        new_param.set(fe_name = self.qcombo_fe_type.currentText(),
                      fe_layers = new_layers,
                      fe_scalings = scalings,
                      fe_order = self.spin_interpolation_order.value(),
                      fe_use_min_features = self.check_use_min_features.isChecked())

        # Get default non-FE params from temp model and update the GUI (also setting the params)
        fe_defaults = self.temp_fe_defaults
        adjusted_params = [] # List of adjusted parameters for raising a warning
        img = self._get_selected_img()
        data_dims = self._get_data_dims(img.data, img.ndim) if img is not None else None
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
            else: # If data is compatible, set the model's default multichannel setting; also assume this in case data_dims is None/invalid
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
        self.cp_model = self._cpm_class(param=new_param)
        self._reset_device_options()
        self._reset_clf() # Call to take all actions needed after resetting the clf
        # Reset the features for continuous training
        self._reset_train_features()
        # Flag that the FE is now set (since we just set the model)
        self.flag_fe_as_set() # Flag that the FE is now set (since we just set the model)

    def _on_reset_default_fe(self, event=None):
        """Reset the feature extraction model to the default model."""
        # self._update_gui_fe_layers(self.default_layer_keys)
        # self._update_gui_fe_scalings(self.default_proposed_scalings)
        self._reset_fe_params() # Note: this calls _on_set_fe_model() --> also flags the FE as set and resets the clf

    # Classifier

    def flag_clf_as_temp(self):
        """Flag classifier settings as changed but not yet applied."""
        self.set_clf_btn.setText('Set classifier parameters *')
        self.set_clf_btn.setStyleSheet("font-weight: bold")
        self.classifier_params_group.gbox.setTitle("Classifier (CatBoost) (unsaved changes) *")

    def flag_clf_as_set(self):
        """Flag classifier settings as applied."""
        self.set_clf_btn.setText('Set classifier parameters')
        self.set_clf_btn.setStyleSheet("color: gray")
        self.classifier_params_group.gbox.setTitle("Classifier (CatBoost)")

    def _on_set_clf_params(self, event=None):
        """Apply classifier parameters from the GUI and discard any trained classifier."""
        self.cp_model.set_param("clf_iterations", self.spin_iterations.value(), ignore_warnings=True)
        self.cp_model.set_param("clf_learning_rate", self.spin_learning_rate.value(), ignore_warnings=True)
        self.cp_model.set_param("clf_depth", self.spin_depth.value(), ignore_warnings=True)
        self._reset_clf()
        self._reset_train_features()
        self.flag_clf_as_set()

    def _on_reset_clf_params(self):
        """Reset the classifier parameters to the default values
        and discard the trained model."""
        self._reset_clf_params()


### Helper functions

    def _get_layer_transform_kwargs(self, img_layer, num_spatial_dims, num_leading_dims=0):
        """Get transform kwargs from image layer for creating a derived layer.

        - Take the trailing `num_spatial_dims` entries from the image layer's
          `scale`/`translate` (if present).
        - Prepend `num_leading_dims` neutral entries (1.0 for scale, 0.0 for
          translate) for layers that add a leading axis (e.g. features/probas).
        """
        if img_layer is None:
            return {}

        # Use numpy to coerce array-like transforms into tuples and avoid
        # ambiguous truth-value checks (e.g. empty lists, or None)
        raw_scale = getattr(img_layer, "scale", None)
        raw_translate = getattr(img_layer, "translate", None)

        if raw_scale is None:
            scale = (1.0,) * num_spatial_dims
        else:
            scale = tuple(np.asarray(raw_scale).tolist())

        if raw_translate is None:
            translate = (0.0,) * num_spatial_dims
        else:
            translate = tuple(np.asarray(raw_translate).tolist())

        tail_scale = tuple(scale[-num_spatial_dims:]) if num_spatial_dims else ()
        tail_translate = tuple(translate[-num_spatial_dims:]) if num_spatial_dims else ()
        leading_scale = (1.0,) * num_leading_dims
        leading_translate = (0.0,) * num_leading_dims

        return {"scale": leading_scale + tail_scale, "translate": leading_translate + tail_translate}

    def _add_empty_annot(self, event=None, force_add=True, from_multifile=False):
        """Add annotations layer to viewer. If the layer already exists,
        remove it (or rename and keep it if specified in self.keep_layers) and add a new one.
        If the widget is used as third party (self.third_party=True), no layer is added if it didn't exist before,
        unless force_add=True (e.g. when the user clicks on the add layer button)"""

        img = self._get_selected_img(check=True)
        if img is None:
            warnings.warn('No image selected. No layers added.')
            return
        layer_shape = self._get_annot_shape(img)

        # Get transform kwargs (scale/translate) from the image layer to apply to new layers
        num_spatial = len(layer_shape)
        transform_kwargs = self._get_layer_transform_kwargs(img, num_spatial_dims=num_spatial, num_leading_dims=0)

        # Create a new annotations layer if it doesn't exist yet
        annotations_exists = self.annot_tag in self.viewer.layers

        if (not self.third_party) | (self.third_party & annotations_exists) | (force_add):
            # Create a temp named annotations layer, so we can select it before renaming the old one
            # (and not trigger a problem with incompatible layer shapes)
            temp_name = self._get_unique_layer_name("temp_annotations")
            self.viewer.add_labels(
                data=np.zeros((layer_shape), dtype=np.uint8),
                name=temp_name,
                **transform_kwargs
                )

            # Select the temp layer in the dropdown
            if self.viewer.layers[temp_name] in self.annotations_layer_selection_widget.choices:
                self.annotations_layer_selection_widget.value = self.viewer.layers[temp_name]
            else:
                warnings.warn(f'The temporary annotations layer {temp_name} could not be selected. ' +
                              'This might cause problems with the annotations layer creation.')
            # If we replace a current layer, we can backup the old one, and replace it
            if annotations_exists:
                # Backup the old annotations layer if keep_layers is set (and the layer exists)
                if self.keep_layers:
                    self._rename_annot_for_backup()
                # Otherwise, just remove the old annotations layer
                else:
                    self.viewer.layers.remove(self.annot_tag)
            # Create a copy of the temp layer with the original name (so we can select it)
            self.viewer.add_labels(
                data=self.viewer.layers[temp_name].data,
                name=self.annot_tag,
                **transform_kwargs
                )
            new_annot_layer = self.viewer.layers[self.annot_tag]
            # Select the new annotations layer in the dropdown
            if new_annot_layer in self.annotations_layer_selection_widget.choices:
                self.annotations_layer_selection_widget.value = new_annot_layer
            else:
                warnings.warn(f'The annotations layer {self.annot_tag} could not be selected. ' +
                              'This might cause problems with the annotations layer creation.')
            # Remove the temp layer
            self.viewer.layers.remove(temp_name)

            # Save information about the annotations layer to be able to rename it later
            self._set_old_annot_tag()
            
            # Add it to the list of layers where class names shall be updated
            self.annot_layers.add(new_annot_layer)

        # Activate the annotations layer, select it in the dropdown and activate paint mode
        if self.annot_tag in self.viewer.layers:
            self.viewer.layers.selection.active = new_annot_layer
            self.annotations_layer_selection_widget.value = new_annot_layer
            new_annot_layer.mode = 'paint'
            new_annot_layer.brush_size = self.default_brush_size

        # Sync the class names
        self.update_all_class_names_and_cmaps()

        # Track annotations data changes to keep in-memory store in sync (for Multifile)
        self.store_annot = from_multifile # Only store if the annot was added from Multifile, to avoid storing unnecessarily when not using Multifile

    def _check_create_segmentation_layer(self):
        """Check if segmentation layer exists and create it if not."""
        
        img = self._get_selected_img(check=True)
        if img is None:
            warnings.warn('No image selected. No layers added.')
            return
        layer_shape = self._get_annot_shape(img)
        num_spatial = len(layer_shape)
        transform_kwargs = self._get_layer_transform_kwargs(img, num_spatial_dims=num_spatial, num_leading_dims=0)
    
        # Create a new segmentation layer if it doesn't exist yet or we need a new one
        seg_exists = self.seg_tag in self.viewer.layers

        # If we replace a current layer, we can backup the old one, and remove it
        if self.new_seg & seg_exists:
            # Backup the old segmentation layer if keep_layers is set (and the layer exists)
            if self.keep_layers:
                self._rename_seg_for_backup()
            # Otherwise, just remove the old segmentation layer
            else:
                self.viewer.layers.remove(self.seg_tag)

        # If there was no segmentation layer, or we need a new one, create it
        if (not seg_exists) or self.new_seg:
            self.viewer.add_labels(
                data=np.zeros((layer_shape), dtype=np.uint8),
                name=self.seg_tag,
                **transform_kwargs
                )
            # Save information about the segmentation layer to be able to rename it later
            self._set_old_seg_tag()
            
            # Add it to the list of layers where class names shall be updated
            self.seg_layers.add(self.viewer.layers[self.seg_tag])
            self.update_all_class_names_and_cmaps()

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
            # probabilities have a leading class dimension then spatial dims
            num_spatial = len(spatial_dims)
            kwargs = self._get_layer_transform_kwargs(img, num_spatial_dims=num_spatial, num_leading_dims=1)
            self.viewer.add_image(
                data=np.zeros(num_classes+spatial_dims, dtype=np.float32),
                name=self.proba_prefix,
                **kwargs
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
                # features have a leading features dimension then spatial dims
                num_spatial = len(spatial_dims)
                kwargs = self._get_layer_transform_kwargs(img, num_spatial_dims=num_spatial, num_leading_dims=1)
                self.viewer.add_image(
                    data=np.zeros((num_features,)+spatial_dims, dtype=np.float32),
                    name=self.features_prefix,
                    **kwargs
                    )
                # Change the colormap to one suited for features
                self.viewer.layers[self.features_prefix].colormap = "viridis"
            else: # Kmeans feature image (2D or 3D)
                # kmeans features are labels with same spatial dims
                num_spatial = len(spatial_dims)
                kwargs = self._get_layer_transform_kwargs(img, num_spatial_dims=num_spatial, num_leading_dims=0)
                self.viewer.add_labels(
                    data=np.zeros(spatial_dims, dtype=np.uint8),
                    name=self.features_prefix,
                    **kwargs
                    )
            # Save information about the features layer to be able to rename it later
            self._set_old_features_tag()

    def _rename_annot_for_backup(self):
        """Name the annotations with a unique name according to its image,
        so it can be kept when adding a new with the standard name."""
        # Rename the layer to avoid overwriting it
        if self.annot_tag in self.viewer.layers:
            full_name = self._get_unique_layer_name(self.old_annot_tag)
            self.viewer.layers[self.annot_tag].name = full_name
            # Add it to the layers where class names shall be updated
            self.annot_layers.add(self.viewer.layers[full_name])

    def _rename_seg_for_backup(self):
        """Name the segmentation layer with a unique name according to its image,
        so it can be kept when adding a new with the standard name."""
        # Rename the layer to avoid overwriting it
        if self.seg_tag in self.viewer.layers:
            full_name = self._get_unique_layer_name(self.old_seg_tag)
            self.viewer.layers[self.seg_tag].name = full_name
            # Add it to the list of layers where class names shall be updated
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
        """Set the old annotations tag based on the current image layer and data dimensions.
        This is used to rename old annotations layers when creating new ones."""
        self.old_annot_tag = f"{self.annot_tag}_{self._get_old_data_tag()}"

    def _set_old_seg_tag(self):
        """Set the old segmentation tag based on the current image layer and data dimensions.
        This is used to rename old segmentation layers when creating new ones."""
        self.old_seg_tag = f"{self.seg_tag}_{self._get_old_data_tag()}"

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
        else: # Unsupported number of dimensions
            for x in self.channel_buttons: x.setEnabled(False)

    def _reset_radio_norm_choices(self, event=None):
        """Set radio buttons for normalization active/inactive depending on selected image type.
        If current parameter is not valid, set a default."""
        # Reset the stats; not necessary, since they are recalculated anyway if changing the norm mode
        # self.image_mean, self.image_std = None, None

        if self.image_layer_selection_widget.value is None:
            for x in self.norm_buttons: x.setEnabled(False)
            return
        
        img = self._get_selected_img()
        data_dims = self._get_data_dims(img.data, img.ndim) if img is not None else None
        if data_dims not in self.supported_data_dims:
            warnings.warn(f'Non-supported image dimensions {data_dims}. Normalization buttons not updated.')
            return
        norm_scope = self.cp_model.get_param("normalize")
        
        if data_dims in ['2D', '2D_RGB', '3D_multi']: # No z dim available -> no stack norm
            self.radio_no_normalize.setEnabled(True)
            self.radio_normalize_over_stack.setEnabled(False)
            self.radio_normalize_by_image.setEnabled(True)
            if norm_scope == 2: # If initially over stack, reset to by image
                self.radio_normalize_by_image.setChecked(True)
            else: # Otherwise, keep the current setting
                self.button_group_normalize.button(norm_scope).setChecked(True)
        elif data_dims in ['3D_single', '3D_RGB', '4D']: # With z dim available -> all options
            self.radio_no_normalize.setEnabled(True)
            self.radio_normalize_over_stack.setEnabled(True)
            self.radio_normalize_by_image.setEnabled(True)
            self.button_group_normalize.button(norm_scope).setChecked(True)
        # Case `data_dims is None` and other invalid cases are already caught above, so we don't need an else statement here

    def _reset_predict_buttons(self):
        """Enable or disable predict buttons based on the current state."""
        # An image layer needs to be selected
        if self.image_layer_selection_widget.value is not None:
            img = self._get_selected_img()
            data_dims = self._get_data_dims(img.data, img.ndim) if img is not None else None
            if data_dims not in self.supported_data_dims:
                warnings.warn(f'Non-supported image dimensions {data_dims}. Predict buttons disabled.')
                self.segment_btn.setEnabled(False)
                self.segment_all_btn.setEnabled(False)
                return
            is_stacked = data_dims in ['4D', '3D_single', '3D_RGB']
            # We need a trained model to enable segmentation
            if self.trained:
                self.segment_btn.setEnabled(True)
                self.segment_all_btn.setEnabled(is_stacked)
            else:
                self.segment_btn.setEnabled(False)
                self.segment_all_btn.setEnabled(False)
            # ... but not for getting features
            self.btn_add_features_stack.setEnabled(is_stacked)
        else: # No image selected
            self.segment_btn.setEnabled(False)
            self.segment_all_btn.setEnabled(False)
            self.btn_add_features_stack.setEnabled(False)

    def _on_change_device(self, event=None):
        """Update FE/CLF device policies when the dropdown selection changes."""
        selected_device = self.device_dropdown.currentText()
        # Pass selected policy through; runtime fallback/warnings are handled by ConvpaintModel.
        self.fe_device = "gpu" if "gpu" in selected_device else (
                         "auto" if "auto" in selected_device else
                         "cpu")
        self.clf_device = self.fe_device # Currently, the FE and CLF device policies are always the same, so we use the FE dropdown as proxy for both.

    def _reset_device_options(self):
        """Reset device dropdown availability and synchronize FE/CLF device policies."""
        if not hasattr(self, "device_dropdown"):
            return

        default_tooltip = 'Select device policy for feature extraction and classifier.'
        no_gpu_tooltip = 'No CUDA/MPS backend available. Device is fixed to CPU.'
        cuda_both_tooltip = 'CUDA is available and supported by this feature extractor. GPU can be used for both feature extraction and classifier.'
        cuda_clf_only_tooltip = 'CUDA is available, but this feature extractor does not support CUDA. GPU can still be used for the classifier; feature extraction will run on CPU.'
        mps_fe_only_tooltip = 'MPS is available and supported by this feature extractor, but CatBoost does not support MPS. Classifier will use CPU in all cases.'
        mps_none_tooltip = 'MPS is available, but neither this feature extractor nor CatBoost supports MPS. Device is fixed to CPU.'

        import torch
        mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        cuda_available = torch.cuda.is_available()

        fe_model = getattr(self.cp_model, "fe_model", None)
        fe_supported_devices = fe_model.supported_devices()
        supported_types = {device.type for device in fe_supported_devices if isinstance(device, torch.device)}
        fe_supports_cuda = "cuda" in supported_types
        fe_supports_mps = "mps" in supported_types
        gpu_available = cuda_available or (mps_available and fe_supports_mps)

        self.device_dropdown.blockSignals(True)
        # First, adjust the available options in the dropdown based on FE support and GPU availability
        self.device_dropdown.clear()
        new_options = self.device_options_gpu_only_clf if cuda_available and not fe_supports_cuda else self.device_options_default
        self.device_dropdown.addItems(new_options)
        # Then, adjust the selected option and tooltip based on the FE support and GPU availability
        try:
            if not gpu_available:
                self.device_dropdown.setEnabled(False)
                downgrade_from_gpu = self.clf_device == 'gpu'
                self.fe_device = 'cpu'
                self.clf_device = 'cpu'
                self.device_dropdown.setCurrentText('cpu')
                if mps_available and not cuda_available: # FE supports only cuda or neither, but only MPS is available
                    self.device_dropdown.setToolTip(mps_none_tooltip)
                    if downgrade_from_gpu: # Warn actively in case the policy was 'gpu' and downgrading to CPU was necessary
                        show_info(mps_none_tooltip)
                else: # No GPU available (this should usually only happen at startup, not when changing FE)
                    self.device_dropdown.setToolTip(no_gpu_tooltip)
                    if downgrade_from_gpu: # Warn actively in case the policy was 'gpu' and downgrading to CPU was necessary
                        show_info(no_gpu_tooltip)
            else: # Some combination of availability and FE support that allows GPU usage
                self.device_dropdown.setEnabled(True)
                # Change the currently selected policy
                if cuda_available and not fe_supports_cuda:
                    gpu_option = [opt for opt in self.device_options_gpu_only_clf if 'gpu' in opt][0]
                    selected_policy = gpu_option if self.clf_device == 'gpu' else (
                                      self.clf_device if self.clf_device in self.device_options_default else
                                      'auto')
                else:
                    selected_policy = self.clf_device if self.clf_device in self.device_options_default else 'auto'
                self.device_dropdown.setCurrentText(selected_policy)
                # Change the tooltip
                if cuda_available and fe_supports_cuda:
                    self.device_dropdown.setToolTip(cuda_both_tooltip)
                elif cuda_available and not fe_supports_cuda:
                    self.device_dropdown.setToolTip(cuda_clf_only_tooltip)
                elif mps_available and fe_supports_mps:
                    self.device_dropdown.setToolTip(mps_fe_only_tooltip)
                else: # This case should not happen, but we catch it just in case
                    self.device_dropdown.setToolTip(default_tooltip)
        finally:
            self.device_dropdown.blockSignals(False)

        # Re-apply policy through the dropdown handler.
        self._on_change_device()

    def _update_gui_fe_layer_keys(self, all_fe_layer_keys=None):
        """Update GUI FE layer list and selected layers based on the input."""
        all_fe_layer_texts = self._layer_keys_to_texts(all_fe_layer_keys)
        self.set_fe_btn.setEnabled(True)
        # Adjust the list of selectable layers based on the input (e.g. from the temp model)
        if all_fe_layer_texts is not None:
            # Add layer index to layer names
            self.fe_layer_selection.clear()
            self.fe_layer_selection.addItems(all_fe_layer_texts)
            self.fe_layer_selection.setEnabled(len(all_fe_layer_texts) > 1) # Only need to enable if multiple layers available
        # For non-hookmodels, disable the selection
        else:
            self.fe_layer_selection.clear()
            self.fe_layer_selection.setEnabled(False)

    def _update_gui_fe_layers(self, layers=None):
        """Update GUI FE layer list and selected layers based on the input."""
        # Select the layers as given
        # layer_texts = self._layer_keys_to_texts(layers)
        # if layer_texts is not None:
            # for layer in layer_texts:
        if layers is None:
            return
        self.fe_layer_selection.clearSelection()
        for layer in layers:
            # items = self.fe_layer_selection.findItems(layer, Qt.MatchExactly)
            items = self.fe_layer_selection.findItems(layer, Qt.MatchContains) # In case the layer names in the GUI have indices added, we use contains instead of exactly
            if not items:
                warnings.warn(f'Tried to set the layer "{layer}", but it was not in the list of available layers. ' +
                                'This might indicate a problem with the feature extractor selection.')
            for item in items:
                item.setSelected(True)

    def _update_gui_fe_proposed_scalings(self, all_fe_scalings=None):
        """Update GUI FE scalings list and selection based on input."""
        # Adjust the list of selectable scalings based on the input (e.g. from the temp model)
        if all_fe_scalings is not None:
            # Add layer index to layer names
            self.fe_scaling_factors.clear()
            for s in all_fe_scalings:
                self.fe_scaling_factors.addItem(f'[{",".join(map(str, s))}]', s)

    def _update_gui_fe_scalings(self, scalings=None):
        """Update GUI FE scalings selection based on input."""
        # Select the scalings as given
        if scalings is not None:
            # Find index of the input value (-1 if not found)
            index = self.fe_scaling_factors.findData(scalings)
            if index != -1:
                self.fe_scaling_factors.setCurrentIndex(index)
            # If not in list yet, add and select it
            else:
                self.fe_scaling_factors.addItem(str(scalings), scalings)
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

        self.qcombo_fe_type.setCurrentText(params.fe_name) # It's important that this is the first FE param to be set, as it sets the other to default values
        val_to_setter = {
            "image_downsample": self.spin_downsample.setValue,
            "seg_smoothening": self.spin_smoothen.setValue,
            "tile_annotations": self.check_tile_annotations.setChecked,
            "tile_image": self.check_tile_image.setChecked,
            "fe_order": self.spin_interpolation_order.setValue,
            "fe_use_min_features": self.check_use_min_features.setChecked,
            "clf_iterations": self.spin_iterations.setValue,
            "clf_learning_rate": self.spin_learning_rate.setValue,
            "clf_depth": self.spin_depth.setValue
        }
        for attr, setter in val_to_setter.items():
            val = getattr(params, attr, None)
            if val is not None:
                if isinstance(val, list): val = str(val)
                setter(val)

        # self._update_gui_fe_scalings(scalings=params.fe_scalings)
        # self._update_gui_fe_layers(layers=params.fe_layers)
    
    def _get_selected_img(self, check=False):
        """Get the image layer currently selected in Convpaint."""
        img = self.image_layer_selection_widget.value
        if img is None and check:
            warnings.warn('No image layer selected')
        return img

    def _get_annot_shape(self, img):
        """Get shape of annotations and segmentation layers to create."""
        data_dims = self._get_data_dims(img.data, img.ndim) if img is not None else None
        img_shape = img.data.shape
        if data_dims in ['2D_RGB', '2D']:
            return img_shape[0:2]
        elif data_dims in ['3D_RGB', '3D_single']:
            return img_shape[0:3]
        elif data_dims in ['3D_multi', '4D']: # channels first
            return img_shape[1:]
        else:
            warnings.warn(f'Unsupported data dimensions {data_dims}. annotations and segmentation layers might not be created with the correct shape.')
            return img_shape

    def _check_large_image(self, img):
        """Check if the image is very large and should be tiled."""
        data_dims = self._get_data_dims(img.data, img.ndim) if img is not None else None
        img_shape = img.data.shape
        if data_dims in ['2D', '2D_RGB']:
            xy_plane = img_shape[0] * img_shape[1]
        elif data_dims in ['3D_single', '3D_RGB', '3D_multi']:
            xy_plane = img_shape[1] * img_shape[2]
        elif data_dims == '4D':
            xy_plane = img_shape[2] * img_shape[3]
        else:
            warnings.warn(f'Unsupported data dimensions {data_dims}. Image size is not evaluated for tiling.')
            xy_plane = 0
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
        self.spin_iterations.setValue(self.default_cp_param.clf_iterations)
        self.spin_learning_rate.setValue(self.default_cp_param.clf_learning_rate)
        self.spin_depth.setValue(self.default_cp_param.clf_depth)
        # Mimic pressing "Set classifier parameters".
        self._on_set_clf_params()

    def _reset_fe_params(self):
        """Reset feature extraction parameters to default values."""
        # Reset the gui values for the FE, which will also trigger to adjust the param object, unless the text (model) does not change)
        fe_text_before = self.qcombo_fe_type.currentText()
        self.qcombo_fe_type.setCurrentText(self.default_cp_param.fe_name)
        if self.qcombo_fe_type.currentText() == fe_text_before:
            self.qcombo_fe_type.currentIndexChanged.emit(0) # Trigger the signal to update the FE parameters even if the text is the same
        # self._on_fe_selected() # To update the proposed layers and scalings based on the default FE; NOT NECESSARY, as changing the text triggers _on_fe_selected
        # self.fe_layer_selection.clearSelection()
        # default_layers = self._layer_keys_to_texts(self.default_cp_param.fe_layers)
        # for layer in default_layers:
        #     items = self.fe_layer_selection.findItems(layer, Qt.MatchExactly)
        #     for item in items:
        #         item.setSelected(True)
        # self.fe_scaling_factors.setCurrentText(str(self.default_cp_param.fe_scalings))
        # self.spin_interpolation_order.setValue(self.default_cp_param.fe_order)
        # self.check_use_min_features.setChecked(self.default_cp_param.fe_use_min_features)
        # self.check_use_gpu.setChecked(self.default_cp_param.fe_use_gpu)
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
        # Get general model attributes
        if self.cp_model.fe_model is None:
            descr = 'No model set'
            return
        name, layers, scalings = (getattr(self.cp_model.get_params(), x)
                                  for x in ['fe_name', 'fe_layers', 'fe_scalings'])
        fe_name = name if name is not None else 'None'
        num_layers = len(layers) if layers is not None else 0
        num_scalings = len(scalings) if scalings is not None else 0
        # Get device support information for the FE model and system
        supported_devices = self.cp_model.fe_model.supported_devices() if hasattr(self.cp_model.fe_model, 'supported_devices') else []
        supported = [str(d) for d in supported_devices]
        # gpu_device = get_fe_device(use_device="gpu", supported_devices=supported_devices, warn=False) # See if there's a gpu option
        device_string = 'supports: cuda, cpu' if any('cuda' in d for d in supported) else(
                        'supports: mps, cpu' if any('mps' in d for d in supported) else
                        'uses cpu only')
        # Put together and post
        descr = (fe_name +
        f': {num_layers} layer' + ('' if num_layers == 1 else 's') +
        f', {num_scalings} scaling' + ('' if num_scalings == 1 else 's') + 
        f' ({self.current_model_path})' +
        f' | {device_string}'
        )
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
    
    def _get_selected_scaling_factors(self):
        """Get the selected scaling factors for the FE."""
        scaling_text = self.fe_scaling_factors.currentText()
        if "shark" in scaling_text.lower():
            initial_auto_flag = self.auto_add_layers
            self.auto_add_layers = False # Prevent adding new layers for the shark video
            self._show_shark_easter_egg()
            QTimer.singleShot(500, lambda: # Wait a bit before resetting the flag, to make sure the video is loaded before
                              setattr(self, 'auto_add_layers', initial_auto_flag)
                              )
            return None
        # Try to convert the text to a tuple of ints (e.g. "[1,2,3]" -> (1,2,3))
        try:
            scalings = eval(scaling_text)
            if isinstance(scalings, int):
                scalings = [scalings]
            elif isinstance(scalings, tuple):
                scalings = list(scalings)
            elif not isinstance(scalings, list):
                raise ValueError
        except (ValueError, SyntaxError):
            warnings.warn(f"Could not parse scaling factors from text '{scaling_text}'")
            return None
        return scalings
    
    def _show_shark_easter_egg(self):
        """Show a shark in the viewer as an easter egg."""
        shark_data_path = "./images/segmentation_shark.mp4"
        shark_layer_name = "Convpaint Shark"
        self.viewer.open(shark_data_path, name=shark_layer_name)
        for l in self.viewer.layers:
            if not l.name == shark_layer_name:
                l.visible = False
        self.viewer.reset_view()
        from napari.settings import get_settings
        settings = get_settings()
        settings.application.playback_fps = 100
        settings.application.playback_mode = "back_and_forth"
        try:
            with warnings.catch_warnings(action="ignore"): # Suppress deprecation warning about _qt_viewer
                self.viewer.window._qt_viewer.dims.play()
        except Exception as e:
            warnings.warn(f"Could not play shark video. Play it manually if you want :)")

    def _get_data_channel_first(self, data, num_dims):
        """Get data from selected channel. If RGB/RGBA, move channel axis to first
        position and strip alpha channel if present (keep only first 3 channels)."""
        if data is None or num_dims is None:
            return None
        data_dims = self._get_data_dims(data, num_dims)
        if data_dims in ['2D_RGB', '3D_RGB']:
            data = data[..., :3]  # Strip alpha channel if RGBA
            return np.moveaxis(data, -1, 0)
        elif data_dims in ['2D', '3D_single', '3D_multi', '4D']:
            return data
        else:
            warnings.warn(f'Unsupported data dimensions {data_dims}. Data is returned without moving channel axis to first position if needed.')
            return data

    def _get_data_channel_first_norm(self, img):
        """Get data from selected channel. Output has channel (if present) in 
        first position and is normalized."""

        # Get data from selected layer, with channels first if present
        # 2D, 3D or 4D array, with C first if present
        image_stack = self._get_data_channel_first(img.data, img.ndim) if img is not None else None

        # Import heavy image utilities lazily to avoid import-time cost
        from .utils import normalize_image, normalize_image_percentile, normalize_image_imagenet

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
                data_dims = self._get_data_dims(img.data, img.ndim) if img is not None else None
                if data_dims in ["4D", "3D_multi", "2D_RGB", "3D_RGB"]: # Channels dimension present
                    num_ignored_dims = 1 # ignore channels dimension, but norm over stack if present
                elif data_dims in ["2D", "3D_single"]: # No channels dimension present
                    num_ignored_dims = 0 # norm over entire stack
                else:
                    warnings.warn(f'Unsupported data dimensions {data_dims}. Normalization over stack might not be applied correctly.')
                    num_ignored_dims = 0
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
        from .utils import normalize_image, normalize_image_imagenet
        
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
        data_dims = self._get_data_dims(img.data, img.ndim) if img is not None else None

        # If we use default normalization, compute image stats if not already done
        if use_default and (self.image_mean is None or self.image_std is None):
            self._compute_image_stats(img)

        # Get image data and stats depending on the data dim and norm mode
        if fe_norm != "percentile":
            # For default and imagenet norm, we want unnormalized data to apply normalization only on the current plane
            img = self._get_data_channel_first(img.data, img.ndim) if img is not None else None
        else: # "percentile"
            # For percentile norm, we want already normalized data to avoid artifacts when normalizing only the current plane
            img = self._get_data_channel_first_norm(img)

        if data_dims in ['2D', '2D_RGB', '3D_multi'] or data_dims not in self.supported_data_dims:
            # No stack dim, so just take the image as is; use this also in case of invalid data_dims
            if data_dims not in self.supported_data_dims:
                warnings.warn(f'Unsupported data dimensions {data_dims}. Current plane is not selected correctly.')
            # Use img as is
            image_plane = img
            # Get stats for default norm if needed
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

        elif data_dims in ['4D', '3D_RGB']: # Stack dim is third last
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

        # Import heavy utility here to avoid importing at module import time
        from .utils import compute_image_stats

        # If no image is selected or image has unsupported dimensions, set stats to None
        if img is None:
            self.image_mean, self.image_std = None, None
            return

        # Assure to have channels dimension first, get the data_dims and normalization mode
        data_dims = self._get_data_dims(img.data, img.ndim) if img is not None else None

        # If image has unsupported dimensions, set stats to None and warn
        if data_dims not in self.supported_data_dims:
            warnings.warn(f'Unsupported image dimensions {data_dims}. Image stats for normalization are not computed.')
            self.image_mean, self.image_std = None, None
            return

        # Compute image stats depending on the normalization mode

        data = self._get_data_channel_first(img.data, img.ndim) if img is not None else None
        norm_scope = self.cp_model.get_param("normalize")
        if norm_scope == 2: # normalize over stack --> only keep channels dimension
            if data_dims in ["4D", "3D_multi", "2D_RGB", "3D_RGB"]: # Channels dimension present
                # 3D multi/2D_RGB or 4D/3D_RGB --> (C,1,1) or (C,1,1,1)
                self.image_mean, self.image_std = compute_image_stats(
                    image=data,
                    ignore_n_first_dims=1) # ignore channels dimension, but norm over stack if present
            elif data_dims in ["2D", "3D_single"]: # No channels dimension present
                # 2D or 3D_single --> (1,1) or (1,1,1)
                self.image_mean, self.image_std = compute_image_stats(
                    image=data,
                    ignore_n_first_dims=None) # norm over entire stack
            # Case `data_dims is None` and other invalid cases are already caught above, so we don't need an else statement here

        elif norm_scope == 3: # normalize by image --> keep channels and plane dimensions
            # also takes into account CXY case (3D multi-channel image) where first dim is dropped, and 2D where none is
            c_z_channels = data.ndim-2 # number of channels and/or Z dimension (i.e. "non-spatial" dimensions)
            self.image_mean, self.image_std = compute_image_stats(
                image=data, ignore_n_first_dims=c_z_channels) # ignore all but the plane/spatial dimensions
            # --> 2D (1,1) or 3D (single, multi -> Z/C,1,1) or 2D_RGB (C,1,1) or 4D/3D_RGB (C,Z,1,1)

    def _get_data_dims(self, data, num_dims):
        """Get data dimensionality. Also perform checks on the data dimensions.
        Returns '2D', '2D_RGB', '3D_RGB', '3D_multi', '3D_single' or '4D'."""

        if data is None or num_dims is None:
            return None
        # Sanity check for number of dimensions
        if (num_dims == 1 or num_dims > 4):
            warnings.warn(f'Image has {num_dims} dimensions, but only 2D-4D images are supported.')
            return None
        
        # 2D can be either single channel or RGB/RGBA (in which case the underlying data is in fact 3D with 3-4 channels as last dim)
        if num_dims == 2:
            if self.cp_model.get_param("channel_mode") == "rgb":
                if data.shape[-1] in (3, 4):
                    return '2D_RGB'
                else:
                    warnings.warn('Image is 2D, but does not have 3 or 4 channels as last dimension. Setting channel_mode to "single".')
                    self.cp_model.set_param("channel_mode", "single", ignore_warnings=True)
                    return '2D'
            elif self.cp_model.get_param("channel_mode") == "single":
                return '2D'
            else: # multi channel not possible in 2D
                warnings.warn('Image is 2D, but channel_mode was "multi". Setting channel_mode to "single".')
                self.cp_model.set_param("channel_mode", "single", ignore_warnings=True)
                return '2D'

        # 3D can be single channel, multi channel or RGB/RGBA (in which case the underlying data is in fact 4D with 3-4 channels as last dim)
        if num_dims == 3:
            if self.cp_model.get_param("channel_mode") == "rgb":
                if data.shape[-1] in (3, 4):
                    return '3D_RGB'
                else:
                    warnings.warn('Image is 3D, but does not have 3 or 4 channels as last dimension. Setting channel_mode to "multi".')
                    self.cp_model.set_param("channel_mode", "multi", ignore_warnings=True)
                    return '3D_multi'
            if self.cp_model.get_param("channel_mode") == "multi":
                return '3D_multi'
            else: # single
                return '3D_single'

        # 4D can only be multi channel
        if num_dims == 4:
            if self.cp_model.get_param("channel_mode") == "single":
                warnings.warn('Image has 4 dimensions, but channel_mode was "single". Setting channel_mode to "multi".')
                self.cp_model.set_param("channel_mode", "multi", ignore_warnings=True)
            return '4D'
        
    def _approve_annotations_layer_shape(self, annot, img):
        """Check if the annotations layer has the same shape as the image layer."""
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

    def _update_annotations_layers(self):
        """"Update the annotations layers in the viewer sorted by their names."""
        # Update the choices in the annotations layer selection widget
        self.annotations_layer_selection_widget.reset_choices()

        # Sort them
        # annot_layer_list = [layer for layer in self.viewer.layers
        #                     if isinstance(layer, napari.layers.Labels)]
        # annot_layer_list = [layer for layer in self.annotations_layer_selection_widget.choices
        #                     if isinstance(layer, napari.layers.Labels)]
        # annot_layer_list.sort(key=lambda x: x.name)
        # self.annotations_layer_selection_widget.choices = [] # reset; seems necessary to allow sorting
        # self.annotations_layer_selection_widget.choices = [(layer.name, layer) for layer in annot_layer_list]
        # return annot_layer_list

    def _train_multiple(self, img_list, annot_list, id_list):
        """Core training routine used by multiple callers.

        id_list: list of str image ids/names
        img_list: list of numpy arrays (prepared via _get_data_channel_first)
        annot_list: list of numpy arrays (annotations masks)
        """
        if self.cp_model is None:
            warnings.warn('No model set. Cannot train.')
            return

        if not id_list or not img_list or not annot_list:
            warnings.warn('No images or annotations provided for training.')
            return

        if not (len(id_list) == len(img_list) == len(annot_list)):
            warnings.warn('Image and annotations lists must have identical lengths.')
            return

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(True)

        with progress(total=0) as pbr:
            pbr.set_description("Training")
            mem_mode = (self.cont_training == "Image"
                        or self.cont_training == "Global")
            # Train; in this case, normalization is not skipped (but done in the ConvpaintModel)
            in_channels = self._parse_in_channels(self.input_channels)
            _ = self.cp_model.train(img_list, annot_list, memory_mode=mem_mode, img_ids=id_list,
                                    in_channels=in_channels, skip_norm=False,
                                    fe_use_device=self.fe_device, clf_use_device=self.clf_device)
            self._update_training_counts()

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(False)

        # Set the current model path to 'trained, unsaved' and adjust the model description
        self.current_model_path = 'trained, unsaved'
        self.trained = True
        self._reset_predict_buttons()
        self._set_model_description()

### ADVANCED TAB

    def _on_add_all_annot_layers(self):
        """Add annotations layers for all image layers selected in the layers widget (napari).
        Has been deprecated in favor of the "Multifile" Tab."""

        # Get the selected image layers in the order they are in the widget
        img_layer_list = [layer for layer in self.viewer.layers
                          if layer in self.viewer.layers.selection]
        chosen_in_dropdwon = self._get_selected_img()
        # If the layer chosen in the dropdown is selected, move it first, so its annotations is added first
        if chosen_in_dropdwon is not None and chosen_in_dropdwon.name in img_layer_list:
            img_layer_list.remove(chosen_in_dropdwon)
            img_layer_list.insert(0, chosen_in_dropdwon)

        for img in img_layer_list:
            layer_shape = self._get_annot_shape(img)
            channel_mode_str = (self.radio_rgb.isChecked()*"RGB" +
                    self.radio_multi_channel.isChecked()*"multiCh" +
                    self.radio_single_channel.isChecked()*"singleCh")
            layer_name = f'{self.annot_tag}_{img.name}_{channel_mode_str}'
            layer_name = self._get_unique_layer_name(layer_name)
            data = np.zeros((layer_shape), dtype=np.uint8)
            # Create a new annotations layer with the unique name (copy image transforms)
            num_spatial = len(layer_shape)
            kwargs = self._get_layer_transform_kwargs(img, num_spatial_dims=num_spatial, num_leading_dims=0)
            self.viewer.add_labels(data=data, name=layer_name, **kwargs)
            labels_layer = self.viewer.layers[layer_name]
            # Set the annotations layer to paint mode
            labels_layer.mode = 'paint'
            labels_layer.brush_size = self.default_brush_size
            # Connect colormap events in new layers
            labels_layer.events.colormap.connect(self._on_change_annot_cmap)
            # Add it to the list of layers where class names shall be updated
            self.annot_layers.add(labels_layer)
        
        # sync class names and cmaps
        self.update_all_class_names_and_cmaps()

    def _on_train_on_selected(self):
        """Train the model on the image and annotations layers currently selected in the layers widget (napari).
        Has been deprecated in favor of the "Multifile" Tab."""

        # Get selected layers (arbitrary order) and sort them by their names
        layer_list = list(self.viewer.layers.selection)
        layer_list.sort(key=lambda x: x.name)
        
        # Get the image and annotations layers based on the prefix
        prefix_len = len(self.annot_tag)
        img_list = [layer for layer in layer_list
                    if not layer.name[:prefix_len] == self.annot_tag
                    and isinstance(layer, napari.layers.Image)]
        annot_list = [layer for layer in layer_list
                      if layer.name[:prefix_len] == self.annot_tag
                      and isinstance(layer, napari.layers.Labels)]

        # NOTE: Checks are technically not necessary here, as it is done in the CPModel
        if len(annot_list) == 0 or len(img_list) == 0 or len(annot_list) != len(img_list):
            warnings.warn('Please select images and corresponding annotations layers')
            return
        
        # Create lists of images and annotations
        arr_list = [self._get_data_channel_first(img.data, img.ndim) if img is not None else None
                    for img in img_list]
        annot_list = [annot.data for annot in annot_list]
        id_list = [img.name for img in img_list]

        # Delegate core training to helper that can be reused
        self._train_multiple(arr_list, annot_list, id_list)

    def _update_training_counts(self):
        """Update the training counts (used with continuous_training/memory_mode) in the GUI."""
        if self.cp_model is None:
            return
        pix = len(self.cp_model.table)
        imgs = len(np.unique(self.cp_model.table['img_id']))
        lbls = len(np.unique(self.cp_model.table['label']))
        self.label_training_count.setText(f'{pix} pixels, {imgs} image{"s"*(imgs>1)}, {lbls} labels')

    def _on_show_class_distribution(self, trained_data=False):
        """Show the class distribution of the data used with continuous_training/memory_mode (saved in self.cp_model.table)
        in a pie chart using the according cmaps.
        
        trained_data: If True, show the distribution of the data in the training table (i.e. used for training).
                      If False, show the distribution of the data in the currently selected annotations layer.
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
            if self.annotations_layer_selection_widget.value is None:
                warnings.warn('No annotations layer selected. Cannot show class distribution.')
                return
            # Otherwise get the labels from the annotations layer selected in the layers widget (excluding unlabeled pixels)
            labels = self.annotations_layer_selection_widget.value.data.flatten()
            labels = labels[labels != 0]
        if len(labels) == 0:
            warnings.warn('No labels available. Cannot show class distribution.')
            return
        classes = np.unique(labels)
        counts = np.array([np.sum(labels == c) for c in classes])
        percs = counts / np.sum(counts) * 100

        # Get class display names from a list, assuming class numbers start at 1
        if self.class_names is not None and self.class_names:
            class_names = [self.class_names[c - 1].text() if 1 <= c <= len(self.class_names) else str(c) for c in classes]

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

    def _reset_train_features(self):
        """Reset the training features used with continuous_training/memory_mode."""
        self.cp_model.reset_training()
        self._update_training_counts()
        # Save image_layer names and their corresponding annotations layers, to allow only extracting new features
        self.features_annots = {}

    def _on_switch_axes(self):
        """Switch the first two axes of the input image."""
        in_img = self._get_selected_img(check=True)
        data_dims = self._get_data_dims(in_img.data, in_img.ndim) if in_img is not None else None
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
        # Get the coordinates of the non-zero elements in the annotations array
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
        

### MULTIFILE TAB

    # Connect folder selection: open dialog and populate file list
    def _select_multifile_img_folder(self):

        # If there are existing layers in the viewer, warn the user once and allow abort.
        existing_layers = list(self.viewer.layers)
        if self._multifile_annotations_store:
            unsaved = [fname for fname, stored in self._multifile_annotations_store.items()
                    if not isinstance(stored, str)]
        else:
            unsaved = []
        if not getattr(self, '_multifile_warned', False):
            msg = ""
            if existing_layers:
                # Warn the user about removing existing layers
                msg += f'The viewer currently contains {len(existing_layers)} layer(s) that will be removed if you reset the folder.\n'
            if unsaved:
                msg += (f'The following files have unsaved annotations that will be lost if you reset the folder:\n' +
                       '\n'.join(unsaved) + '\n')
            if existing_layers or unsaved:
                msg += "\nDo you want to continue?"
                resp = QMessageBox.question(self, 'Discard annots in memory and/or existing layers?', msg, QMessageBox.Yes | QMessageBox.No)
                if resp != QMessageBox.Yes:
                    return
                self._multifile_warned = True
            
        # User accepted; set warned flag and remove layers
        for l in existing_layers:
            try:
                self.viewer.layers.remove(l)
            except Exception:
                pass

        default_dir = str(Path(self._multifile_last_folder)) if getattr(self, '_multifile_last_folder', None) else str(Path.cwd())
        folder = QFileDialog.getExistingDirectory(self, 'Select image folder', default_dir)
        if not folder:
            return

        # Remember last used images folder for future dialogs
        self._multifile_last_folder = str(Path(folder))

        # Initialize annotations/segmentations when opening a new folder (resetting in-memory annotations)
        self._multifile_annotations_store = {}
        self._multifile_segmentation_store = {}
        # Set flag that shall update multifile annotations tick
        self._multifile_update_annot_tick = True
        self._multifile_last_annot = None
        # Reset import warning flag so user is asked again when importing
        # self.multifile_import_warned = False

        self.multifile_path_edit.setText(folder)
        p = Path(folder)
        try:
            files = sorted([f for f in p.iterdir() if f.is_file()])
        except Exception:
            files = []
        self.multifile_list.setRowCount(0)
        for f in files:
            fname = f.name
            stem = f.stem
            # Filter out annotations and segmentations
            stem_suffix = stem.split('_')[-1].lower() if '_' in stem else ''
            # Note: here we use a wider selection of suffixes to ignore...
            if stem_suffix and (stem_suffix in self.multifile_annot_suffixes or
                                stem_suffix in self.multifile_seg_suffixes):
                continue
            if fname[0] == '.': # ignore hidden files
                continue
            # Add file to the list/table
            row = self.multifile_list.rowCount()
            self.multifile_list.insertRow(row)
            # Annotations column: red X for initial state (no annotations loaded)
            item_annot = QTableWidgetItem('✗')
            item_annot.setTextAlignment(Qt.AlignCenter)
            item_annot.setForeground(QtGui.QBrush(QtGui.QColor('red')))
            # Segmentations column: red X for initial state (no segmentations loaded)
            item_seg = QTableWidgetItem('✗')
            item_seg.setTextAlignment(Qt.AlignCenter)
            item_seg.setForeground(QtGui.QBrush(QtGui.QColor('red')))
            # make annotated and segmented flags non-editable
            item_annot.setFlags(item_annot.flags() & ~Qt.ItemIsEditable)
            item_seg.setFlags(item_seg.flags() & ~Qt.ItemIsEditable)
            item_filename = QTableWidgetItem(fname)
            # make filename non-editable
            item_filename.setFlags(item_filename.flags() & ~Qt.ItemIsEditable)
            self.multifile_list.setItem(row, 0, item_annot)
            self.multifile_list.setItem(row, 1, item_filename)
            self.multifile_list.setItem(row, 2, item_seg)

        # Automatically open the first image in the list (if any)
        if files:
            try:
                self._on_multifile_open_file(0, 1)
            except Exception:
                pass

    def _multifile_get_selected(self):
        """Get row selected in the image list."""
        try:
            sel = self.multifile_list.selectionModel().selectedRows()
            if not sel:
                warnings.warn('No files selected for segmentation.')
                return
            rows = [s.row() for s in sel]
        except Exception:
            warnings.warn('Could not determine selected files.')
            return None
        return rows

    def _multifile_clear_annot(self):
        """Clear the annotations for the selected multifile images, in the in-memory store.
        If one of the images is currently open in the viewer, also clear the annotations layer."""

        rows = self._multifile_get_selected()
        if rows is None:
            return

        for row in rows:
            fname = self.multifile_list.item(row, 1).text()
            img_opened = fname == getattr(self, '_current_multifile_filename', None)
            # Clear annotations layer data
            annot_opened = self.annot_tag in self.viewer.layers
            if img_opened and annot_opened:
                self.viewer.layers[self.annot_tag].data = np.zeros_like(self.viewer.layers[self.annot_tag].data)
            # Clear in-memory annotations store for current file
            if fname and fname in getattr(self, '_multifile_annotations_store', {}):
                del self._multifile_annotations_store[fname]
            # Update annotations status tick in the file list
            self._update_multifile_annot_tick(fname)

    def _multifile_clear_seg(self):
        """Clear the segmentation for the selected multifile images, in the in-memory store.
        If one of the images is currently open in the viewer, also remove the segmentation layer."""
        rows = self._multifile_get_selected()
        if rows is None:
            return
        
        for row in rows:
            fname = self.multifile_list.item(row, 1).text()
            img_opened = fname == getattr(self, '_current_multifile_filename', None)
            # Remove segmentation layer
            seg_opened = self.seg_tag in self.viewer.layers
            if img_opened and seg_opened:
                self.viewer.layers.remove(self.viewer.layers[self.seg_tag])
            # Clear in-memory segmentation store for current file
            if fname and fname in getattr(self, '_multifile_segmentation_store', {}):
                del self._multifile_segmentation_store[fname]
            # Update segmentation status tick in the file list
            self._update_multifile_seg_tick(fname)

    def _reset_multifile_folder(self):
        """Reset the multifile folder selection and clear the file list."""
        # Check for unsaved annots (those that are saved as arrays, not strings, and have yellow tick), and warn if any
        if self._multifile_annotations_store:
            unsaved = [fname for fname, stored in self._multifile_annotations_store.items()
                       if not isinstance(stored, str)]
            if unsaved:
                msg = (f'The following files have unsaved annotations that will be lost if you reset the folder:\n' +
                       '\n'.join(unsaved) + '\n' +
                       '\nDo you want to continue?')
                resp = QMessageBox.question(self, 'Unsaved annotations', msg, QMessageBox.Yes | QMessageBox.No)
                if resp != QMessageBox.Yes:
                    return

        self.multifile_path_edit.setText('')
        self.multifile_list.setRowCount(0)
        # Clear in-memory annotations store and imported tracking
        self._multifile_warned = False
        self._multifile_annotations_store = {}
        self._multifile_segmentation_store = {}
        # self.multifile_import_warned = False
        # Reset flag to update annotations tick
        self._multifile_update_annot_tick = True
        self._multifile_last_annot = None

    def _on_multifile_open_file(self, row, column):
        """Open the file at the given row in the viewer without making it the Convpaint 'image layer'.

        If there are existing layers, warn the user they will be removed and allow abort.
        After loading, reset Convpaint state to avoid side-effects.
        """
        # get filename from list/table
        try:
            item = self.multifile_list.item(row, 1)
            if item is None:
                return
            filename = item.text()
            folder = self.multifile_path_edit.text()
            if not folder:
                return
            path = Path(folder) / filename
        except Exception:
            return

        # Temporarily disable auto-adding annotations layers (we do it manually below)
        original_auto = getattr(self, 'auto_add_layers', False) # Save old value to restore later
        setattr(self, 'auto_add_layers', False) 
    
        # If there are existing layers in the viewer, remove them
        existing_layers = list(self.viewer.layers)
        if existing_layers:
            try:
                for l in existing_layers:
                    try:
                        self.viewer.layers.remove(l)
                    except Exception:
                        pass
            finally:
                pass
        
        # load image data and add as a new layer
        try:
            self.viewer.open(path, name=path.name)
        except Exception as e:
            QMessageBox.critical(self, 'Failed to open', f'Could not open {path}: {e}')
        # Restore auto-adding annotations layers setting
        QTimer.singleShot(100, lambda: setattr(self, 'auto_add_layers', original_auto))

        # Track current filename opened via Multifile
        self._current_multifile_filename = filename
        self._current_multifile_row = row
        # Update bold state in file list to indicate opened file
        try:
            self._update_multifile_opened_bold(filename)
        except Exception:
            pass

        # Set flag to not update tick when opening an image, esp. not through adding from file/memory (annot status cannot change here)
        self._multifile_update_annot_tick = False

        # Add/open annotations and segmentations
        if self.multifile_import_open_annotations:
            QTimer.singleShot(250, lambda: self._multifile_open_annot(filename, auto_add=original_auto))
        if self.multifile_import_open_segmentations:
            QTimer.singleShot(250, lambda: self._multifile_open_segmentation(filename))

        # Reset flag to update annotations status on annotations changes (e.g. painting)
        QTimer.singleShot(250, lambda: setattr(self, '_multifile_update_annot_tick', True))
        QTimer.singleShot(250, lambda: setattr(self, '_multifile_last_annot', self.viewer.layers[self.annot_tag].data.copy()
                                                if self.annot_tag in self.viewer.layers else None))

    def _multifile_open_annot(self, filename, auto_add=True):
        # If we have a stored annotations for this filename, add its data to the labels layer
        if filename in getattr(self, '_multifile_annotations_store', {}):
            # Add a layer (will be filled with data)
            self._add_empty_annot(event=None, force_add=True, from_multifile=True)
            stored = self._multifile_annotations_store[filename]
            try:
                if isinstance(stored, str):
                    # Stored as path -> read on demand but keep store as path
                    arr = self.read_labels_file(stored)
                    if arr is not None:
                        self.viewer.layers[self.annot_tag].data = arr
                else:
                    # ndarray in memory
                    self.viewer.layers[self.annot_tag].data = stored.copy()
            except Exception:
                pass
        elif auto_add: # Add without data, but only if the option is not turned off (Advanced tab)
            self._add_empty_annot(event=None, force_add=True, from_multifile=True)

    def _multifile_open_segmentation(self, filename):
        # If we have a stored segmentation for this filename, add its data to the segmentation layer
        if filename in getattr(self, '_multifile_segmentation_store', {}):
            # Add a layer (will be filled with data)
            self._check_create_segmentation_layer()
            stored = self._multifile_segmentation_store[filename]
            try:
                if isinstance(stored, str):
                    # Stored as path -> read on demand but keep store as path
                    arr = self.read_labels_file(stored)
                    if arr is not None:
                        self.viewer.layers[self.seg_tag].data = arr
            except Exception:
                pass

    def _on_annotations_changed(self, event=None):
        """Save annotations for current multifile image if present."""
        # Case 1: layer removal event (viewer.layers.events.removed)
        if hasattr(event, "value") and hasattr(event.value, "name"):
            layer = event.value
        # Case 2: layer-level event (e.g. set_data)
        elif hasattr(event, "source") and hasattr(event.source, "name"):
            layer = event.source
        # Case 3: mouse release event after painting in annotations layer (checked outside...)
        else:
            layer = self.viewer.layers.selection.active
        # Check if the layer has been deducted correctly
        if layer is None:
            return

        # Check that we are handling the correct layer in the correct context (multifile mode, annotations layer)
        if not self.store_annot or not layer or layer.name != self.annot_tag:
            return
        # Determine filename for current multifile image
        fname = getattr(self, '_current_multifile_filename', None)
        if fname is None:
            return

        try:
            data = np.asarray(layer.data)
            QTimer.singleShot(250, lambda: self._maybe_update_annots(data, fname))
            QTimer.singleShot(500, lambda: setattr(self, '_multifile_last_annot', data.copy()))
        except Exception:
            pass

    def _maybe_update_annots(self, data, filename):
        """If the annotations data has actually changed add array to store and _update_multifile_annot_tick but only. Defined separately, so we can call it with a delay."""
        has_annot = np.sum(data > 0) != 0
        first_annot = self._multifile_last_annot is None # If it's the first file, or we don't have a last annotations to compare to
        new_img = self._multifile_last_annot.shape != data.shape if not first_annot else False # If the file changed (check in this order as otherwise we cannot compare...)
        annot_changed = not np.all(self._multifile_last_annot == data) if not first_annot and not new_img else False # If the annotations data changed
        # Store actual array data when user paints changes into annotations
        if annot_changed:
            if has_annot:
                self._multifile_annotations_store[filename] = data.copy()
            else:
                if hasattr(self, '_multifile_annotations_store') and filename in self._multifile_annotations_store:
                    del self._multifile_annotations_store[filename]
        # update table flag (type-based); this is in some cases also wanted when there is no change in annotations data (e.g. when opening a file ...)
        if has_annot and (first_annot or new_img or annot_changed):
            self._update_multifile_annot_tick(filename)

    def _update_multifile_annot_tick(self, filename):
        """Update the Annotations column in the multifile table for a given filename."""
        # Guard from changing flag when opening images or not making changes to annotations
        if not self._multifile_update_annot_tick:
            return
        # Find row and adjust tick...
        try:
            for r in range(self.multifile_list.rowCount()):
                it = self.multifile_list.item(r, 1)
                if it is not None and it.text() == filename:
                    item_annot = self.multifile_list.item(r, 0) # The annot cell of the selected image
                    if item_annot is None: # No cell found; should not happen...
                        item_annot = QTableWidgetItem()
                        item_annot.setFlags(item_annot.flags() & ~Qt.ItemIsEditable)
                        item_annot.setTextAlignment(Qt.AlignCenter)
                        self.multifile_list.setItem(r, 0, item_annot)
                    # Distinguish imported/exported (persistent) annotations from in-memory ones
                    store = getattr(self, '_multifile_annotations_store', {})
                    val = store.get(filename, None)
                    if isinstance(val, str):
                        # stored as path -> persistent (green)
                        item_annot.setText('✓')
                        item_annot.setForeground(QtGui.QBrush(QtGui.QColor('green')))
                    elif val is not None:
                        # in-memory ndarray -> yellow in brackets
                        item_annot.setText('(✓)')
                        item_annot.setForeground(QtGui.QBrush(QtGui.QColor('gold')))
                    else:
                        item_annot.setText('✗')
                        item_annot.setForeground(QtGui.QBrush(QtGui.QColor('red')))
                    return
        except Exception:
            pass

    def _update_multifile_seg_tick(self, filename):
        """Update the Segm. column in the multifile table for a given filename."""
        try:
            for r in range(self.multifile_list.rowCount()):
                it = self.multifile_list.item(r, 1)
                if it is not None and it.text() == filename:
                    item_seg = self.multifile_list.item(r, 2) # The seg cell of the selected image
                    if item_seg is None:
                        item_seg = QTableWidgetItem()
                        item_seg.setFlags(item_seg.flags() & ~Qt.ItemIsEditable)
                        item_seg.setTextAlignment(Qt.AlignCenter)
                        self.multifile_list.setItem(r, 2, item_seg)
                    store = getattr(self, '_multifile_segmentation_store', {})
                    val = store.get(filename, None)
                    if isinstance(val, str):
                        item_seg.setText('✓')
                        item_seg.setForeground(QtGui.QBrush(QtGui.QColor('green')))
                    else:
                        item_seg.setText('✗')
                        item_seg.setForeground(QtGui.QBrush(QtGui.QColor('red')))
                    return
        except Exception:
            pass

    def _update_multifile_opened_bold(self, filename):
        """Make the filename item bold for the currently opened multifile image."""
        try:
            for r in range(self.multifile_list.rowCount()):
                it = self.multifile_list.item(r, 1)
                if it is None:
                    continue
                item_font = it.font()
                if it.text() == filename:
                    item_font.setBold(True)
                else:
                    item_font.setBold(False)
                it.setFont(item_font)
        except Exception:
            pass

    def _on_train_on_multifile(self):
        """Train the model on all images that have annotations stored in memory (multifile store).

        Ensures any current open multifile annotations is pushed to the in-memory store
        before assembling lists and delegating to `_train_multiple`.
        """
        # Ensure current open annotations (if any, and if it has annotations) is saved to the store
        fname = getattr(self, '_current_multifile_filename', None)
        if fname is not None and self.annot_tag in self.viewer.layers:
            try:
                labels_layer = self.viewer.layers[self.annot_tag]
                has_annot = np.sum(labels_layer.data > 0) != 0
                if has_annot:
                    self._multifile_annotations_store[fname] = np.copy(labels_layer.data)
            except Exception:
                # best-effort push; continue regardless
                pass

        # Assemble lists from the in-memory store
        if not hasattr(self, '_multifile_annotations_store') or not self._multifile_annotations_store:
            warnings.warn('No annotations in memory to train on.')
            return

        folder_text = self.multifile_path_edit.text() if hasattr(self, 'multifile_path_edit') else ''
        if not folder_text:
            warnings.warn('Multifile path is not set. Cannot load images for training.')
            return
        folder = Path(folder_text) if folder_text else None

        filenames = list(self._multifile_annotations_store.keys())
        img_prepared = []
        annots = []
        for fn in filenames:
            # 1. Open the image
            p = folder / fn
            try:
                arr = imageio.imread(str(p))
            except Exception:
                warnings.warn(f'Could not read image {p}. Skipping.')
                continue

            # 2. Prepare it
            try:
                is_rgb = self.cp_model.get_param('channel_mode') == 'rgb'
                dims = arr.ndim if not is_rgb else arr.ndim - 1
                prep = self._get_data_channel_first(arr, dims) if arr is not None else None
            except Exception:
                warnings.warn(f'Could not prepare image {p} for training. Skipping.')
                continue
            img_prepared.append(prep)

            # 3. Get annotations
            # If annotations in store is a path (string), read it now and cache the ndarray
            stored = self._multifile_annotations_store.get(fn, None)
            if isinstance(stored, str):
                try:
                    arr_annot = self.read_labels_file(stored)
                    if arr_annot is not None:
                        self._multifile_annotations_store[fn] = arr_annot
                        annots.append(arr_annot)
                except Exception:
                    warnings.warn(f'Could not read stored annotations for {fn}. Skipping.')
                    continue
            # If it's already an ndarray in memory, use it directly
            elif stored is not None:
                annots.append(np.copy(stored))
            else:
                warnings.warn(f'No annotations found for {fn}. Skipping.')
                continue
        
        # Double-check that after reading annots we still have image/annot pairs ...
        if len(img_prepared) == 0 or len(annots) == 0:
            warnings.warn('No valid image/annotations pairs available for training.')
            return

        # ... and a matching number of them
        if len(img_prepared) != len(annots):
            warnings.warn('Mismatch between loaded images and annotations. Aborting training.')
            return

        # Train on all images

        # Delegate to core trainer
        self._train_multiple(img_prepared, annots, filenames[:len(img_prepared)])

    def _on_segment_selected_multifile(self):
        """Segment selected files from the multifile list and save outputs to disk.

        - Ask for an output folder (defaulting to last used folder).
        - Warn if files will be overwritten.
        - Loop over selected filenames, run backend predict and save segmentation TIFFs
        - Register saved paths in `_multifile_segmentation_store` and update table ticks.
        """
        # Get selected rows
        try:
            sel = self.multifile_list.selectionModel().selectedRows()
            if not sel:
                warnings.warn('No files selected for segmentation.')
                return
            rows = [s.row() for s in sel]
        except Exception:
            warnings.warn('Could not determine selected files.')
            return
        
        # Check that we have a trained model and abort early if not
        if self.cp_model is None or not self.trained:
            warnings.warn('No trained model available for segmentation.')
            return

        # If the user has selected probabilities as outputs, notify them that this is not (yet) supported and segmentations will be created and saved instead
        if self.add_probas:
            show_info('You selected class probabilities as output. This is not yet supported in the Multifile workflow. Segmentations will be created and saved instead.')
        elif not self.add_seg:
            show_info('You have not selected segmentations or probabilities as output. Segmentations will be created and saved by default.')
        
        # Ask for output folder
        default_dir = str(Path(self._multifile_last_folder)) if getattr(self, '_multifile_last_folder', None) else str(Path.cwd())
        folder = QFileDialog.getExistingDirectory(self, 'Select output folder for segmentations', default_dir)
        if not folder:
            return
        out_dir = Path(folder)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Remember last used folder for future dialogs
        self._multifile_last_folder = str(out_dir)

        # Build list of filenames to process
        filenames = []
        for r in rows:
            it = self.multifile_list.item(r, 1)
            if it is None:
                continue
            filenames.append(it.text())

        if not filenames:
            warnings.warn('No valid filenames selected.')
            return

        # Check for existing files and warn if any will be overwritten
        will_overwrite = False
        for fname in filenames:
            stem = Path(fname).stem
            out_name = out_dir / f"{stem}_{self.seg_tag}.tif"
            if out_name.exists():
                will_overwrite = True
                break
        if will_overwrite:
            msg = f"Some files in {out_dir} will be overwritten. Continue?"
            resp = QMessageBox.question(self, 'Overwrite files?', msg, QMessageBox.Yes | QMessageBox.No)
            if resp != QMessageBox.Yes:
                return

        # Segment each selected file
        folder_text = self.multifile_path_edit.text() if hasattr(self, 'multifile_path_edit') else ''
        if not folder_text:
            warnings.warn('Multifile folder not set. Skipping.')
            return
        folder = Path(folder_text) if folder_text else None
        in_channels = self._parse_in_channels(self.input_channels)

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(True)

        with progress(total=len(filenames)) as pbr:
            pbr.set_description('Segmenting')

            segmented = 0
            for fname in filenames:
                pbr.update(1)
                try:
                    img_path = folder / fname
                    arr = imageio.imread(str(img_path))
                except Exception:
                    warnings.warn(f'Could not read image {fname}. Skipping.')
                    continue

                try:
                    is_rgb = self.cp_model.get_param('channel_mode') == 'rgb'
                    dims = arr.ndim if not is_rgb else arr.ndim - 1 # Account for channel dimension in RGB mode
                    prep = self._get_data_channel_first(arr, dims) if arr is not None else None
                except Exception:
                    warnings.warn(f'Could not prepare image {fname} for prediction. Skipping.')
                    continue

                try:
                    probas, seg = self.cp_model._predict(prep, add_seg=True, in_channels=in_channels,
                                                         skip_norm=False, use_dask=self.use_dask,
                                                         fe_use_device=self.fe_device)
                except Exception:
                    warnings.warn(f'Prediction failed for {fname}. Skipping.')
                    continue

                try:
                    stem = Path(fname).stem
                    out_name = out_dir / f"{stem}_{self.seg_tag}.tif"
                    tifffile.imwrite(str(out_name), seg.astype(np.uint8))
                    # Register saved segmentation and update table
                    self._multifile_segmentation_store[fname] = str(out_name)
                    self._update_multifile_seg_tick(fname)
                    segmented += 1
                except Exception:
                    warnings.warn(f'Could not write segmentation for {fname}.')

        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.viewer.window._status_bar._toggle_activity_dock(False)

        if segmented:
            show_info(f'Segmented {segmented} files and saved to {out_dir}')
        else:
            warnings.warn('No images were segmented.')

        # If segmented the opened image, open its segmentation
        current_open = getattr(self, '_current_multifile_filename', None)
        if current_open in filenames and self.multifile_import_open_segmentations:
            self._multifile_open_segmentation(current_open)

    def _import_annot_and_seg(self):
        """Import annotations TIFFs from a folder and register them in the in-memory store.

        Matches files named `<image_stem>_<suffix>.tif|tiff` against the filenames
        listed in the multifile table (by stem). Imported annotations are recorded
        as persistent (stored as path strings) and flagged green.
        """
        # If there are annotations or segmentations in the stores, confirm clearing them (only ask once)
        # if not getattr(self, 'multifile_import_warned', False):
        #     annots_conflict = (self.multifile_import_open_annotations and
        #                        hasattr(self, '_multifile_annotations_store') and
        #                        self._multifile_annotations_store)
        #     seg_conflicts = (self.multifile_import_open_segmentations and
        #                      hasattr(self, '_multifile_segmentation_store') and
        #                      self._multifile_segmentation_store)
        #     if annots_conflict or seg_conflicts:
        #         msg = 'Importing annotations/segmentations might replace existing data in Convpaint. Continue?'
        #         resp = QMessageBox.question(self, 'Replace existing data?', msg, QMessageBox.Yes | QMessageBox.No)
        #         if resp != QMessageBox.Yes:
        #             return
        #         self.multifile_import_warned = True

        # Ask for input folder
        default_dir = str(Path(self._multifile_last_folder)) if getattr(self, '_multifile_last_folder', None) else str(Path.cwd())
        folder = QFileDialog.getExistingDirectory(self, 'Select input folder', default_dir)
        if not folder:
            return
        in_dir = Path(folder)
        # Remember last used folder for future dialogs
        self._multifile_last_folder = str(in_dir)

        # Clear current annotations layer if open (otherwise it might show stale data)
        if self.annot_tag in self.viewer.layers:
            try:
                self.viewer.layers.remove(self.viewer.layers[self.annot_tag])
            except Exception:
                pass

        # Build mapping from image stem -> filename in table
        table_map = {}
        for r in range(self.multifile_list.rowCount()):
            it = self.multifile_list.item(r, 1)
            if it is None:
                continue
            fname = it.text()
            stem = Path(fname).stem
            table_map[stem] = fname

        # Collect candidate files for each target filename (allow for multiple extensions)
        cand_annots = defaultdict(list)
        cand_segs = defaultdict(list)
        for f in in_dir.iterdir():
            if not f.is_file():
                continue
            stem = f.stem
            if '_' not in stem:
                continue
            stem_suffix = stem.split('_')[-1].lower() # e.g. "annotations"
            if not stem_suffix:
                continue
            img_stem = stem[:-len(f'_' + stem_suffix)] if stem.endswith('_' + stem_suffix) else stem
            if img_stem not in table_map:
                continue
            target = table_map[img_stem] # Image filename (incl. extension) as listed in the table
            # Consider both the singular configured suffix and the known list of suffixes
            is_annot = stem_suffix == getattr(self, 'multifile_annot_suffix', '')
            is_seg = stem_suffix == getattr(self, 'multifile_seg_suffix', '')
            if is_annot and self.multifile_import_open_annotations:
                cand_annots[target].append(f)
            if is_seg and self.multifile_import_open_segmentations:
                cand_segs[target].append(f)

        # Decide which candidates are ambiguous (multiple files for same image) and which will overwrite
        ambiguous_annots = [t for t, lst in cand_annots.items() if len(lst) > 1]
        ambiguous_segs = [t for t, lst in cand_segs.items() if len(lst) > 1]
        single_annots = {t: lst[0] for t, lst in cand_annots.items() if len(lst) == 1}
        single_segs = {t: lst[0] for t, lst in cand_segs.items() if len(lst) == 1}

        will_overwrite_annots = [t for t in single_annots.keys() if t in getattr(self, '_multifile_annotations_store', {})]
        will_overwrite_segs = [t for t in single_segs.keys() if t in getattr(self, '_multifile_segmentation_store', {})]

        total_annots = len(single_annots)
        total_segs = len(single_segs)
        amb_count = len(ambiguous_annots) + len(ambiguous_segs)

        if total_annots == 0 and total_segs == 0:
            warnings.warn('No matching annotations or segmentation files found for import.')
            return

        # Ask user with one concise summary, if anything unclear (overwrites, ambiguities, totals)
        if will_overwrite_annots or will_overwrite_segs or amb_count:
            parts = []
            if total_annots:
                parts.append(f'{total_annots} annotations file(s) to import')
            if total_segs:
                parts.append(f'{total_segs} segmentation file(s) to import')
            if will_overwrite_annots:
                parts.append(f'{len(will_overwrite_annots)} annotation(s) will overwrite existing')
            if will_overwrite_segs:
                parts.append(f'{len(will_overwrite_segs)} segmentation(s) will overwrite existing')
            if amb_count:
                parts.append(f'{amb_count} image(s) have multiple matching files and will be skipped')
            msg = 'Found: ' + '; '.join(parts) + '. Proceed?'
            resp = QMessageBox.question(self, 'Import annotations/segmentations?', msg, QMessageBox.Yes | QMessageBox.No)
            if resp != QMessageBox.Yes:
                return

        # Perform import for single candidates
        imported_annots = 0
        imported_segs = 0
        for target, path in single_annots.items():
            try:
                self._multifile_annotations_store[target] = str(path)
                self._update_multifile_annot_tick(target)
                imported_annots += 1
            except Exception:
                warnings.warn(f'Could not register annotations for {target}.')

        for target, path in single_segs.items():
            try:
                self._multifile_segmentation_store[target] = str(path)
                self._update_multifile_seg_tick(target)
                imported_segs += 1
            except Exception:
                warnings.warn(f'Could not register segmentation for {target}.')

        # Notify about ambiguous/skipped targets (report once)
        if amb_count:
            show_info(f'Skipped {amb_count} image(s) that had multiple matching files (check input folder):' +
                      f'{", ".join(ambiguous_annots + ambiguous_segs)}')

        if imported_annots or imported_segs:
            show_info(f'Imported {imported_annots} annotations and {imported_segs} segmentations')
        else:
            warnings.warn('No files were imported.')

        # Finally re-open the current image, so we get back the correct annotations if applicable...
        try:
            current_row = self._current_multifile_row
            if current_row is not None:
                self._on_multifile_open_file(current_row, 1)
        except Exception:
            pass

    def _reset_all_annot_ticks(self):
        """Set the ticks for all files back to the red X. Use this when resetting the annotations store outside of resetting the entire folder."""
        try:
            for r in range(self.multifile_list.rowCount()):
                item_annot = self.multifile_list.item(r, 0)
                if item_annot is not None:
                    item_annot.setText('✗')
                    item_annot.setForeground(QtGui.QBrush(QtGui.QColor('red')))
        except Exception:
            pass
    
    def _reset_all_seg_ticks(self):
        """Set the ticks for all files back to the red X. Use this when resetting the segmentation store outside of resetting the entire folder."""
        try:
            for r in range(self.multifile_list.rowCount()):
                item_seg = self.multifile_list.item(r, 2)
                if item_seg is not None:
                    item_seg.setText('✗')
                    item_seg.setForeground(QtGui.QBrush(QtGui.QColor('red')))
        except Exception:
            pass

    def _export_annotations(self):
        """Export all annotations currently in memory to a chosen folder as TIFF files.

        Files are named `<image_stem>_<suffix>.tif`. After successful export,
        the corresponding table rows are marked as imported (green tick).
        """
        if not hasattr(self, '_multifile_annotations_store') or not self._multifile_annotations_store:
            warnings.warn('No annotations in memory to export.')
            return

        default_dir = str(Path(self._multifile_last_folder)) if getattr(self, '_multifile_last_folder', None) else str(Path.cwd())
        folder = QFileDialog.getExistingDirectory(self, 'Select export folder', default_dir)
        if not folder:
            return
        out_dir = Path(folder)
        out_dir.mkdir(parents=True, exist_ok=True)
        # Remember last used annotations export folder
        self._multifile_last_folder = str(out_dir)

        # Check for existing files and warn if any will be overwritten (always ask)
        will_overwrite = False
        for fname in list(self._multifile_annotations_store.keys()):
            stem = Path(fname).stem
            out_name = out_dir / f"{stem}_{self.multifile_annot_suffix}.tif"
            if out_name.exists():
                will_overwrite = True
                break
        if will_overwrite:
            msg = f"Some files in {out_dir} will be overwritten. Continue?"
            resp = QMessageBox.question(self, 'Overwrite files?', msg, QMessageBox.Yes | QMessageBox.No)
            if resp != QMessageBox.Yes:
                return

        exported = 0
        for fname, annot in list(self._multifile_annotations_store.items()):
            try:
                stem = Path(fname).stem
                out_name = out_dir / f"{stem}_{self.multifile_annot_suffix}.tif"
                # If store contains a string path, read the file first to get array (technically just copying the file)
                if isinstance(annot, str):
                    try:
                        data = self.read_labels_file(annot)
                    except Exception:
                        warnings.warn(f'Could not read stored annotations file {annot}. Skipping.')
                        continue
                    if data is None:
                        warnings.warn(f'Could not read stored annotations file {annot}. Skipping.')
                        continue
                else:
                    data = np.asarray(annot)
                tifffile.imwrite(str(out_name), data.astype(np.uint8))
                # mark as persistent by storing the exported path string
                self._multifile_annotations_store[fname] = str(out_name)
                self._update_multifile_annot_tick(fname)
                exported += 1
            except Exception:
                warnings.warn(f'Could not export annotations for {fname}.')

        if exported:
            show_info(f'Exported {exported} annotations to {out_dir}')
        else:
            warnings.warn('No annotations were exported.')

    def read_labels_file(self, path):
        """Read a labels file from the given path using tifffile or imageio as fallback."""
        # If tiff file, open with tifffile
        if path.lower().endswith(('.tif', '.tiff')):
            try:
                arr = tifffile.imread(path)
                return np.asarray(arr)
            except Exception:
                pass
        # For any other file type, try with imageio
        elif path.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                arr = imageio.imread(path)
                return np.asarray(arr)
            except Exception:
                pass