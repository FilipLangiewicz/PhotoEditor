import os

import numpy as np
from PIL import Image
from PyQt6.QtCore import Qt, QObject
from PyQt6.QtGui import QPixmap, QIntValidator, QImage, QIcon
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QMainWindow, QVBoxLayout, QLabel, QPushButton, QFileDialog, QLineEdit, \
    QSlider, QGridLayout, QComboBox
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from modifiers.convolution.blur import get_blur_arr
from modifiers.brightness import modify_brightness
from modifiers.contrast import modify_contrast
from modifiers.convolution.convolution import convolute
from modifiers.convolution.edge import get_edge_arr, robert_cross
from modifiers.convolution.embossing import get_embossing_arr
from modifiers.grayscale_binarization import binarize, modify_to_grayscale
from modifiers.negative import modify_to_negative
from plots.plots import update_plots
from modifiers.convolution.sharpen import get_sharpen_arr
from modifiers.convolution.statistic import statistic
from utils.mod_image_utils import display_mod_image

basedir = os.path.dirname(__file__)

class PhotoEditor(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Photo Editor")
        self.setWindowIcon(QIcon(os.path.join(basedir, 'www', 'combayns2025.png')))
        self.setWindowState(Qt.WindowState.WindowMaximized)

        # Variables to store images
        self.orig_image = None
        self.mod_image = None
        self.orig_numpy_image = None
        self.numpy_image = None
        self.tmp_numpy_image = None
        self.orig_gray_image = None
        self.gray_image = None
        self.matrix = np.ones((1, 1), dtype=int)

        self.create_ui()

        self.disable_and_clean_all_modifiers()


    def create_ui(self):
        ##### container - main panel
        container_widget = QWidget()
        container_widget.setObjectName("container")
        self.container_layout = QHBoxLayout(container_widget)
        self.setCentralWidget(container_widget)

        self.create_left_panel()
        self.create_right_panel()

    def create_left_panel(self):
        ###### left panel
        left_panel = QWidget()
        left_panel.setObjectName("left_panel")
        left_panel_layout = QVBoxLayout(left_panel)
        self.container_layout.addWidget(left_panel, 11)

        ####### left upper panel
        left_upper_panel = QWidget()
        left_upper_panel.setObjectName("left_upper_panel")
        left_upper_panel_layout = QHBoxLayout(left_upper_panel)
        left_panel_layout.addWidget(left_upper_panel, 7)

        ####### original image panel
        orig_image_panel = QWidget()
        orig_image_panel.setObjectName("orig_image_panel")
        orig_image_panel_layout = QVBoxLayout(orig_image_panel)
        orig_image_panel_layout.setContentsMargins(10, 10, 10, 10)
        left_upper_panel_layout.addWidget(orig_image_panel, 6)

        ######## original image label
        self.orig_image_label = QLabel()
        self.orig_image_label.setObjectName("orig_image_display")
        self.orig_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        orig_image_panel_layout.addWidget(self.orig_image_label, 6)

        ######## button panel
        button_panel = QWidget()
        button_panel_layout = QVBoxLayout(button_panel)
        left_upper_panel_layout.addWidget(button_panel, 1)

        # button to load image
        self.load_button = QPushButton("Load")
        self.load_button.setProperty("class", "main_button")
        button_panel_layout.addWidget(self.load_button)
        self.load_button.clicked.connect(self.load_image)

        # button to restart
        self.restart_button = QPushButton("Restart")
        self.restart_button.setProperty("class", "main_button")
        button_panel_layout.addWidget(self.restart_button)
        self.restart_button.clicked.connect(self.restart)

        # button to save image
        self.save_button = QPushButton("Save")
        self.save_button.setProperty("class", "main_button")
        button_panel_layout.addWidget(self.save_button)
        self.save_button.clicked.connect(self.save_image)

        ####### left lower panel
        left_lower_panel = QWidget()
        left_lower_panel.setObjectName("left_lower_panel")
        left_lower_panel_layout = QHBoxLayout(left_lower_panel)
        left_panel_layout.addWidget(left_lower_panel, 10)

        ######## left lower left panel
        left_lower_left_panel = QWidget()
        left_lower_left_panel.setObjectName("left_lower_left_panel")
        left_lower_left_panel_layout = QVBoxLayout(left_lower_left_panel)
        left_lower_left_panel_layout.setContentsMargins(0, 0, 0, 0)
        left_lower_panel_layout.addWidget(left_lower_left_panel, 6)

        ######### left lower left mod image panel
        left_lower_left_mod_image_panel = QWidget()
        left_lower_left_mod_image_panel.setObjectName("left_lower_left_mod_image_panel")
        left_lower_left_mod_image_panel_layout = QVBoxLayout(left_lower_left_mod_image_panel)
        left_lower_left_mod_image_panel_layout.setContentsMargins(0, 0, 0, 0)
        left_lower_left_panel_layout.addWidget(left_lower_left_mod_image_panel, 7)

        ########## modified image label
        self.mod_image_label = QLabel()
        self.mod_image_label.setObjectName("mod_image_display")
        self.mod_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_lower_left_mod_image_panel_layout.addWidget(self.mod_image_label)

        ########## left lower left full v projection panel
        self.v_projection_full_layout = QHBoxLayout()
        self.left_lower_left_full_v_projection_panel = QWidget()
        self.left_lower_left_full_v_projection_panel.setObjectName("left_lower_left_full_v_projection_panel")
        self.left_lower_left_full_v_projection_panel.setLayout(self.v_projection_full_layout)
        self.v_projection_full_layout.setContentsMargins(2, 2, 2, 2)
        left_lower_left_panel_layout.addWidget(self.left_lower_left_full_v_projection_panel, 3)

        ########## left lower left v projection panel
        self.v_projection_canvas = FigureCanvas(plt.figure())
        self.v_projection_layout = QHBoxLayout()
        self.left_lower_left_v_projection_panel = QWidget()
        self.left_lower_left_v_projection_panel.setObjectName("left_lower_left_v_projection_panel")
        self.left_lower_left_v_projection_panel.setLayout(self.v_projection_layout)
        self.v_projection_layout.setContentsMargins(0, 0, 0, 0)
        self.v_projection_full_layout.addWidget(self.left_lower_left_v_projection_panel, 3)

        ######## left lower right panel
        left_lower_right_panel = QWidget()
        left_lower_right_panel.setObjectName("left_lower_right_panel")
        left_lower_right_panel_layout = QVBoxLayout(left_lower_right_panel)
        left_lower_right_panel_layout.setContentsMargins(0, 0, 0, 0)
        left_lower_panel_layout.addWidget(left_lower_right_panel, 1)

        ########## left lower right full h projection panel
        self.h_full_projection_layout = QVBoxLayout()
        self.left_lower_full_h_projection_panel = QWidget()
        self.left_lower_full_h_projection_panel.setObjectName("left_lower_full_h_projection_panel")
        self.left_lower_full_h_projection_panel.setLayout(self.h_full_projection_layout)
        self.h_full_projection_layout.setContentsMargins(0, 10, 0, 10)
        left_lower_right_panel_layout.addWidget(self.left_lower_full_h_projection_panel, 7)

        ########## left lower h projection panel
        self.h_projection_canvas = FigureCanvas(plt.figure())
        self.h_projection_layout = QVBoxLayout()
        self.left_lower_h_projection_panel = QWidget()
        self.left_lower_h_projection_panel.setObjectName("left_lower_h_projection_panel")
        self.left_lower_h_projection_panel.setLayout(self.h_projection_layout)
        self.h_projection_layout.setContentsMargins(2, 2, 2, 2)
        self.h_full_projection_layout.addWidget(self.left_lower_h_projection_panel)

        ########## logo panel
        self.box_layout = QVBoxLayout()
        self.box_panel = QWidget()
        self.box_panel.setObjectName("box")
        self.box_panel.setLayout(self.box_layout)
        self.box_layout.setContentsMargins(0, 0, 0, 0)
        self.box_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_lower_right_panel_layout.addWidget(self.box_panel, 3)
        image_label = QLabel(self.box_panel)
        image_label.setObjectName("logo")
        self.box_layout.addWidget(image_label)
        pixmap = QPixmap(os.path.join(basedir, 'www', 'combayns2025.png'))

        target_height = 60
        target_width = self.box_panel.width()
        scaled_pixmap = pixmap.scaled(target_width,
                                      target_height,
                                      Qt.AspectRatioMode.KeepAspectRatio)
        image_label.setPixmap(scaled_pixmap)

    def create_right_panel(self):
        ###### right panel
        right_panel = QWidget()
        right_panel.setObjectName("right_panel")
        right_panel_layout = QVBoxLayout(right_panel)
        self.container_layout.addWidget(right_panel, 9)

        ###### right upper panel
        right_upper_panel = QWidget()
        right_upper_panel.setObjectName("right_upper_panel")
        right_upper_panel_layout = QHBoxLayout(right_upper_panel)
        right_panel_layout.addWidget(right_upper_panel, 1)

        # histograms container
        self.histograms_container = QWidget()
        self.histograms_container.setObjectName("histograms_container")
        self.histograms_layout = QHBoxLayout(self.histograms_container)
        self.histograms_layout.setContentsMargins(0, 0, 0, 0)
        right_upper_panel_layout.addWidget(self.histograms_container)

        # histograms
        self.hist_fig, self.hist_axes = plt.subplots(2, 2, figsize=(8, 6))
        self.hist_fig.set_facecolor("#C1C1C1")
        self.hist_axes[0, 0].set_facecolor("#C1C1C1")
        self.hist_axes[0, 1].set_facecolor("#C1C1C1")
        self.hist_axes[1, 0].set_facecolor("#C1C1C1")
        self.hist_axes[1, 1].set_facecolor("#C1C1C1")
        for ax in self.hist_axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
        self.hist_canvas = FigureCanvas(self.hist_fig)
        self.histograms_layout.addWidget(self.hist_canvas)

        # right bottom panel
        right_bottom_panel = QWidget()
        right_bottom_panel.setObjectName("right_bottom_panel")
        self.right_bottom_panel_layout = QVBoxLayout(right_bottom_panel)
        right_panel_layout.addWidget(right_bottom_panel, 1)

        # simple buttons panel
        simple_buttons = QWidget()
        simple_buttons.setObjectName("simple_buttons")
        simple_buttons_layout = QHBoxLayout(simple_buttons)
        simple_buttons_layout.setContentsMargins(0, 0, 0, 0)
        self.right_bottom_panel_layout.addWidget(simple_buttons)

        # grayscale button
        self.grayscale_button = QPushButton("Convert to Grayscale")
        simple_buttons_layout.addWidget(self.grayscale_button)
        self.grayscale_button.clicked.connect(self.perform_grayscale)

        # binarization button
        self.binary_button = QPushButton("Binarize")
        simple_buttons_layout.addWidget(self.binary_button)
        self.binary_button.clicked.connect(self.perform_binarize)

        # Add threshold input field
        self.threshold_input = QLineEdit()
        self.threshold_input.setPlaceholderText("Enter threshold (0-255)")
        self.threshold_input.setValidator(QIntValidator(0, 255))
        simple_buttons_layout.addWidget(self.threshold_input)

        # negative button
        self.negative_button = QPushButton("Negative")
        simple_buttons_layout.addWidget(self.negative_button)
        self.negative_button.clicked.connect(self.perform_negative)

        # brigthness_contrast_panel
        brightness_contrast_panel = QWidget()
        brightness_contrast_panel.setObjectName("brightness_contrast_panel")
        brightness_contrast_layout = QHBoxLayout(brightness_contrast_panel)
        brightness_contrast_layout.setContentsMargins(0, 0, 0, 0)
        self.right_bottom_panel_layout.addWidget(brightness_contrast_panel)

        # brightness panel
        self.brightness_panel = QWidget()
        self.brightness_panel.setObjectName("brightness_panel")
        self.brightness_panel.setProperty("class", "modifier-panel")
        brightness_contrast_layout.addWidget(self.brightness_panel)
        brigthness_layout = QHBoxLayout(self.brightness_panel)

        # brightness button
        self.brightness_button = QPushButton("Brightness")
        self.brightness_button.setObjectName("brightness_button")
        self.brightness_button.setProperty("class", "modifier-button")
        brigthness_layout.addWidget(self.brightness_button)
        self.brightness_button.clicked.connect(lambda: self.operation_clicked("brightness"))

        # brightness slider
        self.brightness_options = QSlider(Qt.Orientation.Horizontal)
        self.brightness_options.setObjectName("brightness_options")
        self.brightness_options.setMinimum(-255)
        self.brightness_options.setMaximum(255)
        self.brightness_options.setValue(0)
        self.brightness_options.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.brightness_options.setTickInterval(10)
        brigthness_layout.addWidget(self.brightness_options)
        self.brightness_options.sliderReleased.connect(self.perform_brightness)

        # contrast panel
        self.contrast_panel = QWidget()
        self.contrast_panel.setObjectName("contrast_panel")
        self.contrast_panel.setProperty("class", "modifier-panel")
        brightness_contrast_layout.addWidget(self.contrast_panel)
        contrast_layout = QHBoxLayout(self.contrast_panel)

        # contrast button
        self.contrast_button = QPushButton("Contrast")
        self.contrast_button.setObjectName("contrast_button")
        self.contrast_button.setProperty("class", "modifier-button")
        contrast_layout.addWidget(self.contrast_button)
        self.contrast_button.clicked.connect(lambda: self.operation_clicked("contrast"))

        # contrast slider
        self.contrast_options = QSlider(Qt.Orientation.Horizontal)
        self.contrast_options.setObjectName("contrast_options")
        self.contrast_options.setMinimum(-100)
        self.contrast_options.setMaximum(100)
        self.contrast_options.setValue(0)
        contrast_layout.addWidget(self.contrast_options)
        self.contrast_options.sliderReleased.connect(self.perform_contrast)

        # sharpen panel
        self.sharpen_panel = QWidget()
        self.sharpen_panel.setObjectName("sharpen_panel")
        self.sharpen_panel.setProperty("class", "modifier-panel")
        sharpen_layout = QHBoxLayout(self.sharpen_panel)
        self.right_bottom_panel_layout.addWidget(self.sharpen_panel)

        # sharpen button
        self.sharpen_button = QPushButton("Sharpen")
        self.sharpen_button.setObjectName("sharpen_button")
        self.sharpen_button.setProperty("class", "modifier-button")
        sharpen_layout.addWidget(self.sharpen_button)
        self.sharpen_button.clicked.connect(lambda: self.operation_clicked("sharpen"))

        # sharpen slider
        self.sharpen_options = QSlider(Qt.Orientation.Horizontal)
        self.sharpen_options.setObjectName("sharpen_options")
        self.sharpen_options.setMinimum(0)
        self.sharpen_options.setMaximum(4)
        self.sharpen_options.setValue(0)
        sharpen_layout.addWidget(self.sharpen_options)
        self.sharpen_options.setEnabled(False)
        self.sharpen_options.sliderReleased.connect(self.perform_sharpen)

        # blur panel
        blur_options = [
            "None", "Mean", "Square", "Circle", "LP1", "LP2", "LP3", "Pyramid",
            "Cone", "Gauss1", "Gauss2", "Gauss3", "Gauss4", "Gauss5"
        ]
        self.create_modifier_panel("blur", blur_options)

        # edge panel
        edge_options = [
            "None", "Horizontal", "Vertical", "Diagonal /", "Diagonal \\",
            "East", "Southeast", "South", "Southwest", "West",
            "Northwest", "North", "Northeast",
            "Laplacian 1", "Laplacian 2", "Laplacian 3", "Laplacian Diagonal",
            "Laplacian Horizontal", "Laplacian Vertical",
            "Sobel Horizontal", "Sobel Vertical",
            "Prewitt Horizontal", "Prewitt Vertical", "Robert's Cross 1", "Robert's Cross 2"
        ]
        self.create_modifier_panel("edge", edge_options)

        # blur and edge panel
        blur_edge_panel = QWidget()
        blur_edge_panel.setObjectName("blur_edge_panel")
        blur_edge_layout = QHBoxLayout(blur_edge_panel)
        blur_edge_layout.setContentsMargins(0, 0, 0, 0)
        self.right_bottom_panel_layout.addWidget(blur_edge_panel)
        blur_edge_layout.addWidget(self.blur_panel)
        blur_edge_layout.addWidget(self.edge_panel)

        # embossing panel
        embossing_options = [
            "None", "East", "Southeast", "South", "Southwest",
            "West", "Northwest", "North", "Northeast"
        ]
        self.create_modifier_panel("embossing", embossing_options)

        # statistic panel
        statistic_options = [
            "None", "Median", "Minimum", "Maximum"
        ]
        self.create_modifier_panel("statistic", statistic_options)

        # embossing and statistic panel
        embossing_statistic_panel = QWidget()
        embossing_statistic_panel.setObjectName("embossing_statistic_panel")
        embossing_statistic_layout = QHBoxLayout(embossing_statistic_panel)
        embossing_statistic_layout.setContentsMargins(0, 0, 0, 0)
        self.right_bottom_panel_layout.addWidget(embossing_statistic_panel)
        embossing_statistic_layout.addWidget(self.embossing_panel)
        embossing_statistic_layout.addWidget(self.statistic_panel)

        # manual matrix panel
        manual_matrix = QWidget()
        manual_matrix_layout = QHBoxLayout(manual_matrix)
        manual_matrix_layout.setContentsMargins(0, 0, 0, 0)
        manual_matrix.setObjectName("manual_matrix")
        self.right_bottom_panel_layout.addWidget(manual_matrix)

        # manual matrix buttons
        self.manual_matrix_buttons = QWidget()
        manual_matrix_buttons_layout = QVBoxLayout(self.manual_matrix_buttons)
        self.manual_matrix_buttons.setObjectName("manual_matrix_buttons")
        self.manual_matrix_buttons.setProperty("class", "modifier-panel")
        manual_matrix_layout.addWidget(self.manual_matrix_buttons, 1)

        # manual matrix label
        label = QLabel("SET YOUR WEIGHTS")
        manual_matrix_buttons_layout.addWidget(label)
        label.setObjectName("manual_matrix_label")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 33 button
        self.matrix_33_button = QPushButton("3x3")
        manual_matrix_buttons_layout.addWidget(self.matrix_33_button)
        self.matrix_33_button.setObjectName("matrix_33_button")
        self.matrix_33_button.clicked.connect(lambda: self.manual_matrix_clicked(3))

        # 55 button
        self.matrix_55_button = QPushButton("5x5")
        manual_matrix_buttons_layout.addWidget(self.matrix_55_button)
        self.matrix_55_button.setObjectName("matrix_55_button")
        self.matrix_55_button.clicked.connect(lambda: self.manual_matrix_clicked(5))

        # 77 button
        self.matrix_77_button = QPushButton("7x7")
        manual_matrix_buttons_layout.addWidget(self.matrix_77_button)
        self.matrix_77_button.setObjectName("matrix_77_button")
        self.matrix_77_button.clicked.connect(lambda: self.manual_matrix_clicked(7))

        # convolute button
        self.convolute_button = QPushButton("Perform")
        manual_matrix_buttons_layout.addWidget(self.convolute_button)
        self.convolute_button.clicked.connect(self.perform_manual_convolution)
        self.convolute_button.setEnabled(False)

        # matrix
        grid_container = QWidget()
        grid_container.setObjectName("grid_container")
        grid_layout = QGridLayout(grid_container)
        for row in range(7):
            for col in range(7):
                input_field = QLineEdit()
                input_field.setValidator(QIntValidator(-99, 99))
                input_field.setText("0")
                input_field.setObjectName(f"matrix_{row}_{col}")
                input_field.setProperty("class", "matrix-cell-disabled")
                input_field.setEnabled(False)
                grid_layout.addWidget(input_field, row, col)
        manual_matrix_layout.addWidget(grid_container, 2)

    def create_modifier_panel(self, operation_id, options_list):
        panel = QWidget()
        panel.setObjectName(f"{operation_id}_panel")
        panel.setProperty("class", "modifier-panel")
        layout = QHBoxLayout(panel)

        button = QPushButton(operation_id.capitalize())
        button.setObjectName(f"{operation_id}_button")
        button.setProperty("class", "modifier-button")
        layout.addWidget(button)

        options = QComboBox()
        options.setObjectName(f"{operation_id}_options")
        options.addItems(options_list)
        options.setEnabled(False)
        layout.addWidget(options)

        options.currentIndexChanged.connect(getattr(self, f"perform_{operation_id}"))

        setattr(self, f"{operation_id}_button", button)
        setattr(self, f"{operation_id}_options", options)
        setattr(self, f"{operation_id}_panel", panel)

        button.clicked.connect(lambda: self.operation_clicked(operation_id))


    ######### BACKEND FUNCTIONS #########

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            pixmap = QPixmap(file_path)

            self.orig_image = pixmap.toImage()
            self.mod_image = self.orig_image.copy()
            self.orig_numpy_image = np.array(Image.open(file_path).convert("RGB"))

            scaled_pixmap = pixmap.scaled(
                self.orig_image_label.width(),
                self.orig_image_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            self.load_and_clean(scaled_pixmap)

    def load_and_clean(self, scaled_pixmap):
        self.orig_image_label.setFixedSize(self.orig_image_label.width(), self.orig_image_label.height())
        self.orig_image_label.setPixmap(scaled_pixmap)

        display_mod_image(self, scaled_pixmap)

        self.numpy_image = self.orig_numpy_image.copy()
        self.tmp_numpy_image = self.numpy_image.copy()
        self.orig_gray_image = self.convert_to_grayscale()
        self.gray_image = self.orig_gray_image.copy()

        self.clean_app()

    def restart(self):
        if self.orig_image:
            self.mod_image = self.orig_image.copy()

            scaled_pixmap = QPixmap.fromImage(self.mod_image).scaled(
                self.orig_image_label.width(),
                self.orig_image_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            self.load_and_clean(scaled_pixmap)

    def save_image(self):
        if self.tmp_numpy_image is not None:
            height, width, channel = self.tmp_numpy_image.shape
            bytes_per_line = 3 * width
            qimage = QImage(self.tmp_numpy_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)

            if qimage is None or qimage.isNull():
                print("No image to save!")
                return

            default_filename = "edited_image.png"
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", default_filename, "Images (*.png *.jpg *.jpeg *.bmp)")

            if file_path:
                if not qimage.save(file_path):
                    print("Image save error!")
                else:
                    print(f"Image save as: {file_path}")
            else:
                print("Save canceled by user.")

    def convert_to_grayscale(self):
        if self.tmp_numpy_image is not None:
            grayscale_image = np.mean(self.tmp_numpy_image, axis=2)
            return grayscale_image
        else:
            return None

    def clean_app(self):
        self.clean_all()
        update_plots(self)

    def clean_all(self):
        self.disable_and_clean_all_modifiers()
        self.enable_all_buttons()
        self.clean_matrix()

    def disable_and_clean_all_modifiers(self):
        self.save_numpy_tmp()
        self.disable_and_clean()

    def save_numpy_tmp(self):
        if self.tmp_numpy_image is not None:
            self.numpy_image = self.tmp_numpy_image.copy()

    def disable_and_clean(self):
        self.contrast_options.setEnabled(False)
        self.contrast_options.setValue(0)
        self.brightness_options.setEnabled(False)
        self.brightness_options.setValue(0)
        self.sharpen_options.setEnabled(False)
        self.sharpen_options.setValue(0)

        self.brightness_panel.setProperty("class", "modifier-panel")
        self.brightness_panel.style().unpolish(self.brightness_panel)
        self.brightness_panel.style().polish(self.brightness_panel)

        self.contrast_panel.setProperty("class", "modifier-panel")
        self.contrast_panel.style().unpolish(self.contrast_panel)
        self.contrast_panel.style().polish(self.contrast_panel)

        self.sharpen_panel.setProperty("class", "modifier-panel")
        self.sharpen_panel.style().unpolish(self.sharpen_panel)
        self.sharpen_panel.style().polish(self.sharpen_panel)

        self.blur_panel.setProperty("class", "modifier-panel")
        self.blur_panel.style().unpolish(self.blur_panel)
        self.blur_panel.style().polish(self.blur_panel)

        self.edge_panel.setProperty("class", "modifier-panel")
        self.edge_panel.style().unpolish(self.edge_panel)
        self.edge_panel.style().polish(self.edge_panel)

        self.embossing_panel.setProperty("class", "modifier-panel")
        self.embossing_panel.style().unpolish(self.embossing_panel)
        self.embossing_panel.style().polish(self.embossing_panel)

        self.statistic_panel.setProperty("class", "modifier-panel")
        self.statistic_panel.style().unpolish(self.statistic_panel)
        self.statistic_panel.style().polish(self.statistic_panel)

        self.manual_matrix_buttons.setProperty("class", "modifier-panel")
        self.manual_matrix_buttons.style().unpolish(self.manual_matrix_buttons)
        self.manual_matrix_buttons.style().polish(self.manual_matrix_buttons)

    def enable_all_buttons(self):
        self.contrast_button.setEnabled(True)
        self.brightness_button.setEnabled(True)
        self.sharpen_button.setEnabled(True)
        self.blur_button.setEnabled(True)
        self.disable_operation_options("blur")
        self.edge_button.setEnabled(True)
        self.disable_operation_options("edge")
        self.embossing_button.setEnabled(True)
        self.disable_operation_options("embossing")
        self.statistic_button.setEnabled(True)
        self.disable_operation_options("statistic")
        self.matrix_33_button.setEnabled(True)
        self.matrix_55_button.setEnabled(True)
        self.matrix_77_button.setEnabled(True)
        self.convolute_button.setEnabled(False)

    def disable_operation_options(self, operation_id):
        options = self.findChild(QComboBox, f"{operation_id}_options")
        options.setEnabled(False)
        options.setCurrentIndex(0)

    def clean_matrix(self):
        for row in range(7):
            for col in range(7):
                input_field = self.findChild(QLineEdit, f"matrix_{row}_{col}")
                input_field.setText("0")
                input_field.setEnabled(False)
                input_field.setProperty("class", "matrix-cell-disabled")
                input_field.style().unpolish(input_field)
                input_field.style().polish(input_field)

    def operation_clicked(self, operation_id):
        self.clean_all()
        if self.numpy_image is not None:
            button = self.findChild(QPushButton, f"{operation_id}_button")
            button.setEnabled(False)
            self.findChild(QObject, f"{operation_id}_options").setEnabled(True)
            panel = self.findChild(QWidget, f"{operation_id}_panel")
            panel.setProperty("class", "modifier-panel-active")
            panel.style().unpolish(panel)
            panel.style().polish(panel)

    def manual_matrix_clicked(self, size):
        self.clean_all()
        if self.numpy_image is not None:
            self.disable_and_clean()
            self.enable_all_buttons()
            self.manual_matrix_buttons.setProperty("class", "modifier-panel-active")
            self.manual_matrix_buttons.style().unpolish(self.manual_matrix_buttons)
            self.manual_matrix_buttons.style().polish(self.manual_matrix_buttons)
            self.convolute_button.setEnabled(True)

            button = self.findChild(QPushButton, f"matrix_{size}{size}_button")
            button.setEnabled(False)

            for row in range(7):
                for col in range(7):
                    input_field = self.findChild(QLineEdit, f"matrix_{row}_{col}")
                    if row in range(int((7 - size) / 2), int((7 + size) / 2)) and col in range(int((7 - size) / 2), int((7 + size) / 2)):
                        input_field.setEnabled(True)
                        input_field.setProperty("class", "matrix-cell")
                    else:
                        input_field.setEnabled(False)
                        input_field.setProperty("class", "matrix-cell-disabled")
                    input_field.style().unpolish(input_field)
                    input_field.style().polish(input_field)

    def load_matrix(self):
        k = 0
        for i in range(7):
            input_field = self.findChild(QLineEdit, f"matrix_{i}_{i}")
            if input_field.isEnabled():
                k = i
                break
        self.matrix = np.zeros((7 - 2 * k, 7 - 2 * k), dtype=int)

        for row in range(k, 7 - k):
                for col in range(k, 7 - k):
                    input_field = self.findChild(QLineEdit, f"matrix_{row}_{col}")
                    self.matrix[row - k, col - k] = int(input_field.text()) if input_field.text() != '' else 0

    def print_matrix_from_array(self, array):
        shape = array.shape[0]
        for row in range(7):
            for col in range(7):
                input_field = self.findChild(QLineEdit, f"matrix_{row}_{col}")
                if row in range(int((7 - shape) / 2), int((7 + shape) / 2)) and col in range(int((7 - shape) / 2), int((7 + shape) / 2)):
                    input_field.setText(str(array[row - int((7 - shape) / 2), col - int((7 - shape) / 2)]))
                    input_field.setProperty("class", "matrix-cell")
                else:
                    input_field.setText("0")
                    input_field.setProperty("class", "matrix-cell-disabled")
                input_field.setEnabled(False)
                input_field.style().unpolish(input_field)
                input_field.style().polish(input_field)

    def print_matrix_from_array_robert_cross(self, array):
        arr1 = array[0]
        arr2 = array[1]

        for row in range(7):
            for col in range(7):
                input_field = self.findChild(QLineEdit, f"matrix_{row}_{col}")
                input_field.setText("0")
                input_field.setProperty("class", "matrix-cell-disabled")
                input_field.style().unpolish(input_field)
                input_field.style().polish(input_field)

        self.print_robert_cell(arr1, 2, 1)
        self.print_robert_cell(arr2, 2, 4)

    def print_robert_cell(self, array, i, j):
        for row in range(i, i + 2):
            for col in range(j, j + 2):
                input_field = self.findChild(QLineEdit, f"matrix_{row}_{col}")
                input_field.setText(str(array[row - i, col - j]))
                input_field.setProperty("class", "matrix-cell")
                input_field.setEnabled(False)
                input_field.style().unpolish(input_field)
                input_field.style().polish(input_field)

    def disable_matrix(self):
        for row in range(7):
            for col in range(7):
                input_field = self.findChild(QLineEdit, f"matrix_{row}_{col}")
                input_field.setEnabled(False)


    ######### MODIFIERS #########
    def perform_grayscale(self):
        self.clean_all()
        modify_to_grayscale(self)

    def perform_binarize(self):
        self.clean_all()
        binarize(self)

    def perform_negative(self):
        self.clean_all()
        modify_to_negative(self)

    def perform_brightness(self):
        modify_brightness(self)

    def perform_contrast(self):
        modify_contrast(self)

    def perform_convolution(self, arr):
        self.disable_matrix()
        self.print_matrix_from_array(arr)
        convolute(self)

    def perform_manual_convolution(self):
        convolute(self)

    def perform_sharpen(self):
        arr = get_sharpen_arr(self)
        self.perform_convolution(arr)

    def perform_blur(self):
        if not self.blur_button.isEnabled():
            arr = get_blur_arr(self)
            if arr is not None:
                self.perform_convolution(arr)

    def perform_edge(self):
        if not self.edge_button.isEnabled():
            arr = get_edge_arr(self)
            if arr is not None:
                if len(arr.shape) == 3:
                    robert_cross(self, arr)
                    self.disable_matrix()
                    self.print_matrix_from_array_robert_cross(arr)
                else:
                    self.perform_convolution(arr)

    def perform_embossing(self):
        if not self.embossing_button.isEnabled():
            arr = get_embossing_arr(self)
            if arr is not None:
                self.perform_convolution(arr)

    def perform_statistic(self):
        if not self.statistic_button.isEnabled():
            self.clean_matrix()
            statistic(self)
