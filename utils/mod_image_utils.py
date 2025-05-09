from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap

from plots.plots import update_plots


def update_mod_image(self, image):
    """
    Converts the numpy array to a PIL Image
    """
    self.mod_image = Image.fromarray(image)
    q_image = ImageQt(self.mod_image)
    pixmap = QPixmap.fromImage(q_image)
    self.tmp_numpy_image = image
    display_mod_image(self, pixmap)
    update_plots(self)

def display_mod_image(self, pixmap):
    """
    Scales the QPixmap to fit the label size and displays it.
    """
    scaled_pixmap = pixmap.scaled(
        self.orig_image_label.width(),
        self.orig_image_label.height(),
        Qt.AspectRatioMode.KeepAspectRatio,
    )
    self.mod_image_label.setPixmap(scaled_pixmap)