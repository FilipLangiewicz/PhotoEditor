import numpy as np

from utils.mod_image_utils import update_mod_image


def modify_brightness(self):
    """
    Adjusts the image brightness by adding the selected value to the pixel intensities.
    """
    if self.numpy_image is not None:
        brightness_value = self.brightness_options.value()
        adjusted_image = np.clip(self.numpy_image.astype(np.int16) + brightness_value, 0, 255).astype(np.uint8)
        update_mod_image(self, adjusted_image)
