import numpy as np

from utils.mod_image_utils import update_mod_image


def modify_to_negative(self):
    """
    Creates a negative of the image.
    """
    if self.numpy_image is not None:
        negative_image = 255 - self.numpy_image
        self.numpy_image = negative_image
        update_mod_image(self, negative_image.astype(np.uint8))
