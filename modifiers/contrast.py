import numpy as np

from utils.mod_image_utils import update_mod_image


def modify_contrast(self):
    """
    Adjusts the image contrast based on the slider value.
    """
    if self.numpy_image is not None:
        contrast_value = self.contrast_options.value() / 100
        alpha = 1 + contrast_value
        mean_value = np.mean(self.numpy_image)

        adjusted_image = np.clip(alpha * (self.numpy_image.astype(np.float32) - mean_value) + mean_value, 0, 255).astype(np.uint8)
        update_mod_image(self, adjusted_image)
