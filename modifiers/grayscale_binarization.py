import numpy as np

from utils.mod_image_utils import update_mod_image


def modify_to_grayscale(self):
    """
    Converts the image to grayscale and updates the modified image.
    """
    if self.gray_image is not None:
        update_mod_image(self, np.stack([self.gray_image] * 3, axis=-1).astype(np.uint8))

def binarize(self):
    """
    Converts the grayscale image to binary using a threshold value provided by the user.
    """
    if self.gray_image is not None:
        try:
            threshold = int(self.threshold_input.text())
            if not 0 <= threshold <= 255:
                raise ValueError

            binary_image = np.where(self.gray_image > threshold, 255, 0)
            update_mod_image(self, np.stack([binary_image] * 3, axis=-1).astype(np.uint8))

        except ValueError:
            print("Please enter a valid threshold value between 0 and 255.")

