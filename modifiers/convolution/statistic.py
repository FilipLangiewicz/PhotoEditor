import numpy as np

from utils.mod_image_utils import update_mod_image


def statistic(self):
    """
    Applies a statistical operation (neutral, median, min, max) to each pixel neighborhood.
    """
    if self.numpy_image is not None:
        selected_id = self.statistic_options.currentIndex()

        if selected_id == 0:    # neutral
            f = lambda x: x[1, 1]
        elif selected_id == 1:  # median
            f = np.median
        elif selected_id == 2:  # minimum
            f = np.min
        elif selected_id == 3:  # maximum
            f = np.max

        height, width, channels = self.numpy_image.shape
        processed_image = np.zeros((height, width, channels), dtype=int)

        for c in range(channels):
            for i in range(0, height):
                for j in range(0, width):
                    if i == 0 or j == 0 or i == height - 1 or j == width - 1:
                        processed_image[i, j, c] = self.numpy_image[i, j, c]
                        continue
                    region = self.numpy_image[i-1:i+2, j-1:j+2, c]
                    result = f(region)
                    processed_image[i, j, c] = result

        update_mod_image(self, processed_image.astype(np.uint8))
