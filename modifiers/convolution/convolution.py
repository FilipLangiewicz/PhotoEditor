import numpy as np

from utils.mod_image_utils import update_mod_image


def convolute(self):
    """
    Applies convolution on the image using the selected mask.
    """
    if self.numpy_image is not None:
        self.load_matrix()
        mask = self.matrix  # Pobieramy macierz konwolucji
        convoluted_image = convolution(self, mask)

        update_mod_image(self, convoluted_image)

def convolution(self, mask):
    """
    Performs convolution on the image with the given mask for each channel.
    """
    mask_size = mask.shape[0]
    pad = mask_size // 2

    weights_sum = mask.sum()

    height, width, channels = self.numpy_image.shape
    convoluted_image = np.zeros((height, width, channels), dtype=int)

    padded_image = np.pad(self.numpy_image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')

    for c in range(channels):
        for i in range(height):
            for j in range(width):
                region = padded_image[i:i + mask_size, j:j + mask_size, c]
                result = np.sum(region * mask)
                convoluted_image[i, j, c] = (1 / weights_sum) * result if weights_sum != 0 else result

    convoluted_image = np.clip(convoluted_image, 0, 255).astype(np.uint8)
    return convoluted_image
