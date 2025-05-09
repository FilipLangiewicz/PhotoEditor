import numpy as np

from modifiers.convolution.convolution import convolution
from utils.mod_image_utils import update_mod_image


def get_edge_arr(self):
    """
    Returns the selected edge detection matrix based on user input.
    """
    selected_id = self.edge_options.currentIndex()
    if selected_id == -1:
        return None

    edge_matrices = {
        0: np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ]),  # Neutral matrix

        1: np.array([
            [0, 0, 0],
            [-1, 1, 0],
            [0, 0, 0]
        ]),  # Horizontal

        2: np.array([
            [0, -1, 0],
            [0, 1, 0],
            [0, 0, 0]
        ]),  # Vertical

        3: np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ]),  # Diagonal /

        4: np.array([
            [0, 0, -1],
            [0, 1, 0],
            [0, 0, 0]
        ]),  # Diagonal \

        5: np.array([
            [-1, 1, 1],
            [-1, -2, 1],
            [-1, 1, 1]
        ]),  # East

        6: np.array([
            [-1, -1, 1],
            [-1, -2, 1],
            [1, 1, 1]
        ]),  # Southeast

        7: np.array([
            [-1, -1, -1],
            [1, -2, 1],
            [1, 1, 1]
        ]),  # South

        8: np.array([
            [1, -1, -1],
            [1, -2, -1],
            [1, 1, 1]
        ]),  # Southwest

        9: np.array([
            [1, 1, -1],
            [1, -2, -1],
            [1, 1, -1]
        ]),  # West

        10: np.array([
            [1, 1, 1],
            [1, -2, -1],
            [1, -1, -1]
        ]),  # Northwest

        11: np.array([
            [1, 1, 1],
            [1, -2, 1],
            [-1, -1, -1]
        ]),  # North

        12: np.array([
            [1, 1, 1],
            [-1, -2, 1],
            [-1, -1, 1]
        ]),  # Northeast

        13: np.array([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ]),  # LAPL1

        14: np.array([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ]),  # LAPL2

        15: np.array([
            [1, -2, 1],
            [-2, 4, -2],
            [1, -2, 1]
        ]),  # LAPL3

        16: np.array([
            [-1, 0, -1],
            [0, 4, 0],
            [-1, 0, -1]
        ]),  # Laplace'a diagonal

        17: np.array([
            [0, -1, 0],
            [0, 2, 0],
            [0, -1, 0]
        ]),  # Laplace'a horizontal

        18: np.array([
            [0, 0, 0],
            [-1, 2, -1],
            [0, 0, 0]
        ]),  # Laplace'a vertical

        19: np.array([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ]),  # Horizontal Sobel

        20: np.array([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]
        ]),  # Vertical Sobel

        21: np.array([
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]
        ]),  # Horizontal Prewitt

        22: np.array([
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1]
        ]),  # Vertical Prewitt

        23: np.array([
                np.array([
                [1, -1],
                [0, 0]
            ]),
                np.array([
                [1, 0],
                [-1, 0]
            ])
        ]),  # Robert Cross 1

        24: np.array([
            np.array([
                [1, 0],
                [0, -1]
            ]),
            np.array([
                [0, 1],
                [-1, 0]
            ])
        ])  # Robert Cross 2
    }

    return edge_matrices.get(selected_id, None)

def robert_cross(self, arr):
    """
    Applies convolution using both Robert Cross masks and combines the results.
    """
    conv1 = np.abs(convolution(self, arr[0]))
    conv2 = np.abs(convolution(self, arr[1]))

    convoluted_image = conv1 + conv2
    update_mod_image(self, convoluted_image)
