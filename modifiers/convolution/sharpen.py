import numpy as np


def get_sharpen_arr(self):
    """
    Returns sharpening kernel array
    """
    a = self.sharpen_options.value()
    if a == 0:
        return np.array([[0, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]])
    elif a == 1:
        # hp3
        return np.array([[0, -1, 0],
                         [-1, 20, -1],
                         [0, -1, 0]])
    elif a == 2:
        # hp2
        return np.array([[1, -2, 1],
                         [-2, 5, -2],
                         [1, -2, 1]])
    elif a == 3:
        # hp1
        return np.array([[0, -1, 0],
                         [-1, 5, -1],
                         [0, -1, 0]])
    elif a == 4:
        # mean removal
        return np.array([[-1, -1, -1],
                         [-1, 9, -1],
                         [-1, -1, -1]])
