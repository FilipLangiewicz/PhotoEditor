import numpy as np

def get_blur_arr(self):
    """
    Returns a blur matrix based on the selected index.
    """
    selected_id = self.blur_options.currentIndex()
    if selected_id == -1:
        return None

    blur_matrices = {
        0: np.array([[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]]),  # Neutral matrix (one in the middle)
        1: np.array([[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 1]]),  # Averaging filter
        2: np.array([[1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1]]),  # Square filter
        3: np.array([[0, 1, 1, 1, 0],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [0, 1, 1, 1, 0]]),  # Circular filter
        4: np.array([[1, 1, 1],
                     [1, 2, 1],
                     [1, 1, 1]]),  # LP1
        5: np.array([[1, 1, 1],
                     [1, 4, 1],
                     [1, 1, 1]]),  # LP2
        6: np.array([[1, 1, 1],
                     [1, 12, 1],
                     [1, 1, 1]]),  # LP3
        7: np.array([[1, 2, 3, 2, 1],
                     [2, 4, 6, 4, 2],
                     [3, 6, 9, 6, 3],
                     [2, 4, 6, 4, 2],
                     [1, 2, 3, 2, 1]]),  # Pyramidal filter
        8: np.array([[0, 0, 1, 0, 0],
                     [0, 2, 2, 2, 0],
                     [1, 2, 5, 2, 1],
                     [0, 2, 2, 2, 0],
                     [0, 0, 1, 0, 0]]),  # Conical filter
        9: np.array([[1, 1, 1],
                     [2, 4, 2],
                     [1, 1, 1]]),  # Gaussian 1
        10: np.array([[1, 1, 2, 1, 1],
                      [1, 2, 4, 2, 1],
                      [2, 4, 8, 4, 2],
                      [1, 2, 4, 2, 1],
                      [1, 1, 2, 1, 1]]),  # Gaussian 2
        11: np.array([[0, 1, 2, 1, 0],
                      [1, 4, 8, 4, 1],
                      [2, 8, 16, 8, 2],
                      [1, 4, 8, 4, 1],
                      [0, 1, 2, 1, 0]]),  # Gaussian 3
        12: np.array([[1, 4, 7, 4, 1],
                      [4, 16, 26, 16, 4],
                      [7, 26, 41, 26, 7],
                      [4, 16, 26, 16, 4],
                      [1, 4, 7, 4, 1]]),  # Gaussian 4
        13: np.array([[1, 1, 2, 2, 2, 1, 1],
                      [1, 2, 2, 4, 2, 2, 1],
                      [2, 2, 4, 8, 4, 2, 2],
                      [2, 4, 8, 16, 8, 4, 2],
                      [2, 2, 4, 8, 4, 2, 2],
                      [1, 2, 2, 4, 2, 2, 1],
                      [1, 1, 2, 2, 2, 1, 1]]),  # Gaussian 5
    }

    return blur_matrices.get(selected_id, None)
