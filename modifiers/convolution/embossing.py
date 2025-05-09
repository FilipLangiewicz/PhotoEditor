import numpy as np

def get_embossing_arr(self):
    """
    Returns an embossing matrix based on the selected option.
    """
    selected_id = self.embossing_options.currentIndex()
    if selected_id == -1:
        return None

    embossing_matrices = {
        0: np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ]),  # Neutral

        1: np.array([
            [-1,  0,  1],
            [-1,  1,  1],
            [-1,  0,  1]
        ]),  # East

        2: np.array([
            [-1, -1,  0],
            [-1,  1,  1],
            [0,   1,  1]
        ]),  # Southeast

        3: np.array([
            [-1, -1, -1],
            [0,   1,  0],
            [1,   1,  1]
        ]),  # South

        4: np.array([
            [0,  -1, -1],
            [1,   1, -1],
            [1,   1,  0]
        ]),  # Southwest

        5: np.array([
            [1,  0, -1],
            [1,  1, -1],
            [1,  0, -1]
        ]),  # West

        6: np.array([
            [1,   1,  0],
            [1,   1, -1],
            [0,  -1, -1]
        ]),  # Northwest

        7: np.array([
            [1,   1,  1],
            [0,   1,  0],
            [-1, -1, -1]
        ]),  # North

        8: np.array([
            [0,   1,  1],
            [-1,  1,  1],
            [-1, -1,  0]
        ])   # Northeast
    }

    return embossing_matrices.get(selected_id, np.eye(3))
