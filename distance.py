import numpy as np

def euclidean_distance(x, y):

    dist = np.sqrt(
        np.sum((x[:, np.newaxis] - y) ** 2, axis=0)
    )

    return dist