import numpy as np


def broadcast_left_to_right(*arrays: np.ndarray):
    """Make arrays have the same dimensionality by adding extra trailing
    singleton dimensions where needed.

    Normal broadcasting behaviour by numpy adds implicit singleton dimensions
    as leading dimensions. Use this function to force the opposite behaviour.

    e.g.
    A      (2d array):  8 x 7           -> 8 x 7 x 1 x 1
    B      (4d array):  8 x 1 x 1 x 6   -> 8 x 1 x 1 x 6
    C      (3d array):  1 x 1 x 5       -> 1 x 1 x 5 x 1

    Result (4d array):                     8 x 7 x 5 x 6
    (Result is the result of a later array operation on A, B, and C.)
    """
    max_ndim = max(array.ndim for array in arrays)

    return (
        np.expand_dims(array, tuple(range(-max_ndim + array.ndim, 0)))
        for array in arrays
    )
