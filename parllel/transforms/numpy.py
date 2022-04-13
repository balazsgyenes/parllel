import numpy as np
from nptyping import NDArray


def broadcast_across_trailing(*arrays: NDArray):
    max_ndim = max(array.ndim for array in arrays)

    return (
        np.expand_dims(array, tuple(range(-max_ndim + array.ndim, 0)))
        for array in arrays
    )
