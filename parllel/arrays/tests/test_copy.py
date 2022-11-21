import numpy as np

from parllel.arrays.large import LargeArray


if __name__ == "__main__":
    import copy

    arr = LargeArray(shape=(1024, 16), dtype=np.int32, padding=1, apparent_size=128)

    arr2 = copy.copy(arr)

    arr[1] = 1
    arr[2] = 2
    arr[3] = 3
    arr.reset()

    print("done")