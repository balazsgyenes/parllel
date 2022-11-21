import numpy as np

from parllel.arrays.large import shift_index


if __name__ == "__main__":

    indices = [
        5,
        Ellipsis,
        slice(None),
        slice(None, 5),
        slice(5, None, -1),
        slice(None, None, -1),
        np.array([0, 5, 0, 4, 8, -5])
    ]
    shift = 21
    apparent_size = 10

    data = np.arange(102) - 1

    for index in indices:
        print(index)
        print(data[shift_index(index, shift, apparent_size)])
        print()

    print("done")