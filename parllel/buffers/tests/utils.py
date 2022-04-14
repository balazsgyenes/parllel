import numpy as np

from parllel.buffers import NamedTuple


def buffer_equal(ref, test, /, name=""):
    if isinstance(ref, NamedTuple): # non-leaf node
        for ref_elem, field in zip(ref, ref._fields):
            test_elem = getattr(test, field)
            buffer_equal(ref_elem, test_elem, name + "." + field)
    else:
        if ref is None:
            assert test == ref, name
        else:
            assert np.array_equal(ref, test), name

    # can be called like assert buffer_equal
    return True
