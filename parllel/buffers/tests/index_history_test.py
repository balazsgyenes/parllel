import pytest
import numpy as np

from parllel.buffers.named_tuple import (
    NamedTuple, NamedTupleClass,
    NamedArrayTuple, NamedArrayTupleClass,
)


shape = (5,4,3)
fields = ("a", "b", "c")
elements = tuple(i*np.ones(shape) for i in range(len(fields)))

class TestIndexAware:
    def test_index_history(self):
        tup = NamedArrayTuple("test", fields, elements)

        assert tup.index_history == ()

        view = tup[2, 2:3]

        assert view.index_history == ((2, slice(2,3)),)
        assert view.buffer_id == tup.buffer_id

        replaced = tup._replace(a=(42*np.ones(shape)))

        assert replaced.index_history == ()
        assert replaced.buffer_id != tup.buffer_id
