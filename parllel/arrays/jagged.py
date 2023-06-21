

from typing import Optional, Tuple
import numpy as np

from parllel.arrays.array import Array
from parllel.buffers import Indices
from .array import Array


class JaggedArray(Array):
    def __init__(self,
        shape: Tuple[int, ...],
        dtype: np.dtype,
        batch_size: Tuple[int, ...],
        *,
        storage: str = "local",
        padding: int = 0,
        full_size: Optional[int] = None,
    ) -> None:

        self.batch_size = batch_size

        self.dtype = dtype
        self.padding = padding

        if batch_size == ():
            T_dim = 1
        else:
            T_dim, *batch_size = batch_size
        
        node_dim, *feature_dims = shape

        # multiply the T dimension into the node dimension
        self._base_shape = batch_size + (T_dim * node_dim,) + feature_dims
        self._apparent_shape = batch_size + shape

        self._buffer_id: int = id(self)
        self._index_history: list[Indices] = []

        self._allocate()

        # TODO: move this into allocate()
        self._ptr = np.zeros(shape=(T_dim,), dtype=np.int32)

        self._current_array = self._base_array
        self._previous_array = self._base_array

        self._resolve_indexing_history()

    def _resolve_indexing_history(self) -> None:
        for index in self._index_history:
            if T: # T dimension still exists
                pass
