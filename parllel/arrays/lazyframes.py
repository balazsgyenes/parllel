from typing import Any, Tuple

from gym.wrappers import LazyFrames
import numpy as np
from nptyping import NDArray

from parllel.buffers import Indices

from .rotating import RotatingArray


class LazyFramesArray(RotatingArray):
    def __init__(self,
        *args,
        stack_depth: int,
        frame_ndims: int,
        done: RotatingArray,
        **kwargs,
    ) -> None:
        if not stack_depth == done.padding:
            raise ValueError("The 'done' array must have the same padding as "
                f"the requested stack depth={stack_depth}")

        super().__init__(*args, padding=stack_depth, **kwargs)
        self._stack_depth = stack_depth
        self._frame_ndims = frame_ndims

        # explicitly define t=0 as the first step in each batch
        done[done.first - 1] = True
        self._done = done

    def __setitem__(self, location: Indices, value: Any) -> None:
        if isinstance(value, LazyFrames):
            assert len(value) == self._stack_depth
            return super().__setitem__(location, value[-1])
        return super().__setitem__(location, value)

    def __array__(self, dtype=None) -> NDArray:
        if self._index_history:
            raise NotImplementedError

        if self._current_array is None:
            self._resolve_indexing_history()

        slices = [
            self._base_array[slice(self._padding - t, -self._padding - t)]
            for t in reversed(range(self._stack_depth))
        ]
        array = np.stack(slices, axis=-self._frame_ndims - 1)

        for t in range(1, self._stack_depth):
            done_t_steps_ago = self._done[-t : self._done.last + 1 - t]
            # zero out all but the last t frames
            array[np.asarray(done_t_steps_ago), :-t] = 0

        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array
