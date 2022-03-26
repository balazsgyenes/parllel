from typing import Any

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
        reset_mode: str = "zero",  # repeat | zero
        join_op: str = "stack",  # stack | concat
        **kwargs,
    ) -> None:
        super().__init__(*args, padding=stack_depth, **kwargs)
        self._stack_depth = stack_depth
        self._frame_ndims = frame_ndims

        if not stack_depth == done.padding:
            raise ValueError("The 'done' array must have the same padding as "
                f"the requested stack depth={stack_depth}")
        # explicitly define t=0 as the first step in each batch
        done[done.first - 1] = True
        self._done = done

        if reset_mode not in {"repeat", "zero"}:
            raise ValueError("reset mode must be 'repeat' or 'zero'")
        self._reset_mode = reset_mode

        if join_op not in {"stack", "concat"}:
            raise ValueError("join op must be 'stack' or 'concat'")
        self._join_op = join_op

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
            self._base_array[self._padding - t : -self._padding - t]
            for t in reversed(range(self._stack_depth))
        ]
        array = np.stack(slices, axis=-self._frame_ndims - 1)

        # handle past frames right after a reset
        # iterate reversed so that more recent resets take precedence
        for t in reversed(range(1, self._stack_depth)):
            # get whether this env was done exactly t time steps ago
            done_t_steps_ago = self._done[-t : self._done.last + 1 - t]
            # convert from RotatingArray to ndarray, which can be used to index
            done_t_steps_ago = np.asarray(done_t_steps_ago)
            if self._reset_mode == "zero":
                # zero out all but the last t frames
                reset_value = 0
            else:  # self._reset_mode == "repeat"
                # repeat reset observation (same indices as for self._done
                # but incremented by 1)
                reset_value = self._base_array[self._padding - t + 1: -self._padding - t + 1]
                # index with same mask as used to index observation array below
                # add a new axis which will broadcast across frame stack dimension
                reset_value = reset_value[done_t_steps_ago, np.newaxis]
            array[done_t_steps_ago, :-t] = reset_value

        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array
