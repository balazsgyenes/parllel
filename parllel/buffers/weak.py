from typing import Callable, List, Union
import multiprocessing as mp

import numpy as np

from .buffer import Buffer

class WeakBuffer:
    def __init__(self,
        write_callback: Callable[[int, Union[int, slice]], None],
        read_callback: Callable
    ) -> None:
        self._write_callback = write_callback
        self._read_callback = read_callback
        self._buffer = None

    def dispatch_write(self, target_buffer_id: int, location: Union[int, slice]):
        self._write_callback(target_buffer_id, location)
        # TODO: either wait here for lock to be acquired, or acquire lock in buffer

    def obtain(self):
        """Obtain numpy array through pipe for debugging.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        #TODO: just an example for now
        return "Weak reference to an array in another process."

    def __dir__(self) -> List[str]:
        # TODO: just an example for now
        return dir(self) + ["shape", "dtype"]