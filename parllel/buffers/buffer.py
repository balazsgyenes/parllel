from typing import Union

import numpy as np


class Buffer:
    """Abstracts memory management for large arrays.
    """
    def __init__(self, shared_memory: bool, stride: int = 0):
        self._shared_memory = shared_memory
        self._stride = stride

    def initialize(self):
        pass

    def __getitem__(self, location: Union[int, slice]):
        pass

    def __setitem__(self, location: Union[int, slice], value):
        pass

    def __array__(self, dtype = None):
        """Called to convert to numpy array.

        Docs: https://numpy.org/devdocs/user/basics.dispatch.html
        """
        pass
