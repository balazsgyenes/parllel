from typing import Union, Tuple

import numpy as np

class ViewAwareArray(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        obj.view_location = None
        return obj
    
    def __array_finalize__(self, obj):
        # called via explicit constructor
        if obj is None: return

        self.view_location = getattr(obj, "view_location", None)

    def set_view_location(self, location: Union[Union[int, slice, Ellipsis], Tuple[Union[int, slice, Ellipsis]]]):
        if not isinstance(location, Tuple):
            location = (location,)
        location = tuple(elem if elem is not Ellipsis else None for elem in location)
        if location is Ellipsis: location = None
        if self.view_location is None:
            self.view_location = location
        else:
            if isinstance(self.view_location, int):
                self.view_location = (self.view_location, 