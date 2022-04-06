from typing import NamedTuple


class BatchSpec(NamedTuple):
    T: int  # Number of time steps, >=1.
    B: int  # Number of separate trajectory segments (i.e. # env instances), >=1.

    @property
    def size(self):
        return self.T * self.B
