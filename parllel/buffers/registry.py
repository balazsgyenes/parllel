from functools import reduce
from typing import Optional, Sequence, Tuple

from .buffer import Buffer, Indices


class BufferRegistry:
    def __init__(self, buffers: Optional[Sequence[Buffer]] = None):
        self._registry = {}
        if buffers is not None:
            for buf in buffers:
                self.register_buffer(buf)
        
    def register_buffer(self, buffer: Buffer):
        if isinstance(buffer, tuple):
            self._registry[buffer.buffer_id] = buffer
            for element in buffer:
                self.register_buffer(element)
        else:
            if buffer is None:
                return
            self._registry[buffer.buffer_id] = buffer

    def reduce_buffer(self, buffer: Buffer):
        if buffer is None:
            return
        if buffer.buffer_id in self._registry:
            # for registered buffers, just send ID and index history
            return (buffer.buffer_id, buffer.index_history)
        else:
            # for unregistered buffers, resort to sending buffer over pipe
            return buffer

    def rebuild_buffer(self, reduction: Tuple[int, Tuple[Indices, ...]]):
        if reduction is None:
            return
        buffer_id, index_history = reduction
        try:
            base = self._registry[buffer_id]
        except KeyError as e:
            raise RuntimeError(f"Cannot rebuild unregistered buffer with id '{buffer_id}.") from e

        # apply each index in index_history to base in succession, and return result
        return reduce(lambda buf, index: buf[index], index_history, base)

    def close(self):
        for buffer in self._registry.values():
            if not isinstance(buffer, tuple):
                # leaf node
                buffer.close()
