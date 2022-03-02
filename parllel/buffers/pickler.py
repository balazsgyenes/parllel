from functools import reduce
import io
from multiprocessing.reduction import ForkingPickler
import pickle 
from typing import Dict

from . import Buffer


class BufferPickler(ForkingPickler):

    def __init__(self, *args, buffer_registry: Dict[int, Buffer], **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._buffer_registry: Dict[int, Buffer] = buffer_registry

    def persistent_id(self, obj):
        # Instead of pickling the Buffer as a regular class instance, we emit a
        # persistent ID.
        if isinstance(obj, Buffer):
            try:
                _ = self._buffer_registry[obj.buffer_id]
                # Here, our persistent ID is simply a tuple, containing a buffer ID
                # and an index history, allowing us to reconstruct the buffer
                return (obj.buffer_id, obj.index_history)
            except KeyError:
                # for unregistered buffers, fall back to default behaviour. this might
                # be expensive, but it should not happen often
                return None
        else:
            # If obj does not have a persistent ID, return None. This means obj
            # needs to be pickled as usual.
            return None


# def reduce_buffer(buffer: Buffer):
#     # this method needs to reduce the buffer but also pass a callable which
#     # will result in the unpickler registering the buffer
#     pass

# BufferPickler.register(Buffer, reduce_buffer)

class BufferUnpickler(pickle.Unpickler):

    def __init__(self, *args, buffer_registry: Dict[int, Buffer], **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._buffer_registry: Dict[int, Buffer] = buffer_registry

    def persistent_load(self, pid):
        # This method is invoked whenever a persistent ID is encountered.
        # Here, pid is the tuple returned by BufferPickler.
        buffer_id, index_history = pid

        try:
            base = self._buffer_registry[buffer_id]
        except KeyError as e:
            # Always raises an error if you cannot return the correct object.
            # Otherwise, the unpickler will think None is the object referenced
            # by the persistent ID.
            raise pickle.UnpicklingError(f"Cannot unpickle unregistered buffer with id '{buffer_id}.") from e

        # apply each index in index_history to base in succession, and return result
        return reduce(lambda buf, index: buf[index], index_history, base)


