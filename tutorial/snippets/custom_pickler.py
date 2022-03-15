import io
import multiprocessing as mp
from multiprocessing.context import BaseContext
import pickle
from typing import Dict


class TestObject:
    pass


def init_custom_pickler(ctx: BaseContext, data):
    
    PicklerCls = ctx.reducer.ForkingPickler
    
    class CustomUnpickler(pickle.Unpickler):

        _data = None

        def persistent_load(self, pid):
            # This method is invoked whenever a persistent ID is encountered.
            # Here, pid is the tuple returned by BufferPickler.
            _data = pid

            if _data == self._data:
                print(f"Unpickling with custom data: {self._data}")
                return TestObject()
            else:
                # Always raises an error if you cannot return the correct object.
                # Otherwise, the unpickler will think None is the object referenced
                # by the persistent ID.
                raise pickle.UnpicklingError(f"Incoming data {_data} did not match expected"
                    f"{self._data}")

    class CustomPickler(PicklerCls):

        _data = None

        def persistent_id(self, obj):
            # Instead of pickling the Buffer as a regular class instance, we emit a
            # persistent ID.
            if isinstance(obj, TestObject):
                # Here, our persistent ID is simply a tuple, containing a buffer ID
                # and an index history, allowing us to reconstruct the buffer
                print(f"Pickling with custom data: {self._data}")
                return self._data
            else:
                # If obj does not have a persistent ID, return None. This means obj
                # needs to be pickled as usual.
                return None

        @staticmethod
        def loads(s, /, *, fix_imports=True, encoding="ASCII", errors="strict",
                  buffers=None):
            if isinstance(s, str):
                raise TypeError("Can't load pickle from unicode string")
            file = io.BytesIO(s)
            return CustomUnpickler(file, fix_imports=fix_imports, buffers=buffers,
                            encoding=encoding, errors=errors).load()

    # could also do this within the class definition
    CustomPickler._data = data
    CustomUnpickler._data = data

    ctx.reducer.ForkingPickler = CustomPickler


def f(conn):
    x = conn.recv()
    print(f"Received {x} from pipe")


if __name__ == "__main__":
    # WARNING
    # modifying the pickler and unpickler only works with the fork start method
    # otherwise the child process has the default unpickler
    ctx1 = mp.get_context("fork")
    init_custom_pickler(ctx1, 42)

    ctx2 = mp.get_context("fork")
    init_custom_pickler(ctx2, 42*42)

    parent_conn1, child_conn1 = ctx1.Pipe()
    p1 = ctx1.Process(target=f, args=(child_conn1,))
    p1.start()
    parent_conn1.send(TestObject())
    p1.join()

    parent_conn2, child_conn2 = ctx2.Pipe()
    p2 = ctx2.Process(target=f, args=(child_conn2,))
    p2.start()
    parent_conn2.send(TestObject())
    p2.join()