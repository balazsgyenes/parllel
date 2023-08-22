import multiprocessing as mp

import numpy as np
import pytest

from parllel.arrays.storage import Storage


@pytest.fixture(params=["local", "shared"], scope="module")
def kind(request):
    return request.param


@pytest.fixture(scope="module")
def shape() -> tuple[int, ...]:
    return (16, 4, 2)


@pytest.fixture(params=[np.int32, np.float32], scope="module")
def dtype(request) -> np.dtype:
    return request.param


@pytest.fixture(scope="module")
def resizable() -> bool:
    return False


@pytest.fixture
def storage(
    kind: str,
    shape: tuple[int, ...],
    dtype: np.dtype,
    resizable: bool,
) -> Storage:
    return Storage(
        kind=kind,
        shape=shape,
        dtype=dtype,
        resizable=resizable,
    )


@pytest.fixture(params=["fork", "spawn", "forkserver"], scope="module")
def mp_ctx(request):
    method: str = request.param
    if method not in mp.get_all_start_methods():
        pytest.skip(f"'{method}' start method not supported on this platform.")
    return mp.get_context(method)


class TestCreation:
    pass


class TestReadWrite:
    pass


def write_to_piped_storage(pipe):
    while True:
        message = pipe.recv()
        if message == "stop":
            break
        storage, location, value = message
        with storage as ndarray:
            ndarray[location] = value


class TestPiping:
    @pytest.fixture(params=["shared"], scope="module")
    def kind(self, request):
        return request.param

    def test_enter_exit(self, storage, mp_ctx):
        location = (2, 1, 0)
        value = 42

        parent_pipe, child_pipe = mp_ctx.Pipe()
        p = mp_ctx.Process(target=write_to_piped_storage, args=(child_pipe,))
        p.start()
        for _ in range(200):
            parent_pipe.send((storage, location, value))
        parent_pipe.send("stop")
        p.join()

        with storage as ndarray:
            assert np.array_equal(ndarray[location], value)
