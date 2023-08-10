import numpy as np
import pytest

from parllel.arrays import Array


@pytest.fixture(scope="session")
def batch_shape():
    return (10, 4, 4)


@pytest.fixture(params=[np.float32, np.int32], scope="session")
def dtype(request):
    return request.param


@pytest.fixture(
    params=[
        "local",
        "inherited",
        "shared",
    ],
    scope="session",
)
def storage(request):
    return request.param


@pytest.fixture(
    params=[0, 1, 2], ids=["padding=0", "padding=1", "padding=2"], scope="session"
)
def padding(request):
    return request.param


@pytest.fixture(
    params=[None, 10, 20], ids=["default_size", "1X_size", "2X_size"], scope="session"
)
def full_size(request):
    return request.param


@pytest.fixture
def blank_array(batch_shape, dtype, storage, padding, full_size):
    array = Array(
        batch_shape=batch_shape,
        dtype=dtype,
        storage=storage,
        padding=padding,
        full_size=full_size,
    )
    yield array
    array.close()


@pytest.fixture
def np_array(batch_shape, dtype):
    return np.arange(np.prod(batch_shape), dtype=dtype).reshape(batch_shape)


@pytest.fixture
def array(blank_array: Array, np_array: np.ndarray):
    blank_array[:] = np_array
    return blank_array
