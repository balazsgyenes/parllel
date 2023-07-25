import numpy as np
import pytest

from parllel.arrays import Array


@pytest.fixture(scope="module")
def batch_shape():
    return (10, 4, 4)


@pytest.fixture(params=[np.float32, np.int32], scope="module")
def dtype(request):
    return request.param


@pytest.fixture(
    params=[
        "local",
        pytest.param("inherited", marks=pytest.mark.skip),
        "shared",
    ],
    scope="module",
)
def storage(request):
    return request.param


@pytest.fixture(
    params=[0, 1, 2],
    ids=["padding=0", "padding=1", "padding=2"],
    scope="module",
)
def padding(request):
    return request.param


@pytest.fixture(
    params=[None, 1, 2],
    ids=["default_size", "1X_size", "2X_size"],
    scope="module",
)
def full_size(request, batch_shape):
    if request.param is None:
        return None
    else:
        return request.param * batch_shape[0]


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
