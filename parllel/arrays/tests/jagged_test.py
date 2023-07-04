import functools

import numpy as np
import numpy.random as random
import pytest

from parllel.arrays import JaggedArray

MEAN = 0.0
STD = 2.0


# fmt: off
@pytest.fixture(params=[
    JaggedArray,
], scope="module")
def ArrayClass(request):
    return request.param

@pytest.fixture(scope="module")
def max_points():
    return 1000

@pytest.fixture(scope="module")
def feature_shape():
    return (2, 3)

@pytest.fixture(scope="module")
def batch_shape():
    return (64, 8)

@pytest.fixture(params=[np.float32], scope="module")
def dtype(request):
    return request.param

@pytest.fixture(params=[
    "local",
    ], scope="module")
def storage(request):
    return request.param

@pytest.fixture(params=[0], ids=["padding=0"], scope="module")
def padding(request):
    return request.param

@pytest.fixture(params=[None], ids=["default_size"], scope="module")
def full_size(request):
    return request.param

@pytest.fixture
def blank_array(ArrayClass, max_points, feature_shape, dtype, batch_shape, storage, padding, full_size):
    array = ArrayClass(
        shape=(max_points,) + feature_shape,
        dtype=dtype,
        batch_shape=batch_shape,
        storage=storage,
        padding=padding,
        full_size=full_size,
    )
    yield array
    array.close()
    array.destroy()

@pytest.fixture(scope="module")
def rng():
    return random.default_rng()

@pytest.fixture(scope="module")
def graph_generator(rng, max_points, feature_shape, dtype):
    return functools.partial(random_graph,
        rng=rng,
        max_num_points=max_points,
        feature_shape=feature_shape,
        dtype=dtype,
    )

def random_graph(
    rng: random.Generator,
    max_num_points: int,
    feature_shape: tuple[int, ...],
    dtype: np.dtype,
    mean: float = MEAN,
    std: float = STD,
) -> np.ndarray:
    n_points = rng.integers(max_num_points)
    return rng.normal(loc=mean, scale=std, size=(n_points,) + feature_shape).astype(dtype)


class TestJaggedArray:
    def test_write_single_graph(self, blank_array, graph_generator):
        graph = graph_generator()
        loc = (0, 4)
        blank_array[loc] = graph
        assert np.array_equal(blank_array[loc], graph)

    def test_write_consecutive_graphs(self, blank_array, graph_generator):
        graph1 = graph_generator()
        graph2 = graph_generator()

        blank_array[0, 3] = graph1
        blank_array[1, 3] = graph2

        assert np.array_equal(blank_array[0, 3], graph1)
        assert np.array_equal(blank_array[1, 3], graph2)
        assert np.array_equal(blank_array[0:2, 3], np.concatenate((graph1, graph2)))

    def test_write_parallel_graphs(self, blank_array, graph_generator):
        graph1 = graph_generator()
        graph2 = graph_generator()

        blank_array[0, 1] = graph1
        blank_array[0, 2] = graph2

        assert np.array_equal(blank_array[0, 1], graph1)
        assert np.array_equal(blank_array[0, 2], graph2)
        assert np.array_equal(blank_array[0, 1:3], np.concatenate((graph1, graph2)))

    def test_get_array_indices(self, blank_array, graph_generator, rng):
        graphs = [graph_generator() for _ in range(4)]

        blank_array[0, 0] = graphs[0]
        blank_array[0, 1] = graphs[1]
        blank_array[1, 0] = graphs[2]
        blank_array[1, 1] = graphs[3]

        t_locs = rng.integers(0, 2, size=(10,))
        b_locs = rng.integers(0, 2, size=(10,))

        batch = blank_array[t_locs, b_locs]

        items = []
        for t_loc, b_loc in zip(t_locs, b_locs):
            items.append(graphs[t_loc * 2 + b_loc])

        np_batch = np.concatenate(items)

        assert np.array_equal(np_batch, batch)

    def test_rotate(self, blank_array: JaggedArray, graph_generator):
        batch_shape, shape = blank_array.shape[:2], blank_array.shape[2:]
        array = JaggedArray(
            shape=shape,
            dtype=blank_array.dtype,
            batch_shape=batch_shape,
            padding=1,
        )

        graphs = [graph_generator() for _ in range(2)]

        array[array.last, 0] = graphs[0]
        array[array.last + 1, 0] = graphs[1]

        array.rotate()

        assert np.array_equal(array[-1, 0], graphs[0])
        assert np.array_equal(array[0, 0], graphs[1])