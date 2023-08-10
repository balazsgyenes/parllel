import functools

import numpy as np
import numpy.random as random
import pytest

from parllel.arrays.jagged import JaggedArray

MEAN = 0.0
STD = 2.0


# fmt: off
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

@pytest.fixture(params=[None, 128], ids=["default_size", "full_size=2X"], scope="module")
def full_size(request):
    return request.param

@pytest.fixture
def blank_array(max_points, feature_shape, dtype, batch_shape, storage, padding, full_size):
    array = JaggedArray(
        batch_shape=batch_shape,
        dtype=dtype,
        max_mean_num_elem=max_points,
        feature_shape=feature_shape,
        storage=storage,
        padding=padding,
        full_size=full_size,
    )
    yield array
    array.close()

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
        if blank_array.full_size > blank_array.shape[0]:
            pytest.skip("Slice access is not supported for JaggedArrayList")
        
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
        if blank_array.padding == 0 and (blank_array.full_size == blank_array.shape[0]):
            pytest.skip("Rotate has no effect in this case.")

        graphs = [graph_generator() for _ in range(2)]

        blank_array[blank_array.last, 0] = graphs[0]
        blank_array[blank_array.last + 1, 0] = graphs[1]

        blank_array.rotate()

        assert np.array_equal(blank_array[-1, 0], graphs[0])
        assert np.array_equal(blank_array[0, 0], graphs[1])

    def test_rotate_full_size(self, blank_array: JaggedArray, graph_generator):
        if (blank_array.full_size == blank_array.shape[0]):
            pytest.skip()

        if blank_array.padding > 0:
            pytest.skip("Rotating might destroy some elements in this case.")

        graphs = [graph_generator() for _ in range(4)]

        blank_array[0, 0] = graphs[0]
        blank_array[blank_array.last, 0] = graphs[1]
        blank_array[blank_array.last + 1, 0] = graphs[2]

        blank_array.rotate()

        assert np.array_equal(blank_array[-1, 0], graphs[1])
        assert np.array_equal(blank_array[0, 0], graphs[2])
        blank_array[blank_array.last, 0] = graphs[3]

        blank_array.rotate()

        assert np.array_equal(blank_array[0, 0], graphs[0])
        assert np.array_equal(blank_array[blank_array.last, 0], graphs[1])
        assert np.array_equal(blank_array[blank_array.last + 1, 0], graphs[2])
        assert np.array_equal(blank_array[blank_array.last * 2 + 1, 0], graphs[3])
