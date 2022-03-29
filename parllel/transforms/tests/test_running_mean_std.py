import pytest
import numpy as np
from numpy.random import default_rng

from parllel.transforms.running_mean_std import RunningMeanStd


@pytest.fixture
def rng():
    return default_rng(seed=42)

batch_size_and_n_batches = [
    (1, 100),
    (1, 1000),
    (10, 10),
    (10, 100),
    (10, 1000),
    (100, 1),
    (100, 2),
    (100, 5),
    (100, 10),
]

mus_and_std_devs = [
    (0., 0.1),
    (0., 1.0),
    (0., 10.),
    (0., 100.),
    (5., 1.),
    (5., 5.),
    (5., 10.),
    (100., 5.),
    (100., 10.),
    (100., 50.),
    (100., 100.),
    (100., 200.),
]


class TestRunningMeanStd:
    @pytest.mark.parametrize("mu, std_dev", mus_and_std_devs,
        ids=[f"mu={mu}-std_dev={std_dev}" for mu, std_dev in mus_and_std_devs])
    @pytest.mark.parametrize("batch_size, n_batches", batch_size_and_n_batches,
        ids=[f"batches={n}x{size}" for n, size in batch_size_and_n_batches])
    def test_correctness_single(self, batch_size, n_batches, mu, std_dev, rng):
        stats = RunningMeanStd(shape=(1,))
        all_data = np.zeros((batch_size*n_batches, 1), dtype=np.float64)
        for i in range(n_batches):
            batch = rng.normal(mu, std_dev, size=(batch_size, 1))
            stats.update(batch)
            all_data[i*batch_size:(i+1)*batch_size] = batch

        true_mean = np.mean(all_data, axis=0)
        true_var = np.var(all_data, axis=0)
        true_count = all_data.shape[0]

        assert np.allclose(true_mean, stats.mean)
        assert np.allclose(true_var, stats.var, rtol=1e-3)
        assert np.allclose(true_count, stats.count, atol=1e-2)
