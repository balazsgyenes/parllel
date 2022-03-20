import pytest
import numpy as np

from parllel.transforms.running_mean_std import RunningMeanStd


class TestRunningMeanStd:
    def test_correctness_single(self):
        n = 1000
        batch_size = 10
        mu = 5.
        std_dev = 5.
        seed = 42

        stats = RunningMeanStd(shape=(1,))
        rng = np.random.default_rng(seed)
        batches = []
        for _ in range(n):
            batch = rng.normal(mu, std_dev, size=(batch_size, 1))
            stats.update(batch)
            batches.append(batch)

        all_data = np.concatenate(batches)
        true_mean = np.mean(all_data, axis=0)
        true_var = np.var(all_data, axis=0)
        true_count = all_data.shape[0]

        error_mean = true_mean - stats.mean
        error_var = true_var - stats.var
        error_count = true_count - stats.count

        assert np.abs(error_mean) < 1e-5
        assert np.abs(error_var) < 1e-5
        assert np.abs(error_count) <= 1e-2
