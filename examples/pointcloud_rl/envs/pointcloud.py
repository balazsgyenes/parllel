from typing import Optional, Union

from gym import Space
import numpy as np

import parllel.logger as logger


class PointCloud(Space):
    def __init__(self,
        max_num_points: int,
        low: Union[float, np.ndarray],
        high: Union[float, np.ndarray],
        shape: tuple[int, ...],
        dtype: np.dtype,
        seed: Optional[int] = None,
    ):
        assert dtype is not None, "dtype must be explicitly provided. "
        self.dtype = np.dtype(dtype)

        # determine shape if it isn't provided directly
        if shape is not None:
            shape = tuple(shape)
            assert (
                np.isscalar(low) or low.shape == shape
            ), "low.shape doesn't match provided shape"
            assert (
                np.isscalar(high) or high.shape == shape
            ), "high.shape doesn't match provided shape"
        elif not np.isscalar(low):
            shape = low.shape
            assert (
                np.isscalar(high) or high.shape == shape
            ), "high.shape doesn't match low.shape"
        elif not np.isscalar(high):
            shape = high.shape
            assert (
                np.isscalar(low) or low.shape == shape
            ), "low.shape doesn't match high.shape"
        else:
            raise ValueError(
                "shape must be provided or inferred from the shapes of low or high"
            )

        if np.isscalar(low):
            low = np.full(shape, low, dtype=dtype)

        if np.isscalar(high):
            high = np.full(shape, high, dtype=dtype)

        self._shape = shape
        self.low = low
        self.high = high

        def _get_precision(dtype):
            if np.issubdtype(dtype, np.floating):
                return np.finfo(dtype).precision
            else:
                return np.inf

        low_precision = _get_precision(self.low.dtype)
        high_precision = _get_precision(self.high.dtype)
        dtype_precision = _get_precision(self.dtype)
        if min(low_precision, high_precision) > dtype_precision:
            logger.warn(
                "Box bound precision lowered by casting to {}".format(self.dtype)
            )
        self.low = self.low.astype(self.dtype)
        self.high = self.high.astype(self.dtype)

        # Boolean arrays which indicate the interval type for each coordinate
        self.bounded_below = -np.inf < self.low
        self.bounded_above = np.inf > self.high

        self.max_num_points = max_num_points

        super().__init__(self.shape, self.dtype, seed)

    def sample(self):
        n_points = self.np_random.randint(self.max_num_points)

        sample_shape = (n_points,) + self.shape

        high = self.high if self.dtype.kind == "f" else self.high.astype("int64") + 1
        sample = np.empty(sample_shape)

        # Masking arrays which classify the coordinates according to interval
        # type
        unbounded = ~self.bounded_below & ~self.bounded_above
        upp_bounded = ~self.bounded_below & self.bounded_above
        low_bounded = self.bounded_below & ~self.bounded_above
        bounded = self.bounded_below & self.bounded_above

        sample[:, unbounded] = self.np_random.normal(
            size=(n_points,) + unbounded[unbounded].shape,
        )

        sample[:, low_bounded] = (
            self.np_random.exponential(
                size=(n_points,) + low_bounded[low_bounded].shape,
            )
            + self.low[low_bounded]
        )

        sample[:, upp_bounded] = (
            -self.np_random.exponential(
                size=(n_points,) + upp_bounded[upp_bounded].shape,
            )
            + self.high[upp_bounded]
        )

        sample[:, bounded] = self.np_random.uniform(
            low=self.low[bounded], high=high[bounded],
            size=(n_points,) + bounded[bounded].shape,
        )
        if self.dtype.kind == "i":
            sample = np.floor(sample)

        return sample.astype(self.dtype)
