from __future__ import annotations

from typing import Callable, Generic, Iterator, TypeVar

import numpy as np

from parllel import ArrayDict, ArrayLike, Location
import parllel.logger as logger
from parllel.types import BatchSpec


ArrayType = TypeVar("ArrayType", bound=ArrayLike)


class BatchedDataLoader(Generic[ArrayType]):
    """Iterates through a tree of samples in a fixed number of batches.
    Fields that cannot be indexed according to time (e.g.
    `initial_rnn_state`) are only indexed according to batch dimension.
    This data structure provides a convenient way to structure samples how the
    algo expects them, abstracting the shuffling and sampling operations. Using
    pre_batch_transform, the all samples can be moved to the GPU at once.
    """

    def __init__(
        self,
        tree: ArrayDict,
        sampler_batch_spec: BatchSpec,  # TODO: can this be inferred?
        n_batches: int,
        batch_only_fields: list[str] | None = None,
        recurrent: bool = False,
        shuffle: bool = True,
        drop_last: bool = True,
        pre_batches_transform: Callable | None = None,
        batch_transform: Callable | None = None,
    ) -> None:
        self.tree = self.source_tree = tree
        self.batch_only_fields = batch_only_fields
        # TODO: maybe renamed sampler_batch_spec to leading_dims and just take tuple
        self.sampler_batch_spec = sampler_batch_spec
        self.recurrent = recurrent
        self.shuffle = shuffle
        self.drop_last = drop_last

        # If recurrent, use whole trajectories, only shuffle B; else shuffle all.
        self.size = sampler_batch_spec.B if recurrent else sampler_batch_spec.size
        self.batch_size = self.size // n_batches

        if self.size % n_batches != 0:
            logger.warn(
                f"{self.size} {'trajectories' if self.recurrent else 'samples'} "
                f"cannot be split evenly into {n_batches} batches. The final "
                f"batch will be {'dropped' if drop_last else 'truncated'}. "
                "To avoid this, modify n_batches when creating the "
                "BatchedDataLoader. "
            )

        if pre_batches_transform is None:
            pre_batches_transform = lambda x: x
        self.pre_batches_transform = pre_batches_transform

        if batch_transform is None:
            batch_transform = lambda x: x
        self.batch_transform = batch_transform

        self.seed(None)

    def seed(self, seed: int | None) -> None:
        # TODO: replace with seeding module
        self.rng: np.random.Generator = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, location: Location) -> ArrayDict[ArrayType]:
        if not isinstance(location, tuple):
            location = (location,)

        tree = self.tree
        if self.batch_only_fields:
            tree = ArrayDict(tree)  # create a shallow copy to avoid modifying self.tree
            batch_only_elems = ArrayDict(
                {field: tree.pop(field) for field in self.batch_only_fields}
            )
        item = tree[location]
        if self.batch_only_fields:
            batch_only_item = batch_only_elems[location[1:]]
            item.update(batch_only_item)
        return item

    def batches(self) -> Iterator[ArrayDict[ArrayType]]:
        self.tree = self.pre_batches_transform(self.source_tree)

        all_indices = np.arange(self.size, dtype=np.int32)
        if self.shuffle:
            self.rng.shuffle(all_indices)
        # split the data into equally-sized pieces of `batch_size`
        if self.drop_last:
            starts = range(0, self.size - self.batch_size + 1, self.batch_size)
        else:
            starts = range(0, self.size, self.batch_size)
        for start in starts:
            # create slice for the nth batch
            # get the next batch of indices from the (shuffled) list of indices
            # if not dropping last, need to truncate final batch
            batch_indices = all_indices[start : min(self.size, start + self.batch_size)]
            if self.recurrent:
                # take entire trajectories, indexing only B dimension
                batch = self[:, batch_indices]
            else:
                # split indices into T and B indices using modulo and remainder
                # in this case, T and B dimensions will be flattened into one
                T_indices = batch_indices % self.sampler_batch_spec.T
                B_indices = batch_indices // self.sampler_batch_spec.T
                batch = self[T_indices, B_indices]

            batch = self.batch_transform(batch)
            yield batch

    def __iter__(self) -> Iterator[ArrayDict[ArrayType]]:
        # calling batches() explicitly is better, but we don't want Python's
        # default iterator behaviour to happen
        yield from self.batches()
