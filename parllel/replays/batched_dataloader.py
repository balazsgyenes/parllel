from typing import Callable, Generic, Iterator, List, Optional, TypeVar

import numpy as np

from parllel.buffers import Indices, NamedArrayTuple
import parllel.logger as logger
from parllel.types import BatchSpec


BufferType = TypeVar("BufferType")


class BatchedDataLoader(Generic[BufferType]):
    """Iterates through a batch of samples in n_batches chunks.

    There are two benefits to defining a new namedarraytuple for data loading.
    First, the original sample buffer cannot (in general) by sliced, because
    the `agent.initial_rnn_state` field has no time dimension. Second, sending
    the entire sample buffer to the GPU is relatively wasteful.
    """
    def __init__(self,
        buffer: NamedArrayTuple,
        sampler_batch_spec: BatchSpec, # TODO: can this be inferred?
        n_batches: int,
        batch_only_fields: Optional[List[str]] = None,
        recurrent: bool = False,
        shuffle: bool = True,
        drop_last: bool = True,
    ) -> None:
        self.buffer = self.source_buffer = buffer
        self.batch_only_fields = batch_only_fields
        # TODO: maybe renamed sampler_batch_spec to leading_dims and just take tuple
        self.sampler_batch_spec = sampler_batch_spec
        self.recurrent = recurrent
        self.shuffle  = shuffle
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

        self.seed(None)

    def seed(self, seed: Optional[int]) -> None:
        # TODO: replace with seeding module
        self.rng: np.random.Generator = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, location: Indices) -> BufferType:
        if not isinstance(location, tuple):
            location = (location, Ellipsis)
        if (loc_len := len(location)) < 2:
            location = location + (Ellipsis,) * (2 - loc_len)
        
        buffer = self.buffer
        if self.batch_only_fields:
            batch_only_elems = {field: getattr(buffer, field) for field in self.batch_only_fields}
            buffer = buffer._replace(**{field: None for field in self.batch_only_fields})
        item = buffer[location]
        if self.batch_only_fields:
            batch_only_elems = {field: element[location[1:]] for field, element in batch_only_elems.items()}
            item = item._replace(**batch_only_elems)
        return item

    def apply_func(self, func: Callable) -> None:
        """Apply func to the wrapped buffer and store the result. Future
        samples are drawn from this stored result.
        """
        self.buffer = func(self.source_buffer)

    def batches(self) -> Iterator[BufferType]:
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
                yield self[:, batch_indices]
            else:
                # split indices into T and B indices using modulo and remainder
                # in this case, T and B dimensions will be flattened into one
                T_indices = batch_indices % self.sampler_batch_spec.T
                B_indices = batch_indices // self.sampler_batch_spec.T
                yield self[T_indices, B_indices]

    def __iter__(self) -> Iterator[BufferType]:
        # calling batches() explicitly is better, but we don't want Python's
        # default iterator behaviour to happen
        yield from self.batches()
