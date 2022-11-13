from typing import Callable, Generic, Iterator, Optional, TypeVar

import numpy as np

from parllel.arrays import Array
from parllel.buffers import Buffer, NamedArrayTupleClass, NamedArrayTuple
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
        B_only_buffer: Optional[NamedArrayTuple] = None,
        recurrent: bool = False,
        shuffle: bool = True,
    ) -> None:
        self.buffer = self.source_buffer = buffer
        self.B_only_buffer = self.source_B_only = B_only_buffer
        # TODO: maybe renamed sampler_batch_spec to leading_dims and just take tuple
        self.sampler_batch_spec = sampler_batch_spec
        self.recurrent = recurrent
        self.shuffle  = shuffle

        # If recurrent, use whole trajectories, only shuffle B; else shuffle all.
        self.size = sampler_batch_spec.B if recurrent else sampler_batch_spec.size
        self.batch_size = self.size // n_batches

        if B_only_buffer is not None:
            # define new NamedArrayTupleClass for buffer and timeless buffer together
            self.BatchSamplesClass = NamedArrayTupleClass(
                typename = "BatchSamples",
                # TODO: B_only_buffer might not have _fields because it might not be NamedArrayTuple
                fields = buffer._fields + B_only_buffer._fields,
            )

    def seed(self, seed: int) -> None:
        # TODO: replace with seeding module
        self.rng: np.random.Generator = np.random.default_rng(seed)

    def apply_func(self, func: Callable) -> None:
        """Apply func to the wrapped buffer and store the result. Future
        samples are drawn from this stored result.
        """
        self.buffer = func(self.source_buffer)
        self.B_only_buffer = func(self.source_B_only)

    def __iter__(self) -> Iterator[BufferType]:
        if self.shuffle:
            all_indices = np.arange(self.size)
            self.rng.shuffle(all_indices)
        # split the data into equally-sized pieces of `batch_size`
        for start in range(0, self.size - self.batch_size + 1, self.batch_size):
            # create slice for the nth batch
            indices = slice(start, start + self.batch_size)
            if self.shuffle:
                # if shuffling, get shuffled indices
                indices = all_indices[indices]
            if self.recurrent:
                # take entire trajectories, indexing only B dimension
                samples = self.buffer[:, indices]
                if self.B_only_buffer is not None:
                    timeless_samples = self.B_only_buffer[indices]
                    samples = self.BatchSamplesClass(
                        **samples._asdict(), **timeless_samples._asdict())
                yield samples
            else:
                # split indices into T and B indices using modulo and remainder
                # in this case, T and B dimensions will be flattened into one
                # TODO: this op will fail if indices is a slice (shuffle == False)
                T_indices = indices % self.sampler_batch_spec.T
                B_indices = indices // self.sampler_batch_spec.T
                samples = self.buffer[T_indices, B_indices]
                if self.B_only_buffer is not None:
                    timeless_samples = self.B_only_buffer[B_indices]
                    samples = self.BatchSamplesClass(
                        **samples._asdict(), **timeless_samples._asdict())
                yield samples
