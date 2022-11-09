from typing import Iterator, Optional

import numpy as np
from numpy import random

from parllel.arrays import Array
from parllel.buffers import Buffer, Samples, buffer_asarray, NamedArrayTupleClass, NamedArrayTuple
from parllel.types import BatchSpec


class BatchedDataLoader:
    def __init__(self,
        buffer: NamedArrayTuple,
        timeless_buffer: NamedArrayTuple,
        sampler_batch_spec: BatchSpec, # TODO: can this be inferred?
        n_batches: int,
        recurrent: bool = False,
        shuffle: bool = True,
    ) -> None:
        self.buffer = buffer
        self.timeless_buffer = timeless_buffer
        self.sampler_batch_spec = sampler_batch_spec
        self.recurrent = recurrent
        self.shuffle  = shuffle

        self.size = sampler_batch_spec.B if recurrent else sampler_batch_spec.size
        self.batch_size = self.size // n_batches

        if timeless_buffer is not None:
            # define new NamedArrayTupleClass for buffer and timeless buffer together
            self.BatchSamplesClass = NamedArrayTupleClass(
                typename = "BatchSamples",
                fields = buffer._fields + timeless_buffer._fields,
            )

    def seed(self, seed: int) -> None:
        # TODO: replace with seeding module
        self.rng = np.random.default_rng(seed)

    def rotate(self) -> None:
        # TODO: is this needed?
        pass

    def to_device(self, device):
        # use this to move entire buffer to device at once
        # internally this copies to device and does further sampling ops on the copy
        # TODO: how to do this without depending on torch?
        # solution: implementation is invariant to buffer leaf type, so build
        # script passes buffer of torch tensors. add apply method (like torch Module)
        # that e.g. send every tensor to device
        pass

    def __iter__(self) -> Iterator[Buffer]:
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
                if self.timeless_buffer is not None:
                    timeless_samples = self.timeless_buffer[indices]
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
                if self.timeless_buffer is not None:
                    timeless_samples = self.timeless_buffer[B_indices]
                    samples = self.BatchSamplesClass(
                        **samples._asdict(), **timeless_samples._asdict())
                yield samples
