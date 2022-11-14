from typing import Callable, Generic, Iterator, List, Optional, TypeVar

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
        B_only_fields: Optional[List[str]] = None,
        recurrent: bool = False,
        shuffle: bool = True,
    ) -> None:
        self.buffer = self.source_buffer = buffer
        self.B_only_fields = B_only_fields
        # TODO: maybe renamed sampler_batch_spec to leading_dims and just take tuple
        self.sampler_batch_spec = sampler_batch_spec
        self.recurrent = recurrent
        self.shuffle  = shuffle

        # If recurrent, use whole trajectories, only shuffle B; else shuffle all.
        self.size = sampler_batch_spec.B if recurrent else sampler_batch_spec.size
        self.batch_size = self.size // n_batches

        self.seed(None)

    def seed(self, seed: Optional[int]) -> None:
        # TODO: replace with seeding module
        self.rng: np.random.Generator = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, location) -> BufferType:
        # TODO: make this method more robust, e.g. if location is not tuple
        buffer = self.buffer
        if self.B_only_fields:
            # TODO: add method _pop to namedtuple to do this
            B_only_elements = {field: getattr(buffer, field) for field in self.B_only_fields}
            buffer = buffer._replace({field: None for field in self.B_only_fields})
        item = buffer[location]
        if self.B_only_fields:
            B_only_elements = {field: element[location[1:]] for field, element in B_only_elements.items()}
            buffer = buffer._replace({field: None for field in self.B_only_fields})
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
        for start in range(0, self.size - self.batch_size + 1, self.batch_size):
            buffer = self.buffer
            # create slice for the nth batch
            # index the shuffled list of all indices to get a batch of shuffled indices
            batch_indices = all_indices[start : start + self.batch_size]
            if self.recurrent:
                # take entire trajectories, indexing only B dimension
                # TODO: call self.__getitem__ for this
                if self.B_only_fields:
                    B_only_elements = {field: getattr(buffer, field) for field in self.B_only_fields}
                    buffer = buffer._replace({field: None for field in self.B_only_fields})
                samples = buffer[:, batch_indices]
                if self.B_only_fields:
                    B_only_elements = {field: element[batch_indices] for field, element in B_only_elements.items()}
                    samples = samples._replace(**B_only_elements)
                yield samples
            else:
                # split indices into T and B indices using modulo and remainder
                # in this case, T and B dimensions will be flattened into one
                T_indices = batch_indices % self.sampler_batch_spec.T
                B_indices = batch_indices // self.sampler_batch_spec.T
                if self.B_only_fields:
                    B_only_elements = {field: getattr(buffer, field) for field in self.B_only_fields}
                    buffer = buffer._replace({field: None for field in self.B_only_fields})
                samples = buffer[T_indices, B_indices]
                if self.B_only_fields:
                    B_only_elements = {field: element[B_indices] for field, element in B_only_elements.items()}
                    buffer = buffer._replace({field: None for field in self.B_only_fields})
                yield samples
