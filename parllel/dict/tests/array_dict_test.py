from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytest
import torch
from numpy import random

from parllel.dict.array_dict import ArrayDict


@dataclass
class Data:
    a: np.ndarray
    b: np.ndarray
    c: Optional[np.ndarray] = None
    d: Optional[np.ndarray] = None


# fmt: off
@pytest.fixture(scope="module")
def shape():
    return (20, 20, 20)

@pytest.fixture(scope="module")
def rng():
    return random.default_rng()

@pytest.fixture
def elem_a(shape):
    return np.arange(np.prod(shape)).reshape(shape)

@pytest.fixture
def elem_b(shape, rng: random.Generator):
    return rng.random(shape)

@pytest.fixture
def ndarraydict(elem_a, elem_b):
    return ArrayDict({
        "a": elem_a,
        "b": elem_b,
    })

@pytest.fixture
def ndarraydict(elem_a, elem_b):
    return ArrayDict({
        "a": elem_a,
        "b": elem_b,
    })

@pytest.fixture
def tensordict(elem_a, elem_b):
    return ArrayDict({
        "a": torch.from_numpy(elem_a),
        "b": torch.from_numpy(elem_b),
    })


class TestArrayDict:
    def test_getitem_int(self, ndarraydict, elem_a, elem_b):
        index = (2, 10)

        subdict = ndarraydict[index]
        subdict_cleaned = dict(subdict)

        assert np.array_equal(subdict_cleaned["a"], elem_a[index])
        assert np.array_equal(subdict_cleaned["b"], elem_b[index])

    def test_getitem_str(self, ndarraydict, elem_a, elem_b):
        assert np.array_equal(ndarraydict["a"], elem_a)
        assert np.array_equal(ndarraydict["b"], elem_b)

    def test_tensordict_to_device(self, tensordict):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        on_device = tensordict.to(device="cuda:0")

        for field, arr in on_device.items():
            assert arr.device == torch.device("cuda:0")
