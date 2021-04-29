
import numpy as np
from collections import namedtuple

from gym.spaces import Dict as GymDict, Space as GymSpace
from parllel.types.named_tuple import (NamedTupleType, dict_to_namedtuple,
    namedtuple_to_dict)
from .space import Space
from .composite import Composite


class GymSpaceWrapper(Space):
    """Wraps a gym space to match the rlpyt interface; most of
    the functionality is for automatically converting a GymDict (dictionary)
    space into an rlpyt Composite space (and converting between the two).  Use
    inside the initialization of the environment wrapper for a gym environment.
    """

    def __new__(cls, space, *args, **kwargs):
        if isinstance(space, GymDict):
            return CompositeWrapper(space=space, *args, **kwargs)
        else:
            return super().__new__(cls)

    def __init__(self,
        space: GymSpace,
        name: str,
        null_value: float = 0,
        classes: dict = {},  # TODO: are we keeping this?
        force_float32: bool = True,
    ):
        self._name = name
        self._null_value = null_value
        self._force_float32 = force_float32
        self.space = space
        self._dtype = np.float32 if (space.dtype == np.float64 and
            force_float32) else None

    def sample(self):
        """Returns a single sample in a numpy array using the the ``sample()``
        method of the underlying gym space."""
        sample = self.space.sample()
        # Force numpy array, might force float64->float32.
        sample = np.asanyarray(sample, dtype=self.dtype)
        return sample

    def null_value(self):
        """Similar to ``sample()`` but returning a null value."""
        null = self.sample()
        if self._null_value is not None:
            try:
                null[:] = self._null_value
            except IndexError:  # e.g. scalar.
                null.fill(self._null_value)
        else:
            null.fill(0)
        return null

    def convert(self, value):
        """Convert from gym to rlpyt."""
        return np.asanyarray(value, dtype=self.dtype)

    def revert(self, value):
        """Revert from rlpyt to gym."""
        return value

    @property
    def dtype(self):
        return self._dtype or self.space.dtype

    @property
    def shape(self):
        return self.space.shape

    def contains(self, x):
        return self.space.contains(x)

    def __repr__(self):
        return self.space.__repr__()

    def __eq__(self, other):
        return self.space.__eq__(other)

    @property
    def low(self):
        return self.space.low

    @property
    def high(self):
        return self.space.high

    @property
    def n(self):
        return self.space.n

    def seed(self, seed=None):
        return self.space.seed(seed=seed)


class CompositeWrapper(Composite):
    def __init__(self,
        space: GymDict,
        name: str,
        null_value: float = 0,
        classes: dict = {},
        force_float32: bool = True,
    ):
        """Input ``name`` is used to disambiguate different gym spaces being
        wrapped, which is necessary if more than one GymDict space is to be
        wrapped in the same file.  The reason is that the associated
        namedtuples might be defined in the globals of this file, and then
        they must have distinct names.
        """
        self._name = name
        self._force_float32 = force_float32

        # TODO: refactor this
        # store all NamedTupleCls in a local variable
        self._classes = classes

        NamedTupleCls = self._classes.get(name)
        # "." is not allowed if storing in globals
        info_keys = [str(k).replace(".", "_") for k in space.spaces.keys()]
        if NamedTupleCls is None:
            NamedTupleCls = NamedTupleType(name, info_keys)
            self._classes[name] = NamedTupleCls
        elif not (isinstance(NamedTupleCls, NamedTupleType) and
                sorted(NamedTupleCls._fields) == sorted(info_keys)):
            raise ValueError(f"Name clash in classes dict: {name}.")
        
        spaces = [GymSpaceWrapper(
            space=v,
            name="_".join([name, k]),
            null_value=null_value,
            classes=self._classes,
            force_float32=force_float32)
            for k, v in space.spaces.items()]
        
        super().__init__(spaces, NamedTupleCls)

    def convert(self, value):
        """For dictionary space, use to convert wrapped env's dict to rlpyt
        namedtuple, i.e. inside the environment wrapper's ``step()``, for
        observation output to the rlpyt sampler (see helper function in
        file)"""
        return dict_to_namedtuple(value, self._name, self._classes, self._force_float32, force_ndarray=True)

    def revert(self, value):
        """For dictionary space, use to revert namedtuple action into wrapped
        env's dict, i.e. inside the environment wrappers ``step()``, for input
        to the underlying gym environment (see helper function in file)."""
        raise namedtuple_to_dict(value)

    def seed(self, seed=None):
        return [space.seed(seed=seed) for space in self._spaces]
