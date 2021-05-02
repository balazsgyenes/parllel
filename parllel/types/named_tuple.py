from collections import OrderedDict
from inspect import Signature as Sig, Parameter as Param
import string
from typing import Any

import numpy as np

RESERVED_NAMES = ("get", "items")


class NamedTupleType:
    """Instances of this class act like a type returned by namedtuple()."""

    def __init__(self, typename, fields):
        if not isinstance(typename, str):
            raise TypeError(f"type name must be string, not {type(typename)}")

        if isinstance(fields, str):
            spaces = any([whitespace in fields for whitespace in string.whitespace])
            commas = "," in fields
            if spaces and commas:
                raise ValueError(f"Single string fields={fields} cannot have both spaces and commas.")
            elif spaces:
                fields = fields.split()
            elif commas:
                fields = fields.split(",")
            else:
                # If there are neither spaces nor commas, then there is only one field.
                fields = (fields,)
        fields = tuple(fields)

        for field in fields:
            if not isinstance(field, str):
                raise ValueError(f"field names must be strings: {field}")
            if field.startswith("_"):
                raise ValueError(f"field names cannot start with an "
                                 f"underscore: {field}")
            if field in ("index", "count"):
                raise ValueError(f"can't name field 'index' or 'count'")
        self.__dict__["_typename"] = typename
        self.__dict__["_fields"] = fields
        self.__dict__["_signature"] = Sig(Param(field,
            Param.POSITIONAL_OR_KEYWORD) for field in fields)

    def __call__(self, *args, **kwargs):
        """Allows instances to act like `namedtuple` constructors."""
        args = self._signature.bind(*args, **kwargs).args  # Mimic signature.
        return self._make(args)

    def _make(self, iterable):
        """Allows instances to act like `namedtuple` constructors."""
        return NamedTuple(self._typename, self._fields, iterable)

    def __setattr__(self, name, value):
        """Make the type-like object immutable."""
        raise TypeError(f"can't set attributes of '{type(self).__name__}' "
                        "instance")

    def __repr__(self):
        return f"{type(self).__name__}({self._typename!r}, {self._fields!r})"


class NamedTuple(tuple):
    """
    Instances of this class act like instances of namedtuple types, but this
    same class is used for all namedtuple-like types created.  Unlike true
    namedtuples, this mock avoids defining a new class for each configuration
    of typename and field names.  Methods from namedtuple source are copied
    here.


    Implementation differences from `namedtuple`:

    * The individual fields don't show up in `dir(obj)`, but they do still
      show up as `hasattr(obj, field) => True`, because of `__getattr__()`.
    * These objects have a `__dict__` (by ommitting `__slots__ = ()`),
      intended to hold only the typename and list of field names, which are
      now instance attributes instead of class attributes.
    * Since `property(itemgetter(i))` only works on classes, `__getattr__()`
      is modified instead to look for field names.
    * Attempts to enforce call signatures are included, might not exactly
      match.
    """

    def __new__(cls, typename, fields, values):
        result = tuple.__new__(cls, values)
        if len(fields) != len(result):
            raise ValueError(f"Expected {len(fields)} arguments, got "
                             f"{len(result)}")
        result.__dict__["_typename"] = typename
        result.__dict__["_fields"] = fields
        return result

    def __getattr__(self, name):
        """Look in `_fields` when `name` is not in `dir(self)`."""
        try:
            return tuple.__getitem__(self, self._fields.index(name))
        except ValueError:
            raise AttributeError(f"'{self._typename}' object has no attribute "
                                 f"'{name}'")

    def __setattr__(self, name, value):
        """Make the object immutable, like a tuple."""
        raise AttributeError(f"can't set attributes of "
                             f"'{type(self).__name__}' instance")

    def _make(self, iterable):
        """Make a new object of same typename and fields from a sequence or
        iterable."""
        return type(self)(self._typename, self._fields, iterable)

    def _replace(self, **kwargs):
        """Return a new object of same typename and fields, replacing specified
        fields with new values."""
        result = self._make(map(kwargs.pop, self._fields, self))
        if kwargs:
            raise ValueError(f"Got unexpected field names: "
                             f"{str(list(kwargs))[1:-1]}")
        return result

    def _asdict(self):
        """Return an ordered dictionary mapping field names to their values."""
        return OrderedDict(zip(self._fields, self))

    def __getnewargs__(self):
        """Returns typename, fields, and values as plain tuple. Used by copy
        and pickle."""
        return self._typename, self._fields, tuple(self)

    def __repr__(self):
        """Return a nicely formatted string showing the typename."""
        return self._typename + '(' + ', '.join(f'{name}={value}'
            for name, value in zip(self._fields, self)) + ')'


class NamedArrayTupleType(NamedTupleType):
    """Instances of this class act like a type returned by rlpyt's
    namedarraytuple()."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name in self._fields:
            if name in RESERVED_NAMES:
                raise ValueError(f"Disallowed field name: '{name}'")

    def _make(self, iterable):
        return NamedArrayTuple(self._typename, self._fields, iterable)


class NamedArrayTuple(NamedTuple):

    def __getitem__(self, loc):
        """Return a new object of the same typename and fields containing the
        selected index or slice from each value."""
        try:
            return self._make(None if s is None else s[loc] for s in self)
        except IndexError as e:
            for j, s in enumerate(self):
                if s is None:
                    continue
                try:
                    _ = s[loc]
                except IndexError:
                    raise Exception(f"Occured in '{self._typename}' at field "
                                    f"'{self._fields[j]}'.") from e

    def __setitem__(self, loc, value):
        """
        If input value is the same named[array]tuple type, iterate through its
        fields and assign values into selected index or slice of corresponding
        value.  Else, assign whole of value to selected index or slice of
        all fields.  Ignore fields that are both None.

        # TODO: add handling of None fields
        """
        if not (isinstance(value, tuple) and  # Check for matching structure.
                getattr(value, "_fields", None) == self._fields):
            # Repeat value for each but respect any None.
            value = tuple(None if s is None else value for s in self)
        try:
            for j, (s, v) in enumerate(zip(self, value)):
                if s is not None or v is not None:
                    s[loc] = v
        except (ValueError, IndexError, TypeError) as e:
            raise Exception(f"Occured in {self.__class__} at field "
                            f"'{self._fields[j]}'.") from e

    def __contains__(self, key):
        """Checks presence of field name (unlike tuple; like dict)."""
        return key in self._fields

    def get(self, index):
        """Retrieve value as if indexing into regular tuple."""
        return tuple.__getitem__(self, index)

    def items(self):
        """Iterate ordered (field_name, value) pairs (like OrderedDict)."""
        for k, v in zip(self._fields, self):
            yield k, v


def NamedArrayTupleType_like(example):
    """Returns a NamedArrayTupleType instance  with the same name and fields
    as input, which can be a class or instance of namedtuple or
    namedarraytuple, or an instance of NamedTupleType, NamedTuple,
    NamedArrayTupleType, or NamedArrayTuple."""
    if isinstance(example, NamedArrayTupleType):
        return example
    elif isinstance(example, (NamedArrayTuple, NamedTuple, NamedTupleType)):
        return NamedArrayTupleType(example._typename, example._fields)
    else:
        raise TypeError("Input must be instance of NamedTuple[Type] or "
            f"NamedArrayTuple[Type]. Instead, got {type(example)}.")


def dict_to_namedtuple(value: Any, name: str, classes: dict, force_float32: bool = True):
    #TODO: this method is ridiculous. Take the expected NamedTupleType as input
    if isinstance(value, dict):
        NamedTupleCls = classes[name]
        # Disregard unrecognized keys:
        values = {k: dict_to_namedtuple(v, "_".join([name, k]), classes)
                  for k, v in value.items() if k in NamedTupleCls._fields}
        # Can catch some missing values (doesn't nest):
        values.update({k: 0 for k in NamedTupleCls._fields if k not in values})
        return NamedTupleCls(**values)
    elif isinstance(value, np.ndarray) and value.dtype == np.float64 and force_float32:
        return np.asanyarray(value, dtype=np.float32)
    else:
        return value


def namedtuple_to_dict(value):
    if isinstance(value, NamedTuple):
        return {k: namedtuple_to_dict(v) for k, v in zip(value._fields, value)}
    return value