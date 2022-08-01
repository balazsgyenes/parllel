# allows postponed evaluation of annotations, i.e. type hints using a class
# inside the definition of that class
from __future__ import annotations
from collections import OrderedDict
from inspect import Signature, Parameter
from itertools import repeat
import string
from typing import Any, Dict, Iterable, NoReturn, Tuple, Union

from .buffer import Buffer


RESERVED_NAMES = ("get", "items")


class NamedTupleClass:
    """Instances of this class act like a type returned by namedtuple()."""

    def __init__(self, typename: str, fields: Union[str, Iterable[str]]):
        if not isinstance(typename, str):
            raise TypeError(f"type name must be string, not {type(typename)}")

        if isinstance(fields, str):
            spaces = any(whitespace in fields for whitespace in string.whitespace)
            commas = "," in fields
            if spaces and commas:
                raise ValueError(f"Single string fields={fields} cannot have both spaces and commas.")
            if spaces:
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
        self.__dict__["_signature"] = Signature(Parameter(field,
            Parameter.POSITIONAL_OR_KEYWORD) for field in fields)

    def __call__(self, *args: Any, **kwargs: Any) -> NamedTuple:
        """Allows instances to act like `namedtuple` constructors."""
        args = self._signature.bind(*args, **kwargs).args  # Mimic signature.
        return self._make(args)

    def _make(self, iterable: Iterable[Any]) -> NamedTuple:
        """Allows instances to act like `namedtuple` constructors."""
        return NamedTuple(self._typename, self._fields, iterable)

    def __setattr__(self, name: str, value: Any) -> NoReturn:
        """Make the type-like object immutable."""
        raise TypeError(f"can't set attributes of '{type(self).__name__}' "
                        "instance")

    def __repr__(self) -> str:
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
    * These objects have a `__dict__` (by omitting `__slots__ = ()`),
      intended to hold only the typename and list of field names, which are
      now instance attributes instead of class attributes.
    * Since `property(itemgetter(i))` only works on classes, `__getattr__()`
      is modified instead to look for field names.
    * Attempts to enforce call signatures are included, might not exactly
      match.
    """

    def __new__(cls, typename, fields: Tuple[str], values: Iterable[Any]):
        """Create a new instance, where each element in values has a
        corresponding name in fields. `fields` is not copied, but is a tuple,
        and therefore cannot be modified anyway.
        """
        result = tuple.__new__(cls, values)
        if len(fields) != len(result):
            raise ValueError(f"Expected {len(fields)} arguments, got "
                             f"{len(result)}")
        result.__dict__["_typename"] = typename
        result.__dict__["_fields"] = fields
        return result

    def __getattr__(self, name: str) -> Any:
        """Look in `_fields` when `name` is not in `dir(self)`."""
        try:
            return tuple.__getitem__(self, self._fields.index(name))
        except ValueError:
            raise AttributeError(f"'{self._typename}' object has no attribute "
                                 f"'{name}'")

    def __setattr__(self, name: str, value: Any) -> NoReturn:
        """Make the object immutable, like a tuple."""
        raise AttributeError(f"can't set attributes of "
                             f"'{type(self).__name__}' instance")

    def _make(self, iterable: Iterable[Any]) -> NamedTuple:
        """Make a new object of same typename and fields from a sequence or
        iterable."""
        return type(self)(self._typename, self._fields, iterable)

    def _replace(self, **kwargs: Any) -> NamedTuple:
        """Return a new object of same typename and fields, replacing specified
        fields with new values."""
        # kwargs.pop(key, default) returns the entry in kwargs each field, or
        # uses the value from this NamedTuple as a default
        result = self._make(map(kwargs.pop, self._fields, self))
        if kwargs:
            raise ValueError(f"Got unexpected field names: "
                             f"{str(list(kwargs))[1:-1]}")
        return result

    def _asdict(self) -> OrderedDict:
        """Return an ordered dictionary mapping field names to their values."""
        return OrderedDict(zip(self._fields, self))

    def __getnewargs__(self) -> Tuple[str, Tuple[str], Tuple[Any]]:
        """Returns typename, fields, and values as plain tuple. Used by copy
        and pickle."""
        return self._typename, self._fields, tuple(self)

    def __repr__(self) -> str:
        """Return a nicely formatted string showing the typename."""
        return self._typename + '(' + ', '.join(f'{name}={value}'
            for name, value in zip(self._fields, self)) + ')'


class NamedArrayTupleClass(NamedTupleClass):
    """Instances of this class act like a type returned by rlpyt's
    namedarraytuple()."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name in self._fields:
            if name in RESERVED_NAMES:
                raise ValueError(f"Disallowed field name: '{name}'")

    def _make(self, iterable: Iterable[Any]) -> NamedArrayTuple:
        return NamedArrayTuple(self._typename, self._fields, iterable)

    def __getitem__(self, loc: Any) -> None:
        # convenience method for type hints, allowing typing like 
        # "Samples[np.ndarray]"
        pass


class NamedArrayTuple(NamedTuple, Buffer):
    def __new__(cls, typename, fields: Tuple[str], values: Iterable[Any]):
        result = super().__new__(cls, typename, fields, values)
        result.__dict__["_index_history"] = []
        result.__dict__["_buffer_id"] = id(result)
        return result

    def __getitem__(self, loc: Any) -> NamedArrayTuple:
        """Return a new object of the same typename and fields containing the
        selected index or slice from each value."""
        try:
            # make new NamedTupleArray from the result of indexing each item using loc
            # do not try to index None items, just return them
            view = self._make(None if elem is None else elem[loc] for elem in self)
            view._index_history.extend(self._index_history)
            view._index_history.append(loc)
            view.__dict__["_buffer_id"] = self._buffer_id
            return view
        except IndexError as e:
            # repeat indexing to find where exception occurred
            for i, elem in enumerate(self):
                if elem is None:
                    continue
                try:
                    _ = elem[loc]
                except IndexError:
                    raise IndexError(f"Occured in '{self._typename}' at field "
                                    f"'{self._fields[i]}'.") from e

    def __setitem__(self, loc: Any, value: Any) -> None:
        """
        If input value is the same named[array]tuple type, iterate through its
        fields and assign values into selected index or slice of corresponding
        value.  Else, assign whole of value to selected index or slice of
        all fields.  Ignore fields where either current value or new value is
        None.
        """
        # if not a NamedTuple or NamedArrayTuple of matching structure
        if isinstance(value, dict):
            d = value
            # set up generator that orders dictionary items according to _fields
            value = (d[field] for field in self._fields)
        elif not (isinstance(value, NamedTuple)
            and getattr(value, "_fields", None) == self._fields):
            # Repeat value, assigning it to each field
            # e.g. tup[:] = 0
            value = repeat(value)
        try:
            for i, (elem, v) in enumerate(zip(self, value)):
                # do not set if either current value or new value is None
                if elem is not None and v is not None:
                    elem[loc] = v
        except (ValueError, IndexError, TypeError) as e:
            raise Exception(f"Occured in {type(self).__name__} at field "
                            f"'{self._fields[i]}'.") from e

    def __contains__(self, key: str) -> bool:
        """Checks presence of field name (unlike tuple; like dict)."""
        return key in self._fields

    def get(self, index: str) -> Any:
        """Retrieve value as if indexing into regular tuple."""
        return tuple.__getitem__(self, index)

    def items(self) -> Iterable[Tuple[str, Any]]:
        """Iterate ordered (field_name, value) pairs (like OrderedDict)."""
        return zip(self._fields, self)


def NamedArrayTupleClass_like(example: Union[NamedArrayTupleClass,
    NamedArrayTuple, NamedTupleClass, NamedTuple]) -> NamedArrayTupleClass:
    """Returns a NamedArrayTupleClass instance  with the same name and fields
    as input, which can be a class or instance of namedtuple or
    namedarraytuple, or an instance of NamedTupleClass, NamedTuple,
    NamedArrayTupleClass, or NamedArrayTuple."""
    if isinstance(example, NamedArrayTupleClass):
        return example
    if isinstance(example, (NamedArrayTuple, NamedTuple, NamedTupleClass)):
        return NamedArrayTupleClass(example._typename, example._fields)
    raise TypeError("Input must be instance of NamedTuple[Class] or "
            f"NamedArrayTuple[Class]. Instead, got {type(example)}.")


def dict_to_namedtuple(value: Dict, name: str):
    if isinstance(value, dict):
        values = tuple(dict_to_namedtuple(v, name = "_".join([name, k]))
                       for k, v in value.items())
        return NamedTupleClass(name, value.keys())(*values)

    return value


def namedtuple_to_dict(value):
    if isinstance(value, NamedTuple):
        return {k: namedtuple_to_dict(v) for k, v in zip(value._fields, value)}
    return value
