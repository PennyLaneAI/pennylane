import enum
from functools import lru_cache
from typing import Union, FrozenSet, Tuple
from collections.abc import Hashable
import typing
from pathlib import PurePosixPath


class ParamArg(enum.Enum):
    """Enum representing special args to ``load()``.

    FULL: used to request all attributes
    DEFAULT: used to request the default attribute
    """

    FULL = "full"
    DEFAULT = "default"

    @classmethod
    @lru_cache(maxsize=1)
    def values(cls) -> FrozenSet[str]:
        """Returns all values."""
        return frozenset(arg.value for arg in cls)

    @classmethod
    def is_arg(cls, val: Union["ParamArg", str]) -> bool:
        """Returns true if ``val`` is a ``ParamArg``, or one
        of its values."""
        return isinstance(val, ParamArg) or val in cls.values()

    def __str__(self) -> str:
        return self.value


DEFAULT = ParamArg.DEFAULT
FULL = ParamArg.FULL

# Type for the name of a parameter, e.g 'molname', 'bondlength'
ParamName = str
# Type for a concrete paramter value, e.g 'H2', '0.5'
ParamVal = str


class Description(typing.Mapping[ParamName, ParamVal], Hashable):
    """An immutable and hashable dictionary that contains all the parameter
    values for a dataset."""

    def __init__(self, params: typing.Iterable[Tuple[ParamName, ParamVal]]):
        self.__data = dict(params)
        self.__hash = None

    def __getitem__(self, __key: ParamName) -> ParamVal:
        return self.__data[__key]

    def __iter__(self) -> typing.Iterator[ParamName]:
        return iter(self.__data)

    def __len__(self) -> int:
        return len(self.__data)

    def __hash__(self) -> int:
        if not self.__hash:
            self.__hash = hash(tuple(self.__data))

        return self.__hash

    def __str__(self) -> str:
        return str(self.__data)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({repr(self.__data)})"
