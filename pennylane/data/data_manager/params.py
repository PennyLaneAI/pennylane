# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Contains types and functions for dataset parameters.
"""


import enum
import typing
from functools import lru_cache
from typing import Any, Dict, FrozenSet, Iterable, List, Tuple, Union


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
        return isinstance(val, ParamArg) or (isinstance(val, str) and val in cls.values())

    def __str__(self) -> str:  # pylint: disable=invalid-str-returned
        return self.value


DEFAULT = ParamArg.DEFAULT
FULL = ParamArg.FULL

# Type for the name of a parameter, e.g 'molname', 'bondlength'
ParamName = str
# Type for a concrete paramter value, e.g 'H2', '0.5'
ParamVal = str


class Description(typing.Mapping[ParamName, ParamVal]):
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


# pylint:disable=too-many-branches
def format_param_args(param: ParamName, details: Any) -> Union[ParamArg, List[ParamVal]]:
    """Ensures each user-inputted parameter is a properly typed list.
    Also provides custom support for certain parameters."""
    if not isinstance(details, list):
        details = [details]

    for detail in details:
        if ParamArg.is_arg(detail):
            return ParamArg(detail)

    if param == "layout":
        # if a user inputs layout=[1,2], they wanted "1x2"
        # note that the above conversion to a list of details wouldn't work as expected here
        if all(isinstance(dim, int) for dim in details):
            return ["x".join(map(str, details))]
        # will turn [(1,2), [3,4], "5x6"] into ["1x2", "3x4", "5x6"]
        for i, detail in enumerate(details):
            if isinstance(detail, Iterable) and all(isinstance(dim, int) for dim in detail):
                details[i] = "x".join(map(str, detail))
            elif not isinstance(detail, str):
                raise TypeError(
                    f"Invalid layout value of '{detail}'. Must be a string or a tuple of ints."
                )
    elif param == "bondlength":
        for i, detail in enumerate(details):
            if isinstance(detail, float):
                details[i] = str(detail)
            elif isinstance(detail, int):
                details[i] = f"{detail:.1f}"
            elif not isinstance(detail, str):
                raise TypeError(f"Invalid bondlength '{detail}'. Must be a string, int or float.")

    for detail in details:
        if not isinstance(detail, str):
            raise TypeError(f"Invalid type '{type(detail).__name__}' for parameter '{param}'")

    return details


def format_params(**params: Any) -> Dict[ParamName, Union[ParamArg, ParamVal]]:
    """Converts params to a dictionary whose keys are parameter names and
    whose values are single ``ParamaterArg`` objects or lists of parameter values."""
    return {
        param_name: format_param_args(param_name, param) for param_name, param in params.items()
    }
