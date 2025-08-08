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
"""Contains an DatasetAttribute that allows for heterogenous dictionaries
of Dataset attributes."""


from collections.abc import Iterator, Mapping, MutableMapping
from typing import Generic

from pennylane.data.base.attribute import DatasetAttribute
from pennylane.data.base.hdf5 import HDF5Any, HDF5Group
from pennylane.data.base.mapper import MapperMixin
from pennylane.data.base.typing_util import T


class DatasetDict(
    Generic[T],
    DatasetAttribute[HDF5Group, Mapping[str, T], Mapping[str, T]],
    MutableMapping[str, T],
    MapperMixin,
):
    """Provides a dict-like collection for Dataset attribute types. Keys must
    be strings."""

    type_id = "dict"

    def __post_init__(self, value: Mapping[str, T]):
        super().__post_init__(value)
        self.update(value)

    @classmethod
    def default_value(cls) -> dict:
        return {}

    def hdf5_to_value(self, bind: HDF5Group) -> MutableMapping[str, T]:
        return self

    def value_to_hdf5(self, bind_parent: HDF5Group, key: str, value: None) -> HDF5Group:
        grp = bind_parent.create_group(key)

        return grp

    def copy_value(self) -> dict[str, T]:
        return {key: attr.copy_value() for key, attr in self._mapper.items()}

    def copy(self) -> dict[str, T]:
        """Returns a copy of this mapping as a builtin ``dict``, with all
        elements copied."""
        return self.copy_value()

    def __getitem__(self, __key: str) -> T:
        self._check_key(__key)

        return self._mapper[__key].get_value()

    def __setitem__(self, __key: str, __value: T | DatasetAttribute[HDF5Any, T, T]) -> None:
        self._check_key(__key)

        if __key in self:
            del self[__key]

        self._mapper[__key] = __value

    def __delitem__(self, __key: str) -> None:
        self._check_key(__key)

        del self._mapper[__key]

    def __len__(self) -> int:
        return len(self.bind)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Mapping):
            return False

        if not len(self) == len(__value):
            return False

        if self.keys() != __value.keys():
            return False

        return all(__value[key] == self[key] for key in __value.keys())

    def __iter__(self) -> Iterator[str]:
        return (key for key in self.bind.keys())

    def __str__(self) -> str:
        return str(dict(self))

    def __repr__(self) -> str:
        return repr(dict(self))

    def _check_key(self, __key: str) -> None:
        """Checks that __key is a string, and raises a ``TypeError`` if it isn't."""
        if not isinstance(__key, str):
            raise TypeError(f"'{type(self).__name__}' keys must be strings.")
