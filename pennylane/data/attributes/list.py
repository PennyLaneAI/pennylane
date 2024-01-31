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
"""Contains an DatasetAttribute that allows for heterogeneous lists of dataset
types."""

import typing
from collections.abc import Sequence
from typing import Generic, List, Union, overload

from pennylane.data.base.attribute import DatasetAttribute
from pennylane.data.base.hdf5 import HDF5Any, HDF5Group
from pennylane.data.base.mapper import MapperMixin
from pennylane.data.base.typing_util import T


class DatasetList(  # pylint: disable=too-many-ancestors
    Generic[T],
    DatasetAttribute[HDF5Group, typing.Sequence[T], typing.Iterable[T]],
    typing.MutableSequence[T],
    MapperMixin,
):
    """Provides a list-like collection type for Dataset Attributes."""

    type_id = "list"

    def __post_init__(self, value: typing.Iterable[T]):
        super().__post_init__(value)

        self.extend(value)

    @classmethod
    def default_value(cls) -> typing.Iterable[T]:
        return []

    def hdf5_to_value(self, bind: HDF5Group) -> typing.MutableSequence[T]:
        return self

    def value_to_hdf5(
        self, bind_parent: HDF5Group, key: str, value: typing.Iterable[T]
    ) -> HDF5Group:
        grp = bind_parent.create_group(key)

        return grp

    def copy_value(self) -> List[T]:
        return [self._mapper[str(i)].copy_value() for i in range(len(self))]

    def copy(self) -> List[T]:
        """Returns a copy of this list as a builtin ``list``, with all
        elements copied.."""
        return self.copy_value()

    def insert(self, index: int, value: Union[T, DatasetAttribute[HDF5Any, T, T]]):
        """Implements the insert() method."""
        if index < 0:
            index = len(self) + index

        if index < 0:
            index = 0
        elif index >= len(self):
            self._mapper[str(len(self))] = value
            return

        for i in reversed(range(index, len(self))):
            self._mapper.move(str(i), str(i + 1))

        self._mapper[str(index)] = value

    def __len__(self) -> int:
        return len(self.bind)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Sequence):
            return False

        if not len(self) == len(__value):
            return False

        return all(x == y for x, y in zip(self, __value))

    def __str__(self) -> str:
        return str(list(self))

    def __repr__(self) -> str:
        items_repr = ", ".join(repr(elem) for elem in self)
        return f"[{items_repr}]"

    @overload
    def __getitem__(self, index: slice) -> typing.List[T]:
        pass

    @overload
    def __getitem__(self, index: int) -> T:
        pass

    def __getitem__(self, index: Union[int, slice]):
        if isinstance(index, slice):
            return [self[i] for i in range(len(self))[index]]

        if index < 0:
            index = len(self) + index

        if not 0 <= index < len(self):
            raise IndexError(index)

        return self._mapper[str(index)].get_value()

    def __setitem__(self, index: int, value: Union[T, DatasetAttribute[HDF5Any, T, T]]):
        if index < 0:
            index = len(self) + index
        if not 0 <= index < len(self):
            raise IndexError("list assignment index out of range")

        key = str(index)
        if key in self._mapper:
            del self._mapper[key]

        self._mapper[key] = value

    def __delitem__(self, index: int):
        init_len = len(self)

        if index < 0:
            index = init_len + index
        if not 0 <= index < init_len:
            raise IndexError(index)

        del self._mapper[str(index)]

        # Move all the objects in front of the deleted object back one
        if index < init_len:
            for i in range(index, init_len - 1):
                self._mapper.move(str(i + 1), str(i))
