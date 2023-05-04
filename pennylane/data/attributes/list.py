# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains an AttributeType that allows for heterogenous lists of dataset
types."""

from collections.abc import Iterable, MutableSequence, Sequence
from typing import Generic, Union, overload

from pennylane.data.base.attribute import AttributeType
from pennylane.data.base.mapper import MapperMixin
from pennylane.data.base.typing_util import T, ZarrAny, ZarrGroup


class DatasetList(
    Generic[T], AttributeType[ZarrGroup, Iterable[T]], MutableSequence[T], MapperMixin
):
    type_id = "list"

    def default_value(self) -> list[T]:
        return []

    def zarr_to_value(self, bind: ZarrGroup) -> MutableSequence[T]:
        return self

    def value_to_zarr(self, bind_parent: ZarrGroup, key: str, value: Iterable[T]) -> ZarrGroup:
        grp = bind_parent.create_group(key)

        self.extend(value)

        return grp

    def insert(self, index: int, value: Union[T, AttributeType[ZarrAny, T]]):
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
        return repr(list(self))

    @overload
    def __getitem__(self, index: slice) -> list[T]:
        ...

    @overload
    def __getitem__(self, index: int) -> T:
        ...

    def __getitem__(self, index: Union[int, slice]):
        if isinstance(index, slice):
            return [self[i] for i in range(len(self))[index]]

        if index < 0:
            index = len(self) + index

        if index < 0 or index >= len(self):
            raise IndexError(index)

        return self._mapper[str(index)].get_value()

    def __setitem__(self, index: int, value: Union[T, AttributeType[ZarrAny, T]]):
        if index < 0:
            index = len(self) + index

        if index < 0 or index >= len(self):
            raise IndexError(index)

        self._mapper[str(index)] = value

    def __delitem__(self, index: int):
        del self._mapper[str(index)]
