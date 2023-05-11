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
"""Contains an AttributeType that allows for heterogenous dictionaries
of Dataset attributes."""


import typing
from collections.abc import Mapping, MutableMapping
from typing import Generic, Optional, TypeVar, Union, Dict

from pennylane.data.base.attribute import AttributeType
from pennylane.data.base.mapper import MapperMixin
from pennylane.data.base.typing_util import T, ZarrAny, ZarrGroup


class DatasetDict(
    Generic[T],
    AttributeType[ZarrGroup, typing.Mapping[str, T], Optional[typing.Mapping[str, T]]],
    MutableMapping,
    MapperMixin,
):
    """Provides a dict-like collection for Dataset attribute types."""

    Self = TypeVar("Self", bound="DatasetDict")

    type_id = "dict"

    def __post_init__(self, value: Optional[typing.Mapping[str, T]], info):
        if value:
            self.update(value)

    def default_value(self) -> None:
        return None

    def zarr_to_value(self, bind: ZarrGroup) -> typing.MutableMapping[str, T]:
        return self

    def value_to_zarr(self, bind_parent: ZarrGroup, key: str, value: None) -> ZarrGroup:
        grp = bind_parent.create_group(key)

        return grp

    def copy_value(self) -> Dict[str, T]:
        return {key: attr.copy_value() for key, attr in self._mapper.items()}

    def copy(self) -> Dict[str, T]:
        return self.copy_value()

    def __getitem__(self, __key: str) -> T:
        return self._mapper[__key].get_value()

    def __setitem__(self, __key: str, __value: Union[T, AttributeType[ZarrAny, T, T]]) -> None:
        self._mapper[__key] = __value

    def __delitem__(self, __key: str) -> None:
        del self._mapper[__key]

    def __len__(self) -> int:
        return len(self.bind)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Mapping):
            return False

        if not len(self) == len(__value):
            return False

        return all(
            my_item == other_item for my_item, other_item in zip(self.items(), __value.items())
        )

    def __iter__(self) -> typing.Iterator[str]:
        return (key for key in self.bind.keys())

    def __str__(self) -> str:
        return str(dict(self))

    def __repr__(self):
        return repr(dict(self))
