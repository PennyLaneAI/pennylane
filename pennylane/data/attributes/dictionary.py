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


from collections.abc import Iterator, Mapping, MutableMapping
from typing import Generic, TypeVar, Union, Optional

from pennylane.data.base.attribute import AttributeType, AttributeInfo
from pennylane.data.base.mapper import MapperMixin
from pennylane.data.base.typing_util import T, ZarrAny, ZarrGroup


class DatasetDict(
    Generic[T],
    AttributeType[ZarrGroup, Mapping[str, T]],
    MutableMapping[str, T],
    MapperMixin,
):
    Self = TypeVar("Self", bound="DatasetDict")

    type_id = "dict"

    def __init__(
        self,
        value: Optional[Mapping[str, T]] = None,
        info: Optional[AttributeInfo] = None,
        *,
        bind: Optional[ZarrGroup] = None,
        parent_and_key: Optional[tuple[ZarrGroup, str]] = None
    ) -> None:
        super().__init__(value, info, bind=bind, parent_and_key=parent_and_key)

        if value:
            self.update(value)

    def default_value(self) -> Mapping[str, T]:
        return {}

    def zarr_to_value(self, bind: ZarrGroup) -> MutableMapping[str, T]:
        return self

    def value_to_zarr(self, bind_parent: ZarrGroup, key: str, value: Mapping[str, T]) -> ZarrGroup:
        grp = bind_parent.create_group(key)

        return grp

    def __getitem__(self, __key: str) -> T:
        return self._mapper[__key].get_value()

    def __setitem__(self, __key: str, __value: Union[T, AttributeType[ZarrAny, T]]) -> None:
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

    def __iter__(self) -> Iterator[str]:
        return (key for key in self.bind.keys())

    def __str__(self) -> str:
        return str(dict(self))

    def __repr__(self):
        return repr(dict(self))
