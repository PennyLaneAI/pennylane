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
"""Contains a class for mapping Zarr groups to Dataset Attributes, and a mixin
class that provides the mapper class."""

from collections.abc import KeysView, Mapping
from types import MappingProxyType
from typing import Any, Union

from pennylane.data.base.attribute import (
    AttributeType,
    get_attribute_type,
    match_obj_type,
)
from pennylane.data.base.typing_util import ZarrAny, ZarrGroup


class AttributeTypeMapper:
    """
    This class performs the mapping between the objects contained
    in a Zarr group and Dataset attributes.
    """

    _cache: dict[str, AttributeType]

    def __init__(self, zroot: ZarrGroup) -> None:
        self._cache = {}
        self.zroot = zroot

    def __getitem__(self, key: str) -> AttributeType:
        if key in self._cache:
            return self._cache[key]

        zobj = self.zroot[key]

        attr_type = get_attribute_type(zobj)
        attr = attr_type(parent=self.zroot, key=key)
        self._cache[key] = attr

        return attr

    def __setitem__(self, key: str, value: Union[Any, AttributeType[ZarrAny, Any]]):
        if not isinstance(value, AttributeType):
            attr_type = match_obj_type(type(value))
            attr = attr_type(value, parent=self.zroot, key=key)
        else:
            value._set_parent(self.zroot, key)
            attr = value

        self._cache[key] = attr

    def move(self, from_: str, to: str):
        self.zroot.move(from_, to)
        del self._cache[from_]

    def view(self) -> Mapping[str, AttributeType]:
        return MappingProxyType(self)

    def __len__(self) -> int:
        return len(self.zroot)

    def keys(self) -> KeysView[str]:
        return self.zroot.keys()

    def __contains__(self, key: str) -> bool:
        return key in self._cache or key in self.zroot

    def __delitem__(self, key: str):
        self._cache.pop(key, None)

        del self.zroot[key]


class MapperMixin:
    """Mixin class for Dataset types that provide an interface
    to a Zarr group, e.g `DatasetList`, `DatasetDict`. Provides
    a `_mapper` property over the type's ``bind`` attribute."""

    bind: ZarrGroup

    __mapper: AttributeTypeMapper = None

    @property
    def _mapper(self) -> AttributeTypeMapper:
        if self.__mapper is None:
            self.__mapper = AttributeTypeMapper(self.bind)

        return self.__mapper
