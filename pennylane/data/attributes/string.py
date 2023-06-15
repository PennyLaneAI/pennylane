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
"""Contains an AttributeType for str objects."""


from typing import Tuple, Type

from pennylane.data.base.attribute import AttributeType
from pennylane.data.base.hdf5 import HDF5Array, HDF5Group


class DatasetString(AttributeType[HDF5Array, str, str]):
    """Attribute type for strings."""

    type_id = "string"

    @classmethod
    def consumes_types(cls) -> Tuple[Type[str]]:
        return (str,)

    def hdf5_to_value(self, bind: HDF5Array) -> str:
        return bind.asstr()[()]

    def value_to_hdf5(self, bind_parent: HDF5Group, key: str, value: str) -> HDF5Array:
        if key in bind_parent:
            del bind_parent[key]

        bind_parent[key] = value

        return bind_parent[key]
