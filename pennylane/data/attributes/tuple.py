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
"""Contains an DatasetAttribute that allows for heterogeneous tuples of dataset
types."""

from typing import Generic, Type

from pennylane.data.base.attribute import DatasetAttribute
from pennylane.data.base.hdf5 import HDF5Group
from pennylane.data.base.mapper import AttributeTypeMapper
from pennylane.data.base.typing_util import T


class DatasetTuple(
    Generic[T],
    DatasetAttribute[HDF5Group, tuple[T], tuple[T]],
):
    """Type for tuples."""

    type_id = "tuple"

    @classmethod
    def consumes_types(cls) -> tuple[Type[tuple]]:
        return (tuple,)

    @classmethod
    def default_value(cls) -> tuple[()]:
        return tuple()

    def value_to_hdf5(self, bind_parent: HDF5Group, key: str, value: tuple[T]) -> HDF5Group:
        grp = bind_parent.create_group(key)

        mapper = AttributeTypeMapper(grp)
        for i, elem in enumerate(value):
            mapper[str(i)] = elem

        return grp

    def hdf5_to_value(self, bind: HDF5Group) -> tuple[T]:
        mapper = AttributeTypeMapper(bind)

        return tuple(mapper[str(i)].copy_value() for i in range(len(self.bind)))
