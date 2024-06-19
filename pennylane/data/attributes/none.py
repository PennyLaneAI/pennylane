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
"""Contains DatasetAttribute definition for None"""

from typing import Literal, Tuple, Type

from pennylane.data.base.attribute import DatasetAttribute
from pennylane.data.base.hdf5 import HDF5Array, HDF5Group


class DatasetNone(DatasetAttribute[HDF5Array, type(None), type(None)]):
    """Datasets type for 'None' values."""

    type_id = "none"

    @classmethod
    def default_value(cls) -> Literal[None]:
        return None

    @classmethod
    def consumes_types(cls) -> Tuple[Type[None]]:
        return (type(None),)

    def hdf5_to_value(self, bind) -> None:
        """Returns None."""
        return None

    def value_to_hdf5(self, bind_parent: HDF5Group, key: str, value: None) -> HDF5Array:
        """Creates an empty HDF5 array under 'key'."""
        return bind_parent.create_dataset(key, dtype="f")

    def __bool__(self) -> Literal[False]:
        return False
