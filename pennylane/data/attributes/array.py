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
"""Contains DatasetAttribute definition for numpy arrays."""

import numpy
from numpy.typing import ArrayLike

from pennylane.data.base.attribute import DatasetAttribute
from pennylane.data.base.hdf5 import HDF5Array, HDF5Group


class DatasetArray(DatasetAttribute[HDF5Array, numpy.ndarray, ArrayLike]):
    """
    Attribute type for objects that implement the Array protocol, including numpy arrays.
    """

    type_id = "array"

    def hdf5_to_value(self, bind: HDF5Array) -> numpy.ndarray:
        return numpy.array(self.bind, dtype=bind.dtype)

    def value_to_hdf5(self, bind_parent: HDF5Group, key: str, value: ArrayLike) -> HDF5Array:
        bind_parent[key] = value
        return bind_parent[key]
