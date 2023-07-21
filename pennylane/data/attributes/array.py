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

from pennylane.data.base.attribute import AttributeInfo, DatasetAttribute
from pennylane.data.base.hdf5 import HDF5Array, HDF5Group
from pennylane.math import array, get_interface
from pennylane.typing import TensorLike


class DatasetArray(DatasetAttribute[HDF5Array, numpy.ndarray, TensorLike]):
    """
    Attribute type for objects that implement the Array protocol, including numpy arrays
    and pennylane.math.tensor.
    """

    type_id = "array"

    def __post_init__(self, value: TensorLike) -> None:
        super().__post_init__(value)

        array_interface = get_interface(value)
        if array_interface not in ("numpy", "autograd"):
            raise TypeError(
                f"Expected a 'numpy.ndarray' or 'pennylane.numpy.tensor' array, got '{type(value).__name__}'"
            )

        self.info["array_interface"] = array_interface

        if array_interface == "autograd":
            self.info["requires_grad"] = value.requires_grad

    def hdf5_to_value(self, bind: HDF5Array) -> TensorLike:
        info = AttributeInfo(bind.attrs)

        interface = info.get("array_interface", "numpy")
        if info.get("requires_grad") is not None:
            return array(
                self.bind, dtype=bind.dtype, like=interface, requires_grad=info["requires_grad"]
            )

        return array(self.bind, dtype=bind.dtype, like=interface)

    def value_to_hdf5(self, bind_parent: HDF5Group, key: str, value: TensorLike) -> HDF5Array:
        bind_parent[key] = value

        return bind_parent[key]
