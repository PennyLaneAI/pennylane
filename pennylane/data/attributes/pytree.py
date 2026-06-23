# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains DatasetAttribute definition for PyTree types."""

from typing import TypeVar

import numpy as np

from pennylane import queuing
from pennylane.data.attributes import DatasetArray, DatasetList, serialization
from pennylane.data.base.attribute import DatasetAttribute
from pennylane.data.base.hdf5 import HDF5Group
from pennylane.data.base.mapper import AttributeTypeMapper
from pennylane.math import get_interface
from pennylane.pytrees import flatten, unflatten

T = TypeVar("T")


class DatasetPyTree(DatasetAttribute[HDF5Group, T, T]):
    """Attribute type for an object that can be converted to
    a Pytree. This is the default serialization method for
    all PennyLane Pytrees, including subclasses of ``Operator``.
    """

    type_id = "pytree"

    def hdf5_to_value(self, bind: HDF5Group) -> T:
        with queuing.QueuingManager.stop_recording():
            return unflatten(
                AttributeTypeMapper(bind)["leaves"].get_value(),
                serialization.pytree_structure_load(bind["treedef"][()].tobytes()),
            )

    def value_to_hdf5(self, bind_parent: HDF5Group, key: str, value: T) -> HDF5Group:
        bind = bind_parent.create_group(key)
        leaves, treedef = flatten(value)

        bind["treedef"] = np.void(serialization.pytree_structure_dump(treedef, decode=False))

        if _storable_as_array(leaves):
            DatasetArray(leaves, parent_and_key=(bind, "leaves"))
        else:
            DatasetList(leaves, parent_and_key=(bind, "leaves"))

        return bind


def _storable_as_array(leaves: list) -> bool:
    """Whether ``leaves`` can be stored as a single array, which is more efficient than a
    list; otherwise they are stored leaf-by-leaf as a list.

    Returns ``False`` for unicode-string leaves (e.g. string wire labels), which an HDF5 array
    reads back as ``bytes``; for ragged, mixed, or object leaves, which cannot be stacked into
    one array; and for non-numpy leaves such as ``torch``/``jax`` tensors, which ``DatasetArray``
    does not handle. That interface check also guards the ``np.asarray`` calls below, which would
    otherwise raise on e.g. a ``torch`` tensor that requires grad. Numeric and ``bytes`` leaves
    use the array path, as they did before string leaves were special-cased. ``"biufcS"`` are the
    numpy dtype ``kind`` codes that round-trip safely through an HDF5 array: boolean, signed and
    unsigned integer, float, complex, and byte string.
    """
    if any(get_interface(leaf) not in ("numpy", "autograd") for leaf in leaves):
        return False

    leaf_arrays = [np.asarray(leaf) for leaf in leaves]
    return all(arr.dtype.kind in "biufcS" for arr in leaf_arrays) and (
        len({arr.shape for arr in leaf_arrays}) <= 1
    )
