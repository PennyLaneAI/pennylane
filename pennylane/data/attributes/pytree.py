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

import json
from typing import TypeVar

import numpy as np

from pennylane.core import queuing
from pennylane.data.attributes import DatasetArray, DatasetList, serialization
from pennylane.data.base.attribute import DatasetAttribute
from pennylane.data.base.hdf5 import HDF5Group
from pennylane.data.base.mapper import AttributeTypeMapper
from pennylane.math import get_interface
from pennylane.pytrees import flatten, unflatten
from pennylane.wires import Wires

T = TypeVar("T")


def _is_wires(obj) -> bool:
    """Whether ``obj`` should be treated as a single (opaque) leaf when flattening, i.e. a
    ``Wires`` object. This keeps wire labels out of the numeric leaves so they can be serialized
    through the JSON path (see :func:`~.value_to_hdf5`)."""
    return isinstance(obj, Wires)


class DatasetPyTree(DatasetAttribute[HDF5Group, T, T]):
    """Attribute type for an object that can be converted to
    a Pytree. This is the default serialization method for
    all PennyLane Pytrees, including subclasses of ``Operator``.
    """

    type_id = "pytree"

    def hdf5_to_value(self, bind: HDF5Group) -> T:
        with queuing.QueuingManager.stop_recording():
            structure = serialization.pytree_structure_load(bind["treedef"][()].tobytes())
            leaves = list(AttributeTypeMapper(bind)["leaves"].get_value())

            # HDF5 reads scalar leaves back as numpy scalars. Restore wire labels (the leaves that
            # live under a ``Wires`` node) to native Python scalars so that, e.g., an integer wire
            # ``0`` comes back as ``int`` rather than ``np.int64``. Parameters are left as numpy,
            # which is their expected representation.
            leaves = [
                _to_python_scalar(leaf) if is_wire else leaf
                for leaf, is_wire in zip(leaves, _wire_leaf_flags(structure), strict=True)
            ]

            return unflatten(leaves, structure)

    def value_to_hdf5(self, bind_parent: HDF5Group, key: str, value: T) -> HDF5Group:
        bind = bind_parent.create_group(key)
        leaves, treedef = flatten(value)

        bind["treedef"] = np.void(serialization.pytree_structure_dump(treedef, decode=False))

        if _storable_as_array(leaves):
            DatasetArray(leaves, parent_and_key=(bind, "leaves"))
        else:
            DatasetList(leaves, parent_and_key=(bind, "leaves"))

        return bind


def _to_python_scalar(leaf):
    """Convert a 0-d numpy leaf back to a native Python scalar, leaving other leaves untouched."""
    if isinstance(leaf, (np.generic, np.ndarray)) and np.ndim(leaf) == 0:
        return leaf.item()
    return leaf


def _wire_leaf_flags(structure) -> list:
    """Return, in leaf order, whether each leaf is a wire label (i.e. lives under a ``Wires`` node).

    The traversal order matches ``unflatten``'s depth-first consumption of the leaves, so the
    returned flags line up one-to-one with the stored leaves.
    """
    flags: list = []

    def _walk(node, in_wires: bool) -> None:
        if node.is_leaf:
            flags.append(in_wires)
            return
        in_wires = in_wires or node.type_ is Wires
        for child in node.children:
            _walk(child, in_wires)

    _walk(structure, False)
    return flags


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

    Also returns ``False`` when the leaves do not all share the same dtype ``kind``. Stacking
    leaves of different kinds into one array promotes them to a common dtype, which silently
    changes a leaf's type (e.g. an integer wire label ``0`` (kind ``"i"``) stored alongside a
    float parameter ``0.5`` (kind ``"f"``) comes back as ``0.0``). Comparing the ``kind`` rather
    than the full dtype still allows losslessly stackable leaves that only differ in width, such
    as byte strings of different lengths (``"S2"`` and ``"S3"``, both kind ``"S"``) or integers of
    different sizes, to use the efficient array path.
    """
    if any(get_interface(leaf) not in ("numpy", "autograd") for leaf in leaves):
        return False

    leaf_arrays = [np.asarray(leaf) for leaf in leaves]
    return (
        all(arr.dtype.kind in "biufcS" for arr in leaf_arrays)
        and (len({arr.shape for arr in leaf_arrays}) <= 1)
        and (len({arr.dtype.kind for arr in leaf_arrays}) <= 1)
    )
