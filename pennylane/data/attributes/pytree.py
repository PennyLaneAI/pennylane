from functools import lru_cache
from typing import Any, TypeVar

import numpy as np

from pennylane.data.base.attribute import DatasetAttribute
from pennylane.data.base.hdf5 import HDF5Group
from pennylane.data.base.mapper import AttributeTypeMapper
from pennylane.pytrees import flatten, list_pytree_types, serialization, unflatten

T = TypeVar("T")


class DatasetPyTree(DatasetAttribute[HDF5Group, T, T]):
    """Attribute type for an object that can be converted to
    a Pytree. This is the default serialization method for
    all Pennylane Pytrees, including sublcasses of ``Operator``.
    """

    type_id = "pytree"

    def hdf5_to_value(self, bind: HDF5Group) -> T:
        mapper = AttributeTypeMapper(bind)

        return unflatten(
            [mapper[str(i)].get_value() for i in range(bind["num_leaves"][()])],
            serialization.pytree_structure_load(bind["treedef"][()].tobytes()),
        )

    def value_to_hdf5(self, bind_parent: HDF5Group, key: str, value: T) -> HDF5Group:
        bind = bind_parent.create_group(key)
        mapper = AttributeTypeMapper(bind)

        leaves, treedef = flatten(value)

        bind["treedef"] = np.void(serialization.pytree_structure_dump(treedef, decode=False))
        bind["num_leaves"] = len(leaves)
        for i, leaf in enumerate(leaves):
            mapper[str(i)] = leaf

        return bind
