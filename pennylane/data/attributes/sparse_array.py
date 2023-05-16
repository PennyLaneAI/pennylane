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
"""Contains AttributeType definition for ``scipy.sparse.csr_array``."""

from typing import Generic, Tuple, Type, TypeVar, Union, cast

import numpy as np
from scipy.sparse import (
    bsr_array,
    coo_array,
    csc_array,
    csr_array,
    dia_array,
    dok_array,
    lil_array,
    csc_matrix,
    csr_matrix,
    bsr_matrix,
    coo_matrix,
    dia_matrix,
    dok_matrix,
    lil_matrix,
)

from pennylane.data.base.attribute import AttributeInfo, AttributeType
from pennylane.data.base.typing_util import ZarrGroup

_ALL_SPARSE = (
    bsr_array,
    coo_array,
    csc_array,
    csr_array,
    dia_array,
    dok_array,
    lil_array,
    csc_matrix,
    csr_matrix,
    bsr_matrix,
    coo_matrix,
    dia_matrix,
    dok_matrix,
    lil_matrix,
)
_ALL_SPARSE_MAP = {type_.__qualname__: type_ for type_ in _ALL_SPARSE}

SparseArray = Union[bsr_array, coo_array, csc_array, csr_array, dia_array, dok_array, lil_array]
SparseArrayT = TypeVar("SparseArrayT", bound=SparseArray)


class DatasetSparseArray(
    Generic[SparseArrayT], AttributeType[ZarrGroup, SparseArrayT, SparseArrayT]
):
    """Attribute type for Scipy sparse arrays. Can accept values of any type in
    ``scipy.sparse``. Arrays are stored in CSR format."""

    type_id = "sparse_array"

    def __post_init__(self, value: SparseArrayT, info) -> None:
        super().__post_init__(value, info)
        self.info["sparse_array_class"] = type(value).__qualname__

    @property
    def sparse_array_class(self) -> Type[SparseArrayT]:
        return cast(Type[SparseArrayT], _ALL_SPARSE_MAP[self.info["sparse_array_class"]])

    @classmethod
    def consumes_types(
        cls,
    ) -> Tuple[
        Type[bsr_array],
        Type[coo_array],
        Type[csc_array],
        Type[csr_array],
        Type[dia_array],
        Type[dok_array],
        Type[lil_array],
    ]:
        return _ALL_SPARSE

    @classmethod
    def py_type(cls, value_type: Type[SparseArray]) -> str:
        """The module path of sparse array types is private, e.g ``scipy.sparse._csr.csr_array``.
        This method returns the public path e.g ``scipy.sparse.csr_array`` instead."""

        return f"scipy.sparse.{value_type.__qualname__}"

    def zarr_to_value(self, bind: ZarrGroup) -> SparseArrayT:
        info = AttributeInfo(bind.attrs)

        value = csr_array(
            (np.array(bind["data"]), np.array(bind["indices"]), np.array(bind["indptr"])),
            shape=tuple(bind["shape"]),
        )

        sparse_array_class = cast(Type[SparseArrayT], _ALL_SPARSE_MAP[info["sparse_array_class"]])
        if not isinstance(value, sparse_array_class):
            value = sparse_array_class(value)

        return value

    def value_to_zarr(self, bind_parent: ZarrGroup, key: str, value: SparseArrayT) -> ZarrGroup:
        if not isinstance(value, csr_array):
            csr_value = csr_array(value)
        else:
            csr_value = value

        bind = bind_parent.create_group(key)

        bind["data"] = csr_value.data
        bind["indices"] = csr_value.indices
        bind["indptr"] = csr_value.indptr
        bind["shape"] = csr_value.shape

        return bind
