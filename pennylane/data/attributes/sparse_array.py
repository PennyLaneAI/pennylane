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

from typing import Tuple, Type

import numpy as np
from scipy.sparse import csr_array

from pennylane.data.base.attribute import AttributeType
from pennylane.data.base.typing_util import ZarrGroup


class DatasetSparseArray(AttributeType[ZarrGroup, csr_array, csr_array]):
    """Attribute type for ``scipy.sparse.csr_array``."""

    type_id = "sparse_array"

    @classmethod
    def consumes_types(cls) -> Tuple[Type[csr_array], ...]:
        return (csr_array,)

    @classmethod
    def py_type(cls, value_type: Type[csr_array]) -> str:
        """The module path of ``csr_matrix`` is ``scipy.sparse._csr.csr_matrix``, which
        is private. This method returns the public path ``scipy.sparse.csr_matrix``
        instead."""

        return "scipy.sparse.csr_array"

    def zarr_to_value(self, bind: ZarrGroup) -> csr_array:
        return csr_array(
            (np.array(bind["data"]), np.array(bind["indices"]), np.array(bind["indptr"])),
            shape=tuple(bind["shape"]),
        )

    def value_to_zarr(self, bind_parent: ZarrGroup, key: str, value: csr_array) -> ZarrGroup:
        bind = bind_parent.create_group(key)

        bind["data"] = value.data
        bind["indices"] = value.indices
        bind["indptr"] = value.indptr
        bind["shape"] = value.shape

        return bind
