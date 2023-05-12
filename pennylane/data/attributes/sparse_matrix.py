from typing import Tuple, Type

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse._csr import csr_matrix

from pennylane.data.base.attribute import AttributeType
from pennylane.data.base.typing_util import ZarrGroup


class DatasetSparseMatrix(AttributeType[ZarrGroup, csr_matrix, csr_matrix]):
    type_id = "sparse_matrix"

    @classmethod
    def consumes_types(cls) -> Tuple[Type[csr_matrix], ...]:
        return (csr_matrix,)

    @classmethod
    def py_type(cls, value_type: Type[csr_matrix]) -> str:
        """The module path of ``csr_matrix`` is ``scipy.sparse._csr.csr_matrix``, which
        is private. This method returns the public path ``scipy.sparse.csr_matrix``
        instead."""

        return "scipy.sparse.csr_matrix"

    def zarr_to_value(self, bind: ZarrGroup) -> csr_matrix:
        return csr_matrix(
            (np.array(bind["data"]), np.array(bind["indices"]), np.array(bind["indptr"])),
            shape=tuple(bind["shape"]),
        )

    def value_to_zarr(self, bind_parent: ZarrGroup, key: str, value: csr_matrix) -> ZarrGroup:
        bind = bind_parent.create_group(key)

        bind["data"] = value.data
        bind["indices"] = value.indices
        bind["indptr"] = value.indptr
        bind["shape"] = value.shape

        return bind
