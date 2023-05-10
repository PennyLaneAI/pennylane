from typing import Tuple, Type

import numpy as np

from pennylane import Hamiltonian
from pennylane.data.base.attribute import AttributeType
from pennylane.data.base.mapper import AttributeTypeMapper
from pennylane.data.base.typing_util import ZarrGroup


class DatasetsHamiltonian(AttributeType[ZarrGroup, Hamiltonian, Hamiltonian]):
    @classmethod
    def consumes_types(cls) -> Tuple[Type[Hamiltonian]]:
        return tuple

    def value_to_zarr(self, bind_parent: ZarrGroup, key: str, value: Hamiltonian) -> ZarrGroup:
        grp = bind_parent.create_group(key, overwrite=True)
        mapper = AttributeTypeMapper(grp)

        if value.id:
            mapper["id"] = value.id

        mapper["coeffs"] = np.array(value.coeffs, dtype=np.dtype(np.float64))
        mapper["observables"] = []

        for op in value.ops:
            try:
                mapper["observables"].append(op)
            except ValueError as exc:
                raise ValueError(
                    f"Serialization of Operator type {type(op).__name__} is not supported"
                ) from exc

        return grp

    def zarr_to_value(self, bind: ZarrGroup) -> Hamiltonian:
        mapper = AttributeTypeMapper(bind)
        return Hamiltonian(
            coeffs=mapper["coeffs"], observables=mapper["observables"], id=mapper.get("id")
        )
