import typing
from functools import cache
from typing import Dict, Generic, MutableMapping, Tuple, Type, TypeVar

import numpy as np

from pennylane import Hamiltonian
from pennylane.data.base.attribute import AttributeInfo, AttributeType
from pennylane.data.base.mapper import AttributeTypeMapper
from pennylane.data.base.typing_util import ZarrGroup
from pennylane.operation import Operator
from pennylane.ops.qubit.hamiltonian import Hamiltonian


@cache
def _get_all_operator_classes() -> Tuple[Type[Operator], ...]:
    acc = set()

    def rec(cls):
        for subcls in cls.__subclasses__():
            if subcls not in acc:
                acc.add(subcls)
                rec(subcls)

    rec(Operator)

    return tuple(acc)


@cache
def _op_name_to_class_dict() -> Dict[str, Type[Operator]]:
    op_classes = _get_all_operator_classes()

    return {op.__qualname__: op for op in op_classes}


Op = TypeVar("Op", bound=Operator)


class DatasetsHamiltonian(AttributeType[ZarrGroup, Hamiltonian, Hamiltonian]):
    @classmethod
    def consumes_types(cls) -> Tuple[Type[Hamiltonian]]:
        return (Hamiltonian,)

    def value_to_zarr(self, bind_parent: ZarrGroup, key: str, value: Hamiltonian) -> ZarrGroup:
        grp = bind_parent.create_group(key, overwrite=True)
        mapper = AttributeTypeMapper(grp)

        mapper["id"] = value.id
        mapper["coeffs"] = np.array(value.coeffs, dtype=np.dtype(np.float64))
        mapper["observables"] = []

        for op in value.ops:
            try:
                mapper["observables"].append(op)
            except ValueError as exc:
                raise ValueError(
                    f"Serialization of operator type {type(op).__name__} is not supported"
                ) from exc

        return grp

    def zarr_to_value(self, bind: ZarrGroup) -> Hamiltonian:
        mapper = AttributeTypeMapper(bind)
        return Hamiltonian(
            coeffs=mapper["coeffs"], observables=mapper["observables"], id=mapper["id"]
        )
