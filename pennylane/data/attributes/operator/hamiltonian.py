from typing import Tuple, Type

from pennylane import Hamiltonian
from pennylane.data.attributes.operator.operator import DatasetOperatorList
from pennylane.data.base.attribute import AttributeType
from pennylane.data.base.typing_util import HDF5Group


class DatasetHamiltonian(AttributeType[HDF5Group, Hamiltonian, Hamiltonian]):
    """Attribute type that can serialize any ``pennylane.operation.Operator`` class."""

    type_id = "hamiltonian"

    @classmethod
    def consumes_types(cls) -> Tuple[Type[Hamiltonian]]:
        return (Hamiltonian,)

    def hdf5_to_value(self, bind: HDF5Group) -> Hamiltonian:
        ops = DatasetOperatorList(bind=bind["ops"]).get_value()
        coeffs = list(bind["coeffs"])

        return Hamiltonian(coeffs, ops)

    def value_to_hdf5(self, bind_parent: HDF5Group, key: str, value: Hamiltonian) -> HDF5Group:
        bind = bind_parent.create_group(key)

        coeffs, ops = value.terms()

        DatasetOperatorList(ops, parent_and_key=(bind, "ops"))
        bind["coeffs"] = coeffs

        return bind
