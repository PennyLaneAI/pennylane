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
"""Contains AttributeType definition for ``qml.Hamiltonian``."""

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
