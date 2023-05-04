# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains AttributeType for a pennylane Hamiltonian."""

from collections.abc import Iterable

from pennylane import Hamiltonian
from pennylane.data.base.attribute import AttributeType
from pennylane.data.base.typing_util import ZarrGroup
from pennylane.pauli import pauli_word_to_string, string_to_pauli_word


class DatasetHamiltonian(AttributeType[ZarrGroup, Hamiltonian]):
    type_id = "hamiltonian"

    @classmethod
    def consumes_types(cls) -> Iterable[type]:
        return (Hamiltonian,)

    def zarr_to_value(self, bind: ZarrGroup) -> Hamiltonian:
        wire_map = {w: i for i, w in enumerate(bind["wires"])}
        return Hamiltonian(
            coeffs=bind["coeffs"],
            observables=[string_to_pauli_word(pauli_str, wire_map) for pauli_str in bind["ops"]],  # type: ignore
        )

    def value_to_zarr(self, bind_parent: ZarrGroup, key: str, value: Hamiltonian) -> ZarrGroup:
        wire_map = {w: i for i, w in enumerate(value.wires)}
        ops = [pauli_word_to_string(op, wire_map) for op in value.ops]

        grp = bind_parent.create_group(key)
        grp["ops"] = ops
        grp["wires"] = value.wires
        grp["coeffs"] = value.coeffs

        return grp
