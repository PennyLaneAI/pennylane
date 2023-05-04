from collections.abc import Iterable

import zarr
from pennylane import Hamiltonian
from pennylane.pauli import pauli_word_to_string, string_to_pauli_word

from pennylane.data.base.attribute import AttributeType


class DatasetHamiltonian(AttributeType[zarr.Group, Hamiltonian]):
    type_id = "hamiltonian"

    @classmethod
    def consumes_types(cls) -> Iterable[type]:
        return (Hamiltonian,)

    def zarr_to_value(self, bind: zarr.Group) -> Hamiltonian:
        wire_map = {w: i for i, w in enumerate(bind["wires"])}
        return Hamiltonian(
            coeffs=bind["coeffs"],
            observables=[string_to_pauli_word(pauli_str, wire_map) for pauli_str in bind["ops"]],  # type: ignore
        )

    def value_to_zarr(self, bind_parent: zarr.Group, key: str, value: Hamiltonian) -> zarr.Group:
        wire_map = {w: i for i, w in enumerate(value.wires)}
        ops = [pauli_word_to_string(op, wire_map) for op in value.ops]

        grp = bind_parent.create_group(key)
        grp["ops"] = ops
        grp["wires"] = value.wires
        grp["coeffs"] = value.coeffs

        return grp
