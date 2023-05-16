
class QChemHamiltonian(AttributeType[ZarrGroup, Hamiltonian, Hamiltonian]):
    """Attribute type for QChem dataset hamiltonians, which use only Pauli operators."""

    type_id = "qchem_hamiltonian"

    def __post_init__(self, value: Hamiltonian, info):
        """Save the class name of the operator ``value`` into the
        attribute info."""
        super().__post_init__(value, info)
        self.info["operator_class"] = type(value).__qualname__

    @classmethod
    def consumes_types(cls) -> Tuple[Type[Hamiltonian]]:
        return (Hamiltonian,)

    def zarr_to_value(self, bind: ZarrGroup) -> Hamiltonian:
        wire_map = {json.loads(w): i for i, w in enumerate(bind["wires"].asstr())}

        ops = [string_to_pauli_word(pauli_string, wire_map) for pauli_string in bind["ops"].asstr()]
        coeffs = list(bind["coeffs"])

        return Hamiltonian(coeffs, ops)

    def value_to_zarr(self, bind_parent: ZarrGroup, key: str, value: Hamiltonian) -> ZarrGroup:
        bind = bind_parent.create_group(key)

        coeffs, ops = value.terms()
        wire_map = {w: i for i, w in enumerate(value.wires)}

        bind["ops"] = [pauli_word_to_string(op, wire_map) for op in ops]
        bind["coeffs"] = coeffs
        bind["wires"] = [json.dumps(w) for w in value.wires]

        return bind
