from typing import List, Union

from pennylane.tape import QuantumScript
from pennylane import simplify, map_wires


def stopping_condition(obj):
    if obj.name == "QFT" and len(obj.wires) >= 6:
        return False
    if obj.name == "GroverOperator" and len(obj.wires) >= 13:
        return False
    return getattr(obj, "has_matrix", False)


def simple_preprocessor(
    qscript: Union[QuantumScript, List[QuantumScript]]
) -> Union[QuantumScript, List[QuantumScript]]:

    if not isinstance(qscript, QuantumScript):
        return [simple_preprocessor(qs) for qs in qscript]

    max_expansion = 20

    new_qscript = qscript.expand(
        depth=max_expansion, stop_at=stopping_condition, expand_measurements=False
    )

    for op in new_qscript.operations:
        if not stopping_condition(op):
            raise NotImplementedError(f"{op} not supported on device")
    if new_qscript.num_wires > 30:
        raise NotImplementedError(
            f"Requested execution with {new_qscript.num_wires} qubits. We support at most 30."
        )

    wire_map = {w: i for i, w in enumerate(qscript.wires)}
    new_qscript = map_wires(new_qscript, wire_map)

    return new_qscript  # simplify(new_qscript)
