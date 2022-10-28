from typing import List, Union

from pennylane.tape import QuantumScript
from pennylane import simplify


def stopping_condition(obj):
    if obj.name == "QFT" and len(obj.wires) >= 6:
        return False
    if obj.name == "GroverOperator" and len(obj.wires) >= 13:
        return False
    return getattr(obj, "has_matrix", False)


def simple_preprocessor(qscript: Union[QuantumScript, List[QuantumScript]]) -> Union[QuantumScript, List[QuantumScript]]:

    if not isinstance(qscript, QuantumScript):
        return [simple_preprocessor(qs) for qs in qscript]

    max_expansion = 20
    new_qscript = qscript.expand(depth=max_expansion, stop_at=stopping_condition)

    for op in new_qscript.operations:
        if not stopping_condition(op):
            raise NotImplementedError(f"{op} not supported on device")
    if new_qscript.num_wires > 30:
        raise NotImplementedError(f"Requested execution with {new_qscript.num_wires} qubits. We support at most 30.")

    return simplify(new_qscript)
