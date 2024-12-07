"""
Provides a transform to combine all qml.GlobalPhase gates in a circuit into a single one applied at the end.
"""

import pennylane as qml
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import transform
from pennylane.typing import PostprocessingFn


@transform
def combine_global_phases(tape: QuantumScript) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    """todo"""
    phi = 0
    operations = []
    for op in tape.operations:
        if isinstance(op, qml.GlobalPhase):
            phi += op.parameters[0]
        else:
            operations.append(op)

    if phi != 0:
        operations.append(qml.GlobalPhase(phi=phi))

    new_tape = type(tape)(operations, tape.measurements, shots=tape.shots)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing
