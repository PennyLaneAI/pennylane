import pennylane as qml
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import transform
from pennylane.typing import PostprocessingFn

has_pyzx = True
try:
    import pyzx
except ImportError:
    has_pyzx = False


@transform
def reduce_zx_calculus(tape: QuantumScript) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    if not has_pyzx:
        raise ModuleNotFoundError("pyzx is required.")

    zx_graph = qml.transforms.to_zx(tape, expand_measurements=False)
    pyzx.full_reduce(zx_graph)
    qscript = qml.transforms.from_zx(zx_graph, decompose_phases=False)
    new_tape = tape.copy(operations=qscript.operations)

    def null_postprocessing(results):
        return results[0]

    return (new_tape,), null_postprocessing
