import pennylane as qp
import catalyst
from jax import numpy as jnp

from pennylane.templates.subroutines.qrom import (
    _calculate_n_select_work_wires,
    _measurement_qrom_count_TemporaryAnd,
    _qrom_decomposition,
    _qrom_measurement_decomposition,
)

qp.decomposition.enable_graph()
from pennylane.ops.mid_measure.pauli_measure import PauliMeasure

gate_set = {
    qp.H,
    qp.X,
    qp.Y,
    qp.Z,
    qp.T,
    qp.S,
    'Adjoint(T)',
    'Adjoint(S)',
    qp.CNOT,
    qp.PauliX,
    qp.PauliY,
    qp.PauliZ,
    qp.Hadamard,
    qp.GlobalPhase,
    PauliMeasure,
}



p = [("my_pipe", ["quantum-compilation-stage"])]

@qp.qjit#(pipelines=p, target="mlir", capture=False)
#@qp.transforms.ppr_to_ppm
#@qp.transforms.to_ppr
@qp.set_shots(1)
@qp.decompose(gate_set=gate_set)
@qp.qnode(qp.device('lightning.qubit', wires = 3))
def circuit():

    qp.H(0)
    m1 = qp.pauli_measure("Z", 2)
    qp.cond(m1, qp.CZ)([0,1])

    return qp.probs(wires=[0,1])

print(circuit())
#ppm_specs = catalyst.passes.ppm_specs(circuit)
#print(ppm_specs)