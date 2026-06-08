import pennylane as qp
import catalyst
from jax import numpy as jnp



from pennylane.templates.subroutines.qrom import (
    _calculate_n_select_work_wires,
    _measurement_qrom_count_TemporaryAnd,
    _qrom_decomposition,
    _qrom_measurement_decomposition,
)

from pennylane.ops.mid_measure.pauli_measure import PauliMeasure

gate_set = {
    qp.H,
    qp.X,
    qp.Y,
    qp.Z,
    qp.CNOT,
    qp.PauliX,
    qp.T,
    qp.S,
    'Adjoint(T)',
    'Adjoint(S)',
    qp.PauliY,
    qp.PauliZ,
    qp.Hadamard,
    qp.GlobalPhase,
    PauliMeasure,
    qp.PauliRot,
    "Cond",
    "WhileLoop",
    "ForLoop",
    "Switch",
    "HybridAdjoint",
    "Snapshot",
    "HybridCtrl"
    "Measure"
}

p = [("my_pipe", ["quantum-compilation-stage"])]
device = qp.device("lightning.qubit", wires=7)

"""
@qp.qjit#(pipelines=p, target="mlir", capture=True, autograph=True)
@catalyst.passes.ppm_compilation
@qp.decompose(gate_set=gate_set)
@qp.qnode(device)
def circuit():
    _qrom_measurement_decomposition(data=jnp.array([ [1, 1, 1], [1, 0, 1], [1, 0, 0], [1, 1, 0]]),
            control_wires=jnp.array([3, 4]),
            target_wires=jnp.array([0, 1, 2]),
            work_wires=jnp.array([5, 6]),
    )
    return catalyst.measure(0)


#ppm_specs = catalyst.passes.ppm_specs(circuit)
#print(ppm_specs)
circuit()
"""
import numpy as np
import math

results = []

@qp.decomposition.register_resources({qp.resource_rep(qp.PauliRot, pauli_word="ZZY"): 1,
                                      qp.resource_rep(qp.PauliRot, pauli_word="IZY"): 1,
                                      qp.resource_rep(qp.PauliRot, pauli_word="ZIY"): 1,
                                      qp.resource_rep(qp.PauliRot, pauli_word="IIY"): 1})
def _tempAND_rule(wires, control_values):
    qp.PauliRot(np.pi/4, "ZZY", wires)
    qp.PauliRot(-np.pi/4, "IZY", wires)
    qp.PauliRot(-np.pi/4, "ZIY", wires)
    qp.PauliRot(np.pi/4, "IIY", wires)


@qp.decomposition.register_resources({qp.Hadamard: 1, qp.ops.MidMeasure: 1})
def _adj_tempAND_rule(base, wires):
    qp.Hadamard(wires[2])
    m = qp.measure(wires[2])
    qp.cond(m, qp.CZ)(wires[:2])

rango = range(2, 7)
for j in rango:
    L = 2**j
    rng = np.random.default_rng(42 + L)
    n_target = 20
    n_input = math.ceil(math.log2(L))
    n_work = n_input

    bitstrings = rng.integers(0, 2, size=(L, n_target)).tolist()

    total_wires = n_input + n_work + n_target
    dev = qp.device("lightning.qubit", wires=total_wires)

    control_wires = list(range(n_input))
    work_wires = list(range(n_input, n_input + n_work))
    target_wires = list(range(n_input + n_work, total_wires))

    x_state = rng.random(L) + 1j * rng.random(L)
    x_state /= np.linalg.norm(x_state)

    qp.decomposition.enable_graph()

    @qp.qjit(pipelines=p, target="mlir", capture = False, autograph=False)
    #@catalyst.passes.ppm_compilation
    @qp.transforms.ppr_to_ppm
    @qp.transforms.to_ppr
    @catalyst.device.decomposition.catalyst_decompose(target_gates=gate_set, capabilities=None, fixed_decomps={
        qp.QROM: _qrom_measurement_decomposition
    }) #, })
    @qp.qnode(dev)
    def circuit():

        qp.QROM(
        #_qrom_decomposition(
        #_qrom_measurement_decomposition(
            data=bitstrings,
            control_wires=control_wires,
            target_wires=target_wires,
            work_wires=work_wires,
            clean=False
        )

        return qp.probs(wires=control_wires)

    #print(circuit.mlir)
    print(qp.specs(circuit, level='all-mlir')())
    # results.append(ppm_specs['circuit_0']['num_of_ppm'])
    #print(qp.draw(circuit)())
    #print(circuit.mlir_opt)

import matplotlib.pyplot as plt

# plt.plot(results)
# plt.show()
