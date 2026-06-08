import pennylane as qp
import numpy as np
from catalyst.device.decomposition import catalyst_decompose

@qp.decomposition.register_resources({qp.resource_rep(qp.PauliRot, pauli_word="XYZ"): 1})
def _tempAND_dummy_rule(wires, control_values):
    qp.PauliRot(np.pi/4, "XYZ", wires)


target_gates = {
        "PauliRot": 1,
        # "CNOT": 1,
        # "X": 1,
        # "Hadamard": 1,
        # "GlobalPhase": 1,
        "Cond": 1,
        "WhileLoop": 1,
        "ForLoop": 1,
        "Switch": 1,
        "HybridAdjoint": 1,
        "Snapshot": 1,
        "HybridCtrl": 1
        }

@qp.qjit(capture=False, target="mlir")
@qp.transforms.ppr_to_ppm
@qp.transforms.to_ppr
@catalyst_decompose(
    capabilities=None,
    target_gates=target_gates,
    fixed_decomps={qp.TemporaryAND: _tempAND_dummy_rule}
)
@qp.qnode(qp.device("null.qubit"))
def trotter_circuit():
    qp.TemporaryAND([0, 1, 2])
    return qp.expval(qp.Z(0))

specs = qp.specs(trotter_circuit, level="all-mlir")()
print(specs)