import pennylane as qml
import numpy as np
import jax.numpy as jnp
from pennylane.labs.transforms import make_rz_to_phase_gradient_decomp
from catalyst.device.decomposition import catalyst_decompose
from pennylane.labs.templates import trotter_fragmented
from pennylane.labs.templates import SumOfSlatersPrep2
from pennylane.labs.registers import registers as registers2
from pennylane.labs.templates.sum_of_slaters2 import _sos_state_prep



CLIFFORD_T_REDUCED = {
    "CNOT": 50,
    "X": 1,
    "Y": 1,
    "T": 10,
    "Adjoint(T)": 10,
    "Z": 1,
    "GlobalPhase": 0,
    "S": 10,
    "Adjoint(S)": 10,
    "Hadamard": 10,
    "ForLoop": 0,
    "Cond": 0,
    "HybridAdjoint": 0,
    "HybridCtrl": 0,
}

L = 1; N=2
hamiltonian = {
    "core_tensors": np.random.rand(L+1, N, N),
    "leaf_tensors": np.random.rand(L+1, N, N),
    "nuc_constant": 0.5
}
qml.decomposition.enable_graph()

prec = 3


n = 1 + 2 * N                       # target qubits = hadamard_test(1) + system(2*N)
k =5
rng = np.random.default_rng(21)

# --- the three variables, derived from k ---
indices = tuple(sorted(rng.choice(2 ** n, size=k, replace=False).tolist()))
amps = rng.standard_normal(k) + 1j * rng.standard_normal(k)
coefficients = amps / np.linalg.norm(amps)

wires1 = registers2(SumOfSlatersPrep2.required_register_sizes(indices, n))


wires2 = registers2({"angle_wires": prec,
                       "phase_grad_wires": prec,
                       "work_wires": prec - 1,
                       "GQSP": 1,
                       "hadamard": 1,
                       "system": 2*N,
})

wires = wires1 + wires2


custom_decomp = make_rz_to_phase_gradient_decomp(
    wires["angle_wires"], wires["phase_grad_wires"], wires["work_wires"]
)


np.random.seed(0)
n_angles =  51
n_max = n_angles // 2

dev = qml.device("lightning.qubit")



pipelines = [
    ("Quantum", [
        "canonicalize",                               # <-- add this
        "builtin.module(apply-transform-sequence)",   # MUST be here, or your decorators don't run
        "inline-nested-module",                       # optional, tidies the IR
    ]),
]

@qml.qjit(target="mlir",capture=False,  pipelines=pipelines)
@qml.transforms.ppr_to_ppm
@qml.transforms.to_ppr
@qml.transform(pass_name="adjoint-lowering")
@catalyst_decompose(
    capabilities=None,
    target_gates=CLIFFORD_T_REDUCED,
    num_work_wires=0,
    fixed_decomps={qml.RZ: custom_decomp},
)
@qml.qnode(dev)
def circuit():

    @qml.for_loop(0, n_max, 1)
    def loop1(ind):

        trotter_fragmented(
                evolution_time=-1., num_trotter_steps=10, hamiltonian=hamiltonian,
                wires=wires["system"],
                control_wires=wires["GQSP"],
        )

    loop1()

    return qml.state()


print(qml.specs(circuit, level='all-mlir')())
