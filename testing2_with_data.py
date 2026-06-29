import numpy as np

from algo_data.odmr.hBN_norbs_6 import cdf, soc_factorized, elec_state_S_0_MS_0

ham_data = np.load(cdf)
hamiltonian = {
    "core_tensors": ham_data["core_tensors"],
    "leaf_tensors": ham_data["leaf_tensors"],
    "nuc_constant": ham_data["nuc_constant"].item(),
}

soc_data = np.load(soc_factorized)
soc = {
    "soc_leaf": soc_data["soc_leaf"],
    "soc_core": soc_data["soc_core"],
}

state_data = np.load(elec_state_S_0_MS_0)
state = {
    "coefficients": state_data["coefficients"],
    "indices": state_data["indices"],
    "num_qubits": state_data["num_qubits"].item(),
}

print(np.array(hamiltonian["core_tensors"].shape))
print(np.array(hamiltonian["leaf_tensors"].shape))
print(np.array(hamiltonian["nuc_constant"]))
print(np.array(soc["soc_core"]).shape)
print(np.array(soc["soc_leaf"]).shape)

print(state["indices"])
print(state["num_qubits"])


import pennylane as qml
import numpy as np
import jax.numpy as jnp
from pennylane.labs.transforms import make_rz_to_phase_gradient_decomp
from catalyst.device.decomposition import catalyst_decompose
from pennylane.labs.templates import trotter_fragmented
from pennylane.labs.templates import SumOfSlatersPrep2
from pennylane.labs.registers import registers as registers2
from pennylane.labs.templates.sum_of_slaters2 import _sos_state_prep

def ctrl_soc_trotter(hamiltonian, control_wire, time, wires):
    U = np.asarray(hamiltonian.get("leaf_tensors", hamiltonian.get("soc_leaf")))
    Z = np.asarray(hamiltonian.get("core_tensors", hamiltonian.get("soc_core")))
    if U.ndim == 3:
        U, Z = U[0], Z[0]
    lam = np.real(np.diag(Z))
    B = U.T
    qml.adjoint(qml.BasisRotation(unitary_matrix=B, wires=wires))
    for i, w in enumerate(wires):
        qml.CRZ(-lam[i] * time, [control_wire, w])
    qml.BasisRotation(unitary_matrix=B, wires=wires)

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

_shape = hamiltonian["core_tensors"].shape
L = _shape[0] - 1
N = _shape[1]


qml.decomposition.enable_graph()

prec = 3


n = 1 + 2 * N                       # target qubits = hadamard_test(1) + system(2*N)
k =5
rng = np.random.default_rng(21)

# --- the three variables, derived from k ---
indices = tuple(sorted(rng.choice(2 ** n, size=k, replace=False).tolist()))
amps = rng.standard_normal(k) + 1j * rng.standard_normal(k)
coefficients = amps / np.linalg.norm(amps)

print()
print(n, indices)
print()
print()

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

thetas = jnp.array(np.random.rand(n_angles))
phis = jnp.array(np.random.rand(n_angles))
lambds = jnp.array(np.random.rand(n_angles))

dev = qml.device("lightning.qubit")


def U3(theta, phi, lambds, wires):
    qml.X(wires)
    qml.U3(2 * theta, phi, lambds, wires=wires)
    qml.X(wires)
    qml.Z(wires)

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


    _sos_state_prep(
        coefficients,
        wires=wires["hadamard"].union(wires["system"]),  # due to SOS multiplexor trick
        identification_wires=wires["identification_wires"],
        enumeration_wires=wires["enumeration_wires"],
        qrom_work_wires=wires["qrom_work_wires"],
        mcx_cache_wires=wires["mcx_cache_wires"],
        indices=indices,
    )


    U3(thetas[0], phis[0], lambds[0], wires=wires["GQSP"])


    @qml.for_loop(0, n_max, 1)
    def loop1(ind):

        trotter_fragmented(
                evolution_time=-1., num_trotter_steps=10, hamiltonian=hamiltonian,
                wires=wires["system"],
                control_wires=wires["GQSP"],
        )


        U3(thetas[ind + 1], phis[ind + 1], lambds[ind + 1], wires=wires["GQSP"])

    loop1()

    @qml.for_loop(n_max, len(thetas) - 1, 1)
    def loop2(ind):

        qml.X(wires["GQSP"])
        trotter_fragmented(
            evolution_time=1., num_trotter_steps=10, hamiltonian=hamiltonian,
            wires=wires["system"],
            control_wires=wires["GQSP"],
        )
        qml.X(wires["GQSP"])
        U3(thetas[ind + 1], phis[ind + 1], lambds[ind + 1], wires=wires["GQSP"])

    loop2()

    ctrl_soc_trotter(soc_data, wires["hadamard"], 1, wires["system"])


    return qml.state()


print(qml.specs(circuit, level='all-mlir')())

