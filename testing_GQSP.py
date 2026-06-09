import pennylane as qml
import numpy as np
import jax.numpy as jnp
from pennylane.labs.transforms import make_rz_to_phase_gradient_decomp
from catalyst.device.decomposition import catalyst_decompose
from pennylane.labs.templates import trotter_fragmented

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

L = 2; M=1; N=2
hamiltonian = {
    "core_tensors": np.random.rand(L, M, M, N, N),
    "leaf_tensors": np.random.rand(L, M, N, N),
    "nuc_constant": 0.5
}
qml.decomposition.enable_graph()

prec = 3

wires = qml.registers({"angle_wires": prec,
                       "phase_grad_wires": prec,
                       "work_wires": prec - 1,
                       "hadamard": 1,
                       "system": M*N})

custom_decomp = make_rz_to_phase_gradient_decomp(
    wires["angle_wires"], wires["phase_grad_wires"], wires["work_wires"]
)

np.random.seed(0)
n_angles =  51
n_max = n_angles // 2

thetas = jnp.array(np.random.rand(n_angles))
phis = jnp.array(np.random.rand(n_angles))
lambds = jnp.array(np.random.rand(1))

dev = qml.device("lightning.qubit", wires=3 * prec + 1)


def U3(theta, phi, lambds, wires):
    qml.X(wires)
    qml.U3(2 * theta, phi, lambds, wires=wires)
    qml.X(wires)
    qml.Z(wires)

p = [("my_pipe", ["quantum-compilation-stage"])]
@qml.qjit(target='mlir', pipelines=p, capture=False, autograph=False)
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

    U3(thetas[0], phis[0], lambds[0], wires=wires["hadamard"])


    @qml.for_loop(0, n_max, 1)
    def loop1(ind):

        trotter_fragmented(
                evolution_time=-1., num_trotter_steps=10, hamiltonian=hamiltonian,
                wires=wires["system"],
                control_wires=wires["hadamard"],
        )


        U3(thetas[ind + 1], phis[ind + 1], lambds[ind + 1], wires=wires["hadamard"])

    loop1()

    @qml.for_loop(n_max, len(thetas) - 1, 1)
    def loop2(ind):

        qml.X(wires["hadamard"])
        trotter_fragmented(
            evolution_time=1., num_trotter_steps=10, hamiltonian=hamiltonian,
            wires=wires["system"],
            control_wires=wires["hadamard"],
        )
        qml.X(wires["hadamard"])
        U3(thetas[ind + 1], phis[ind + 1], lambds[ind + 1], wires=wires["hadamard"])

    loop2()

    return qml.state()


print(qml.specs(circuit, level='all-mlir')())
