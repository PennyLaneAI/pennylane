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

L = 2; M=2; N=3
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
n_angles =  651
n_max = n_angles // 2

thetas = jnp.array(np.random.rand(n_angles))
phis = jnp.array(np.random.rand(n_angles))
lambds = jnp.array(np.random.rand(1))

dev = qml.device("lightning.qubit", wires=3 * prec + 1)

c_ctrl = wires['circuit_wires'][0]
c_targ = wires['circuit_wires'][1]


def U3(theta, phi, lambds, wires):
    qml.X(wires)
    qml.U3(2 * theta, phi, lambds, wires=wires)
    qml.X(wires)
    qml.Z(wires)




@qml.qjit(target="mlir", capture=False, autograph=False)
#@qml.transforms.ppr_to_ppm
@qml.transforms.to_ppr
@catalyst_decompose(
    capabilities=None,
    target_gates=CLIFFORD_T_REDUCED,
    num_work_wires=0,
    fixed_decomps={qml.RZ: custom_decomp},
)
@qml.qnode(dev)
def circuit():

    U3(thetas[0], phis[0], lambds[0], wires=c_ctrl)

    @qml.for_loop(0, n_max, 1)
    def loop1(ind):
        qml.X(wires["system"])
        trotter_fragmented(
            evolution_time=1., num_trotter_steps=10, hamiltonian=hamiltonian,
            wires=wires["system"],
            control_wires=wires["hadamard"],
        )
        qml.X(wires["system"])

        qml.adjoint(
            trotter_fragmented(
                evolution_time=1., num_trotter_steps=10, hamiltonian=hamiltonian,
                wires=wires["system"],
                control_wires=wires["hadamard"],
            )
        )


        U3(thetas[ind + 1], phis[ind + 1], lambds[ind + 1], wires=c_ctrl)

    loop1()
    """
    @qml.for_loop(n_max, len(thetas) - 1, 1)
    def loop2(ind):
        # control_values=[0] triggers an infinite recursion in the graph-based
        # decomposition under catalyst_decompose (PL 0.45 / Catalyst 0.15).
        # Equivalent rewrite: X-conjugate the control wire and use control_values=[1].
        qml.X(c_ctrl)
        qml.ctrl(qml.RZ(input_val, wires=c_targ),
                 control=c_ctrl, control_values=[1])
        qml.X(c_ctrl)
        U3(thetas[ind + 1], phis[ind + 1], lambds[ind + 1], wires=c_ctrl)

    loop2()
    """
    return qml.state()


print(qml.specs(circuit, level='all-mlir')())
