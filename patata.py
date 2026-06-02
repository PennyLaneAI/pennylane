import pennylane as qp
from pennylane.labs.templates import SuperpositionTHC

n = 3
M, N = 5, 2
mu_wires = list(range(0, n))
nu_wires = list(range(n, 2 * n))
work_wires = list(range(2 * n, 2 * n + 3 * n + 5))

dev = qp.device("lightning.qubit", wires=2 * n + 3 * n + 5)

@qp.qnode(dev)
def circuit():
    SuperpositionTHC(M, N, mu_wires, nu_wires, work_wires)
    return qp.probs(mu_wires + nu_wires)

print(circuit())