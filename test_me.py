import jax

import pennylane as qp

qp.capture.enable()


def my_func(phi, op, wires):
    if isinstance(op, qp.X):
        qp.RX(phi, wires[0])
    else:
        qp.RY(phi, wires[0])

    @qp.for_loop(0, 2)
    def loop(i):
        qp.CNOT([wires[i], wires[i + 1]])
        qp.adjoint(op)

    loop()


@qp.qjit
@qp.qnode(qp.device("null.qubit", wires=4))
def main(theta: float):
    qp.H(0)
    # see what happens when an operator crosses scopes
    qp.capture.subroutine(my_func)(0.2, qp.RX(theta, 0), jax.numpy.array([1, 2, 3]))


print(main.mlir)
