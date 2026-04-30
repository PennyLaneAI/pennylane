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


def main():
    qp.H(0)
    # see what happens when an operator crosses scopes
    jax.jit(my_func, static_argnums=1)(0.2, qp.X(0), jax.numpy.array([1, 2, 3]))


print(jax.make_jaxpr(main)())
