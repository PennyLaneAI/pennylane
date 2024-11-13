import pennylane as qml
import jax

qml.capture.enable()


@qml.qnode(qml.device("default.qubit", wires=2))
def circuit(x):
    qml.RX(x, wires=0)
    qml.RY(x, wires=1)
    return qml.expval(qml.Z(0))


x = jax.numpy.array([0.1, 0.2, 0.3])

jaxpr = jax.make_jaxpr(circuit)(x)