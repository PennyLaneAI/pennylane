import pennylane as qml
from pennylane import numpy as np
import jax

@qml.qnode(qml.device("default.qubit"))
def circuit(coeffs):

    H = qml.ops.LinearCombination(coeffs, [qml.PauliZ(0), qml.PauliZ(1), qml.PauliZ(1), qml.PauliZ(1)])

    #new_coeffs = qml.math.abs(qml.math.array(terms[0]))

    qml.AmplitudeEmbedding(coeffs, normalize = True, wires=(3,4))
    qml.Qubitization(H, control = [3,4])
    return qml.expval(qml.PauliZ(3)@qml.PauliZ(4))

coeffs = jax.numpy.array([0.4, 0.5, 0.1, 0.3])

grad = jax.grad(circuit)(coeffs)
#grad = circuit(coeffs)
print(grad)
