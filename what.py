# import pennylane as qml
#
#
# dev = qml.device("default.qubit", wires=2)
#
# @qml.qnode(dev)
# def circuit(x, y):
#     qml.RX(x * y, wires=0)
#     o = qml.measure(wires=0)
#     return o
#
# print(circuit(4, 5))

import numpy as np
import jax.numpy as jnp
import pennylane.measurements as m

a = m.MeasurementValueV2("a")
b = m.MeasurementValueV2("b")
print(a)

print(a + b)


# def hey(x):
#     return 52 * x
#
# b = m.MeasurementValueV2("b").apply(hey)
# print(b)
# o = a.merge(b).merge(b)
# print(o)
#
# print(o.fn((1, 1)))
#
# print(o)
# print(o[0])
# np.sin(o)



