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


import pennylane.measurements as m

a = m.MeasurementValueV2("a")
b = m.MeasurementValueV2("b").apply_function(lambda x: (x * 52,) )

o = a.merge(b).merge(b)

print(o.fn((1, 1)))

print(o)



