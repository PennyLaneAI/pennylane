import pennylane as qml

dev = qml.device("default.mixed", wires=2)

c0 = qml.noise.op_eq(qml.RX)

def n0(op, **kwargs):
    qml.AmplitudeDamping(0.1, op.wires)

noise_model = qml.NoiseModel({c0: n0})

@qml.noise.add_noise(noise_model, level="top")
@qml.transforms.cancel_inverses
@qml.qnode(dev)
def circuit():
    qml.H(0)
    qml.H(0)
    qml.RX(0.5, 1)
    return qml.expval(qml.Z(0) @ qml.Z(1))

print("Transform program:", circuit.transform_program)
print(qml.draw(circuit)())
print("Result:", circuit())
