poly = np.array([0, -1, 0, 0.5, 0, 0.5])
H = 0.1 * qml.X(3) - 0.7 * qml.X(3) @ qml.Z(4) - 0.2 * qml.Z(3) @ qml.Y(4)

control_wires = [1, 2]
block_encode = qml.Qubitization(H, control=control_wires)
angles = qml.poly_to_angles(poly, "QSVT")
projectors = [
    qml.PCPhase(angles[i], dim=2 ** len(H.wires), wires=control_wires + H.wires)
    for i in range(len(angles))
]

dev = qml.device("default.qubit")


@qml.qnode(dev)
def circuit():
    qml.Hadamard(0)
    qml.ctrl(qml.QSVT, control=0, control_values=[1])(block_encode, projectors)
    qml.ctrl(qml.adjoint(qml.QSVT), control=0, control_values=[0])(
        block_encode, projectors
    )
    qml.Hadamard(0)
    return qml.state()


matrix = qml.matrix(circuit, wire_order=[0] + control_wires + H.wires)()
