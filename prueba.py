import pennylane as qml
import matplotlib.pyplot as plt

dev = qml.device("default.qubit", wires = 4)

@qml.qnode(dev)
def circuit():
    qml.ops.ctrl_evolution(qml.RX, control = [0, 1, 2])(0.25, wires = 3)
    return qml.state()

qml.draw_mpl(circuit, decimals = 2)()
plt.show()