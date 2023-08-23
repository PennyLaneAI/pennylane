import pennylane as qml
import matplotlib.pyplot as plt

dev = qml.device("default.qubit", wires = 3)

@qml.qnode(dev)
def circuit():
    qml.ops.ctrl_evolution(qml.PauliX, control = [1, 2])(wires = 0)
    return qml.state()

qml.draw_mpl(circuit)()
plt.show()