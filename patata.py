import pennylane as qml
import matplotlib.pyplot as plt
import numpy as np

dev = qml.device("default.qubit")

@qml.qnode(dev)
def circuit(angles):
    qml.Hadamard(0)
    qml.Hadamard(1)
    qml.Hadamard(2)
    qml.Hadamard(3)

    qml.MultiplexedRotation.compute_decomposition(angles, control_wires = [0,1,2], target_wire = 3, rot_axis = "Z")
    qml.Select.compute_decomposition([qml.RZ(-angle, 3) for angle in angles], control = [0,1,2])
    qml.Hadamard(0)
    qml.Hadamard(1)
    qml.Hadamard(2)
    qml.Hadamard(3)

    return qml.state()

angles = qml.math.array([1,2,31,4,23,52,1,14])
print(np.round(circuit(angles),2))

#qml.draw_mpl(circuit)(angles)
#plt.show()


