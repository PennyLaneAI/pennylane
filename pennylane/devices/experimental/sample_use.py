import pennylane as qml
import numpy as np

# Create a device which holds a simulator
dev1 = qml.devices.experimental.custom_device_3.python_device.TestDevicePythonSim()

#Prepare a an example circuit
prep = [qml.BasisState(np.array([1,1]), wires=(0,1))]
ops = [qml.RX(0.432, 0), qml.RY(0.543, 0), qml.CNOT((0,1)), qml.RX(0.133, 1)]
qscript = qml.tape.QuantumScript(ops, [qml.expval(qml.PauliZ(0))], prep)

# Run forward pass
dev1.execute(qscript)
