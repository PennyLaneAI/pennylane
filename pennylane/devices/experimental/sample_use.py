import pennylane as qml
import pennylane.numpy as np

# Create a device which holds a simulator
dev1 = qml.devices.experimental.custom_device_3.TestDevicePythonSim()

# define trainable params
params = np.array([0.432, 0.543, 0.133], requires_grad=True)

# Prepare a an example circuit
ops = [qml.RX(0.432, 0), qml.RY(0.543, 0), qml.CNOT((0, 1)), qml.RX(0.133, 1)]
obs =  [qml.expval(qml.PauliZ(0))]
qscript = qml.tape.QuantumScript(ops, obs)

# Run forward pass
dev1.execute(qscript)

# Todo
# 1. Add support for registrartion of a preprocessing step, to be called before execute
# 2. Add support for registrartion of a postprocessing step, to be called after execute
# 3. Integrate with device registratrion functionality
# 4. Tie into QNode structure for ML interface
# 5. Define result-types for local and distributed execution.