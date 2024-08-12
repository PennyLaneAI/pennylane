import pennylane as qml

print(qml.BasisState.compute_decomposition([1,0], wires=(0,1)))