
import pennylane as qml
from pennylane import qaoa
import numpy as np

H1 = qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliY(0)])
H2 = qml.Hamiltonian([1, 1], [qml.PauliX(0), qml.PauliY(1)])

print(H1 + H2)

mixer_h = qaoa.creation_annihilation_mixer(["+-+", "+++"], [1, 1], wires=[0, 1, 2])
print(mixer_h)
