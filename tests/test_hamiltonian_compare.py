#to test if hamiltonian comapre function has float point error
import pennylane as qml

def test_hamiltonian_compare():
    H2 = qml.Hamiltonian([0.3, 0.3], [qml.PauliZ(0), qml.PauliZ(0)])
    H1 = qml.Hamiltonian([0.1, 0.2, 0.3], [qml.PauliZ(0), qml.PauliZ(0), qml.PauliZ(0)])
    assert (H1.compare(H2))