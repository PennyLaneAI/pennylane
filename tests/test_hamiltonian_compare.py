import pennylane as qml

def test_hamiltonian_compare():
    H2 = qml.Hamiltonian([0.3, 0.3], [qml.PauliZ(0), qml.PauliZ(0)])
    H1 = qml.Hamiltonian([0.1, 0.2, 0.3], [qml.PauliZ(0), qml.PauliZ(0), qml.PauliZ(0)])
    #print(H1.simplify()._obs_data())
    #print(H2.simplify()._obs_data())
    assert (H1.compare(H2))
test_hamiltonian_compare()