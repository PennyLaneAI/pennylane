import pennylane as qml
import numpy as np
import pytest
from pennylane.data.attributes.operator import DatasetHamiltonian

H_ONE_QUBIT = np.array([[1.0, 0.5j], [-0.5j, 2.5]])
H_TWO_QUBITS = np.array(
    [[0.5, 1.0j, 0.0, -3j], [-1.0j, -1.1, 0.0, -0.1], [0.0, 0.0, -0.9, 12.0], [3j, -0.1, 12.0, 0.0]]
)

hamiltonians = [
    qml.Hamiltonian(*args)
    for args in [
        (
            [
                1.0,
            ],
            (qml.Hermitian(H_TWO_QUBITS, [0, 1]),),
        ),
        ((-0.8,), (qml.PauliZ(0),)),
        ((0.6,), (qml.PauliX(0) @ qml.PauliX(1),)),
        ((0.5, -1.6), (qml.PauliX(0), qml.PauliY(1))),
        ((0.5, -1.6), (qml.PauliX(1), qml.PauliY(1))),
        ((0.5, -1.6), (qml.PauliX("a"), qml.PauliY("b"))),
        ((1.1, -0.4, 0.333), (qml.PauliX(0), qml.Hermitian(H_ONE_QUBIT, 2), qml.PauliZ(2))),
        ((-0.4, 0.15), (qml.Hermitian(H_TWO_QUBITS, [0, 2]), qml.PauliZ(1))),
        ([1.5, 2.0], [qml.PauliZ(0), qml.PauliY(2)]),
        (np.array([-0.1, 0.5]), [qml.Hermitian(H_TWO_QUBITS, [0, 1]), qml.PauliY(0)]),
        ((0.5, 1.2), (qml.PauliX(0), qml.PauliX(0) @ qml.PauliX(1))),
        ((0.5 + 1.2j, 1.2 + 0.5j), (qml.PauliX(0), qml.PauliY(1))),
        ((0.7 + 0j, 0 + 1.3j), (qml.PauliX(0), qml.PauliY(1))),
    ]
]


@pytest.mark.parametrize("obs_in", hamiltonians)
class TestDatasetHamiltonian:
    def test_value_init(self, obs_in):
        """Test that a DatasetHamiltonian can be value-initialized
        from an observable, and that the deserialized operator
        is equivalent."""
        dset_op = DatasetHamiltonian(obs_in)

        assert dset_op.info["type_id"] == "hamiltonian"
        assert dset_op.info["py_type"] == f"{qml.Hamiltonian.__module__}.Hamiltonian"

        obs_out = dset_op.get_value()
        assert repr(obs_out) == repr(obs_in)
        assert obs_in.compare(obs_out)

    def test_bind_init_observable(self, obs_in):
        """Test that DatasetHamiltonian can be initialized from a HDF5 group
        that contains a operator attribute."""
        bind = DatasetHamiltonian(obs_in).bind

        dset_op = DatasetHamiltonian(bind=bind)

        assert dset_op.info["type_id"] == "hamiltonian"
        assert dset_op.info["py_type"] == f"{qml.Hamiltonian.__module__}.Hamiltonian"

        obs_out = dset_op.get_value()
        assert repr(obs_out) == repr(obs_in)
        assert obs_in.compare(obs_out)
