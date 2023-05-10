from pennylane.data.attributes import DatasetOperator
from pennylane.data.base.typing_util import get_type_str
import pytest
import pennylane as qml
import numpy as np

H_ONE_QUBIT = np.array([[1.0, 0.5j], [-0.5j, 2.5]])

H_TWO_QUBITS = np.array(
    [[0.5, 1.0j, 0.0, -3j], [-1.0j, -1.1, 0.0, -0.1], [0.0, 0.0, -0.9, 12.0], [3j, -0.1, 12.0, 0.0]]
)
valid_hamiltonians = [
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


@pytest.mark.parametrize(
    "value",
    [*[qml.Hamiltonian(*args) for args in valid_hamiltonians]],
)
class TestDatasetOperator:
    def test_value_init_with_hamiltonians(self, value):
        """Test that a DatasetOperator can be value-initialized
        from a Hamiltonian."""
        dset_op = DatasetOperator(value)

        assert dset_op.get_value().compare(value)
        assert repr(dset_op.get_value()) == repr(dset_op.get_value())
        assert dset_op.info.py_type == get_type_str(qml.Hamiltonian)
        assert dset_op.info["operator_class"] == get_type_str(type(value))
