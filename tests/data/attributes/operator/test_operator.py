import itertools

import numpy as np
import pytest

import pennylane as qml
from pennylane.data.attributes import DatasetOperator
from pennylane.data.base.typing_util import get_type_str
from pennylane.operation import Tensor

H_ONE_QUBIT = np.array([[1.0, 0.5j], [-0.5j, 2.5]])
H_TWO_QUBITS = np.array(
    [[0.5, 1.0j, 0.0, -3j], [-1.0j, -1.1, 0.0, -0.1], [0.0, 0.0, -0.9, 12.0], [3j, -0.1, 12.0, 0.0]]
)

hermitian_ops = [
    qml.Hermitian(H_ONE_QUBIT, 2),
    qml.Hermitian(H_TWO_QUBITS, [0, 1]),
    qml.PauliX(1),
    qml.Identity("a"),
]

pauli_ops = [
    op_cls(wires)
    for op_cls, wires in itertools.product(
        [qml.PauliX, qml.PauliY, qml.PauliZ], [0, 1, "q", None, [1]]
    )
]

identity = [qml.Identity(wires) for wires in [0, 1, "q", None, [1, "a"]]]


@pytest.mark.parametrize(
    "obs_in", [*hermitian_ops, *pauli_ops, *identity, Tensor(qml.PauliX(1), qml.PauliY(2))]
)
class TestDatasetOperatorObservable:
    def test_value_init_observable(self, obs_in):
        """Test that a DatasetOperator can be value-initialized
        from an observable, and that the deserialized operator
        is equivalent."""
        dset_op = DatasetOperator(obs_in)

        assert dset_op.info["type_id"] == "operator"
        assert dset_op.info["py_type"] == get_type_str(type(obs_in))

        obs_out = dset_op.get_value()
        assert repr(obs_out) == repr(obs_in)
        assert obs_in.compare(obs_out)

    def test_bind_init_observable(self, obs_in):
        """Test that DatasetOperator can be initialized from a HDF5 group
        that contains a operator attribute."""
        bind = DatasetOperator(obs_in).bind

        dset_op = DatasetOperator(bind=bind)

        assert dset_op.info["type_id"] == "operator"
        assert dset_op.info["py_type"] == get_type_str(type(obs_in))

        obs_out = dset_op.get_value()
        assert repr(obs_out) == repr(obs_in)
        assert obs_in.compare(obs_out)
