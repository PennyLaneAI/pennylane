# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tests for the serializing operators using the ``DatasetOperator`` and ``DatasetPyTree``
attribute types.
"""

import itertools

import numpy as np
import pytest

import pennylane as qml
from pennylane.data.attributes import DatasetOperator, DatasetPyTree
from pennylane.data.base.typing_util import get_type_str
from pennylane.operation import Operator

pytestmark = pytest.mark.data

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
    for op_cls, wires in itertools.product([qml.PauliX, qml.PauliY, qml.PauliZ], [0, 1, "q", [1]])
]

identity = [qml.Identity(wires) for wires in [0, 1, "q", [1, "a"]]]

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

tensors = [qml.prod(qml.PauliX(1), qml.PauliY(2))]


@pytest.mark.parametrize("attribute_cls", [DatasetOperator, DatasetPyTree])
@pytest.mark.parametrize("obs_in", [*hermitian_ops, *pauli_ops, *identity, *hamiltonians, *tensors])
class TestDatasetOperatorObservable:
    """Tests serializing Observable operators using the ``qml.equal`` function."""

    def test_value_init(self, attribute_cls, obs_in):
        """Test that a DatasetOperator can be value-initialized
        from an observable, and that the deserialized operator
        is equivalent."""

        dset_op = attribute_cls(obs_in)

        assert dset_op.info["type_id"] == attribute_cls.type_id
        assert dset_op.info["py_type"] == get_type_str(type(obs_in))

        obs_out = dset_op.get_value()
        qml.assert_equal(obs_out, obs_in)

    def test_bind_init(self, attribute_cls, obs_in):
        """Test that DatasetOperator can be initialized from a HDF5 group
        that contains an operator attribute."""

        bind = attribute_cls(obs_in).bind

        dset_op = attribute_cls(bind=bind)

        assert dset_op.info["type_id"] == attribute_cls.type_id
        assert dset_op.info["py_type"] == get_type_str(type(obs_in))

        obs_out = dset_op.get_value()
        qml.assert_equal(obs_out, obs_in)


@pytest.mark.parametrize("attribute_cls", [DatasetOperator, DatasetPyTree])
@pytest.mark.parametrize(
    "obs_in",
    [
        qml.ops.LinearCombination([1.0, 2.0], [qml.X(0) @ qml.Z(1), qml.Y(1) @ qml.Z(2)]),
        qml.ops.sum(qml.X(0), qml.Y(0)),
        qml.ops.sum(qml.X(0) @ qml.Z(1), 3 * qml.Y(2)),
        qml.ops.prod(qml.X(0), qml.Y(1)),
        qml.ops.s_prod(1.2j, qml.X(1) @ qml.Y(2)),
    ],
)
class TestDatasetArithmeticOperators:
    """Tests serializing Observable operators using the ``qml.equal()`` method."""

    def test_value_init(self, attribute_cls, obs_in):
        """Test that a DatasetOperator can be value-initialized
        from an observable, and that the deserialized operator
        is equivalent."""

        dset_op = attribute_cls(obs_in)

        assert dset_op.info["type_id"] == attribute_cls.type_id
        assert dset_op.info["py_type"] == get_type_str(type(obs_in))

        obs_out = dset_op.get_value()
        qml.assert_equal(obs_out, obs_in)

    def test_bind_init(self, attribute_cls, obs_in):
        """Test that DatasetOperator can be initialized from a HDF5 group
        that contains an operator attribute."""

        bind = attribute_cls(obs_in).bind

        dset_op = attribute_cls(bind=bind)

        assert dset_op.info["type_id"] == attribute_cls.type_id
        assert dset_op.info["py_type"] == get_type_str(type(obs_in))

        obs_out = dset_op.get_value()
        qml.assert_equal(obs_out, obs_in)


@pytest.mark.parametrize("attribute_cls", [DatasetOperator, DatasetPyTree])
class TestDatasetOperator:
    @pytest.mark.parametrize(
        "op_in",
        [
            qml.RX(1.1, 0),
            qml.FermionicSWAP(1.3, [1, "a"]),
            qml.Toffoli([1, "a", None]),
            qml.Hamiltonian([], []),
        ],
    )
    def test_value_init(self, attribute_cls, op_in):
        """Test that a DatasetOperator can be value-initialized
        from an operator, and that the deserialized operator
        is equivalent."""

        dset_op = attribute_cls(op_in)

        assert dset_op.info["type_id"] == attribute_cls.type_id
        assert dset_op.info["py_type"] == get_type_str(type(op_in))

        op_out = dset_op.get_value()
        with np.printoptions(legacy="1.21"):
            assert repr(op_out) == repr(op_in)
        assert op_in.data == op_out.data

    @pytest.mark.parametrize(
        "op_in",
        [
            qml.RX(1.1, 0),
            qml.FermionicSWAP(1.3, [1, "a"]),
            qml.Toffoli([1, "a", None]),
            qml.Hamiltonian([], []),
        ],
    )
    def test_bind_init(self, attribute_cls, op_in):
        """Test that a DatasetOperator can be bind-initialized
        from an operator, and that the deserialized operator
        is equivalent."""

        bind = attribute_cls(op_in).bind

        dset_op = attribute_cls(bind=bind)

        assert dset_op.info["type_id"] == attribute_cls.type_id
        assert dset_op.info["py_type"] == get_type_str(type(op_in))

        op_out = dset_op.get_value()
        with np.printoptions(legacy="1.21"):
            assert repr(op_out) == repr(op_in)
        assert op_in.data == op_out.data
        assert op_in.wires == op_out.wires
        with np.printoptions(legacy="1.21"):
            assert repr(op_in) == repr(op_out)


@pytest.mark.parametrize("attribute_cls", [DatasetOperator, DatasetPyTree])
def test_op_not_queued_on_deserialization(attribute_cls):
    """Tests that ops are not queued upon deserialization."""
    d = qml.data.Dataset(op=attribute_cls(qml.PauliX(0)))
    with qml.queuing.AnnotatedQueue() as q:
        _ = d.op

    assert len(q) == 0

    with qml.queuing.AnnotatedQueue() as q2:
        qml.apply(d.op)

    assert len(q2) == 1


def test_consumed_by_pytree():
    """Test that ops are consumed by the ``DatasetPyTree`` type by default."""

    d = qml.data.Dataset()

    d.op = qml.PauliX(0)

    assert isinstance(d.attrs["op"], DatasetPyTree)


def test_value_init_not_supported():
    """Test that a ValueError is raised if attempting to serialize an unsupported operator
    using the ``DatasetOperator`` attribute type."""

    class NotSupported(Operator):  # pylint: disable=too-few-public-methods, unnecessary-ellipsis
        """An operator."""

    with pytest.raises(
        TypeError, match="Serialization of operator type 'NotSupported' is not supported"
    ):
        DatasetOperator(NotSupported(1))


def test_retrieve_operator_from_loaded_data():
    """Test that uploaded data can be downloaded and used to retrieve an
    operation representing the Hamiltonian"""

    h2 = qml.data.load("qchem", molname="H2", bondlength=0.742, basis="STO-3G")[0]
    H = h2.hamiltonian

    assert isinstance(H, qml.ops.LinearCombination)
