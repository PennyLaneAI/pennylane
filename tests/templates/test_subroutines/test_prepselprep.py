# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Tests for the PrepSelPrep template.
"""
# pylint: disable=protected-access,import-outside-toplevel
import copy

import numpy as np
import pytest

import pennylane as qml


@pytest.mark.parametrize(
    ("lcu", "control", "skip_diff"),
    [
        (qml.ops.LinearCombination([0.25, 0.75], [qml.Z(2), qml.X(1) @ qml.X(2)]), [0], False),
        (qml.dot([0.25, 0.75], [qml.Z(2), qml.X(1) @ qml.X(2)]), [0], False),
        (qml.Hamiltonian([0.25, 0.75], [qml.Z(2), qml.X(1) @ qml.X(2)]), [0], False),
        (0.25 * qml.Z(2) - 0.75 * qml.X(1) @ qml.X(2), [0], False),
        (qml.Z(2) + qml.X(1) @ qml.X(2), [0], False),
        (qml.ops.LinearCombination([-0.25, 0.75j], [qml.Z(3), qml.X(2) @ qml.X(3)]), [0, 1], True),
        (
            qml.ops.LinearCombination([-0.25 + 0.1j, 0.75j], [qml.Z(4), qml.X(4) @ qml.X(5)]),
            [0, 1, 2, 3],
            True,
        ),
    ],
)
def test_standard_checks(lcu, control, skip_diff):
    """Run standard validity tests."""

    op = qml.PrepSelPrep(lcu, control)
    # Skip differentiation for test cases that raise NaNs in gradients (known limitation of ``MottonenStatePreparation``).
    qml.ops.functions.assert_valid(op, skip_differentiation=skip_diff)


def test_repr():
    """Test the repr method."""

    lcu = qml.dot([0.25, 0.75], [qml.Z(2), qml.X(1) @ qml.X(2)])
    control = [0]

    op = qml.PrepSelPrep(lcu, control)
    with np.printoptions(legacy="1.21"):
        assert (
            repr(op)
            == "PrepSelPrep(coeffs=(0.25, 0.75), ops=(Z(2), X(1) @ X(2)), control=Wires([0]))"
        )


def _get_new_terms(lcu):
    """Compute a new sum of unitaries with positive coefficients"""

    new_coeffs = []
    new_ops = []

    for coeff, op in zip(*lcu.terms()):

        angle = qml.math.angle(coeff)
        new_coeffs.append(qml.math.abs(coeff))

        new_op = op @ qml.GlobalPhase(-angle, wires=op.wires)
        new_ops.append(new_op)

    interface = qml.math.get_interface(lcu.terms()[0])
    new_coeffs = qml.math.array(new_coeffs, like=interface)

    return new_coeffs, new_ops


# Use these circuits in tests
def manual_circuit(lcu, control):
    """Circuit equivalent to decomposition of PrepSelPrep"""
    coeffs, ops = _get_new_terms(lcu)

    qml.AmplitudeEmbedding(qml.math.sqrt(coeffs), normalize=True, pad_with=0, wires=control)
    qml.Select(ops, control=control)
    qml.adjoint(
        qml.AmplitudeEmbedding(qml.math.sqrt(coeffs), normalize=True, pad_with=0, wires=control)
    )

    return qml.state()


def prepselprep_circuit(lcu, control):
    """PrepSelPrep circuit used for testing"""
    qml.PrepSelPrep(lcu, control)
    return qml.state()


a_set_of_lcus = [
    qml.ops.LinearCombination([0.25, 0.75], [qml.Z(2), qml.X(1) @ qml.X(2)]),
    qml.dot([0.25, 0.75], [qml.Z(2), qml.X(1) @ qml.X(2)]),
    qml.Hamiltonian([0.25, 0.75], [qml.Z(2), qml.X(1) @ qml.X(2)]),
    0.25 * qml.Z(2) - 0.75 * qml.X(1) @ qml.X(2),
    qml.Z(2) + qml.X(1) @ qml.X(2),
    qml.ops.LinearCombination([-0.25, 0.75j], [qml.Z(3), qml.X(2) @ qml.X(3)]),
    qml.ops.LinearCombination([-0.25 + 0.1j, 0.75j], [qml.Z(4), qml.X(4) @ qml.X(5)]),
]


class TestPrepSelPrep:
    """Test the correctness of the decomposition"""

    # Use these LCUs as test input
    lcu1 = qml.dot([0.25, 0.75], [qml.Z(2), qml.X(1) @ qml.X(2)])
    lcu2 = qml.dot([1 / 2, 1 / 2], [qml.Identity(0), qml.PauliZ(0)])

    a = 0.25
    b = 0.75
    A = np.array([[a, 0, 0, b], [0, -a, b, 0], [0, b, a, 0], [b, 0, 0, -a]])
    decomp = qml.pauli_decompose(A)
    coeffs, unitaries = decomp.terms()
    unitaries = [qml.map_wires(op, {0: 1, 1: 2}) for op in unitaries]
    lcu3 = qml.dot(coeffs, unitaries)

    lcu4 = qml.dot([-0.25, -0.75], [qml.Z(2), qml.X(1) @ qml.X(2)])
    lcu5 = qml.dot([1 + 0.25j, 0 - 0.75j], [qml.Z(3), qml.X(2) @ qml.X(3)])
    lcu6 = qml.dot([0.5, -0.5], [qml.Z(1), qml.X(1)])
    lcu7 = qml.dot([0.5, -0.5, 0 + 0.5j], [qml.Z(2), qml.X(2), qml.X(2)])
    lcu8 = qml.dot([0.5, 0.5j], [qml.X(1), qml.Z(1)])

    dev = qml.device("default.qubit")
    manual = qml.QNode(manual_circuit, dev)
    prepselprep = qml.QNode(prepselprep_circuit, dev)

    @pytest.mark.parametrize(
        ("lcu", "control", "produced_matrix", "expected_matrix"),
        [
            (
                lcu1,
                0,
                qml.matrix(prepselprep, wire_order=[0, 1, 2]),
                qml.matrix(manual, wire_order=[0, 1, 2]),
            ),
            (
                lcu2,
                "ancilla",
                qml.matrix(prepselprep, wire_order=["ancilla", 0]),
                qml.matrix(manual, wire_order=["ancilla", 0]),
            ),
            (
                lcu3,
                [0],
                qml.matrix(prepselprep, wire_order=[0, 1, 2]),
                qml.matrix(manual, wire_order=[0, 1, 2]),
            ),
            (
                lcu4,
                [0],
                qml.matrix(prepselprep, wire_order=[0, 1, 2]),
                qml.matrix(manual, wire_order=[0, 1, 2]),
            ),
            (
                lcu5,
                [0, 1],
                qml.matrix(prepselprep, wire_order=[0, 1, 2, 3]),
                qml.matrix(manual, wire_order=[0, 1, 2, 3]),
            ),
            (
                lcu6,
                [0],
                qml.matrix(prepselprep, wire_order=[0, 1, 2]),
                qml.matrix(manual, wire_order=[0, 1, 2]),
            ),
            (
                lcu7,
                [0, 1],
                qml.matrix(prepselprep, wire_order=[0, 1, 2]),
                qml.matrix(manual, wire_order=[0, 1, 2]),
            ),
            (
                lcu8,
                [0],
                qml.matrix(prepselprep, wire_order=[0, 1]),
                qml.matrix(manual, wire_order=[0, 1]),
            ),
        ],
    )
    def test_against_manual_circuit(self, lcu, control, produced_matrix, expected_matrix):
        """Test that the template produces the corrent decomposition"""

        assert qml.math.allclose(
            produced_matrix(lcu, control),
            expected_matrix(lcu, control),
        )

    @pytest.mark.parametrize(
        ("lcu", "control", "wire_order", "dim"),
        [
            (qml.dot([0.1, -0.4], [qml.Z(1), qml.X(1)]), [0], [0, 1], 2),
            (qml.dot([0.5, -0.5], [qml.Z(1), qml.X(1)]), [0], [0, 1], 2),
            (qml.dot([0.3, -0.1], [qml.Z(1), qml.X(1)]), [0], [0, 1], 2),
            (qml.dot([0.5j, -0.5j], [qml.Z(2), qml.X(2)]), [0, 1], [0, 1, 2], 2),
            (qml.dot([0.5, 0.5], [qml.Identity(0), qml.PauliZ(0)]), "ancilla", ["ancilla", 0], 2),
            (
                qml.dot([0.5, 0.5, 0.5], [qml.PauliX(2), qml.PauliY(2), qml.PauliZ(2)]),
                [0, 1],
                [0, 1, 2],
                2,
            ),
            (qml.dot([0.5, 0.5 + 0.5j], [qml.PauliX(2), qml.PauliY(2)]), [0, 1], [0, 1, 2], 2),
            (qml.dot([1.0, 0.0], [qml.PauliY(1), qml.PauliZ(1)]), [0], [0, 1], 2),
            (qml.dot([0.5, 0.5j], [qml.X(2), qml.Z(2)]), [0, 1], [0, 1, 2], 2),
            (qml.dot([1.0, 1.0, 1.0], [qml.X(2), qml.Y(2), qml.Z(2)]), [0, 1], [0, 1, 2], 2),
            (qml.dot([0.25, 0.75], [qml.I(1) @ qml.Z(2), qml.X(1) @ qml.X(2)]), [0], [0, 1, 2], 4),
            (qml.dot([0.25, 0.75], [qml.I(1) @ qml.Z(2), qml.Y(1) @ qml.Y(2)]), [0], [0, 1, 2], 4),
            (qml.dot([0.25, 0.75], [qml.Z(1), qml.Y(2) @ qml.Y(1)]), [0], [0, 1, 2], 4),
        ],
    )
    def test_block_encoding(self, lcu, control, wire_order, dim):
        """Test that the decomposition is a block-encoding"""
        dev = qml.device("default.qubit")
        prepselprep = qml.QNode(prepselprep_circuit, dev)
        matrix = qml.matrix(lcu)

        coeffs, _ = _get_new_terms(lcu)
        normalization_factor = qml.math.sum(coeffs)
        block_encoding = qml.matrix(prepselprep, wire_order=wire_order)(lcu, control=control)

        assert qml.math.allclose(matrix / normalization_factor, block_encoding[0:dim, 0:dim])

    lcu1 = qml.ops.LinearCombination([0.25, 0.75], [qml.Z(2), qml.X(1) @ qml.X(2)])
    ops1 = [qml.Z(2) @ qml.GlobalPhase(0), (qml.X(1) @ qml.X(2)) @ qml.GlobalPhase(0)]
    coeffs1 = lcu1.terms()[0]

    @pytest.mark.parametrize(
        ("lcu", "control", "results"),
        [
            (
                lcu1,
                [0],
                [
                    qml.AmplitudeEmbedding(
                        qml.math.sqrt(coeffs1), normalize=True, pad_with=0, wires=[0]
                    ),
                    qml.Select(ops1, control=[0]),
                    qml.ops.Adjoint(
                        qml.AmplitudeEmbedding(
                            qml.math.sqrt(coeffs1), normalize=True, pad_with=0, wires=[0]
                        )
                    ),
                ],
            )
        ],
    )
    def test_queuing_ops(self, lcu, control, results):
        """Test that qml.PrepSelPrep queues operations in the correct order."""
        with qml.tape.QuantumTape() as tape:
            qml.PrepSelPrep(lcu, control=control)

        for idx, val in enumerate(tape.expand().operations):
            assert val.name == results[idx].name
            assert len(val.parameters) == len(results[idx].parameters)
            for a, b in zip(val.parameters, results[idx].parameters):
                assert (a == b).all()

    def test_copy(self):
        """Test the copy function"""

        lcu = qml.dot([1 / 2, 1 / 2], [qml.Identity(1), qml.PauliZ(1)])
        op = qml.PrepSelPrep(lcu, control=0)
        op_copy = copy.copy(op)

        qml.assert_equal(op, op_copy)

    @pytest.mark.parametrize("lcu", a_set_of_lcus)
    def test_flatten_unflatten(self, lcu):
        """Test that the class can be correctly flattened and unflattened"""

        lcu_coeffs, lcu_ops = lcu.terms()

        op = qml.PrepSelPrep(lcu, control=0)
        data, metadata = op._flatten()

        data_coeffs, data_ops = data[0].terms()

        assert hash(metadata)

        assert len(data[0]) == len(lcu)
        assert all(coeff1 == coeff2 for coeff1, coeff2 in zip(lcu_coeffs, data_coeffs))
        assert all(op1 == op2 for op1, op2 in zip(lcu_ops, data_ops))

        assert metadata[0] == op.control

        new_op = type(op)._unflatten(*op._flatten())
        assert op.lcu == new_op.lcu
        assert all(coeff1 == coeff2 for coeff1, coeff2 in zip(op.coeffs, new_op.coeffs))
        assert all(qml.equal(op1, op2) for op1, op2 in zip(op.ops, new_op.ops))
        assert op.control == new_op.control
        assert op.wires == new_op.wires
        assert op.target_wires == new_op.target_wires
        assert op is not new_op

    @pytest.mark.parametrize("lcu", a_set_of_lcus)
    def test_label(self, lcu):
        """Test the custom label method of PrepSelPrep."""
        op = qml.PrepSelPrep(lcu, control=0)
        op_with_id = qml.PrepSelPrep(lcu, control=0, id="myID")

        # Default
        assert op.label() == "PrepSelPrep"
        assert op_with_id.label() == 'PrepSelPrep("myID")'

        # decimals do not affect label
        assert op.label(decimals=3) == "PrepSelPrep"
        assert op_with_id.label(decimals=3) == 'PrepSelPrep("myID")'

        # use different base label
        assert op.label(base_label="U(A)") == "U(A)"
        assert op_with_id.label(base_label="U(A)") == 'U(A)("myID")'

        # use cache without matrices
        assert op.label(cache={}) == "PrepSelPrep"
        assert op_with_id.label(cache={}) == 'PrepSelPrep("myID")'

        # use cache with empty matrices
        assert op.label(cache={"matrices": []}) == "PrepSelPrep(M0)"
        assert op_with_id.label(cache={"matrices": []}) == 'PrepSelPrep(M0,"myID")'

        # use cache with non-empty matrices
        assert op.label(cache={"matrices": [0.1]}) == "PrepSelPrep(M1)"
        assert op_with_id.label(cache={"matrices": [0.1, 0.6]}) == 'PrepSelPrep(M2,"myID")'

        # use cache with same matrix existing
        c = qml.math.array(op.coeffs)
        assert op.label(cache={"matrices": [0.1, c]}) == "PrepSelPrep(M1)"
        assert op_with_id.label(cache={"matrices": [c, 0.1, 0.6]}) == 'PrepSelPrep(M0,"myID")'


def test_control_in_ops():
    """Test that using an operation wire as a control wire results in an error"""

    lcu = qml.dot([1 / 2, 1 / 2], [qml.Identity(0), qml.PauliZ(0)])
    with pytest.raises(ValueError, match="Control wires should be different from operation wires."):
        qml.PrepSelPrep(lcu, control=0)


class TestInterfaces:
    """Tests that the template is compatible with interfaces used to compute gradients"""

    params = np.array([0.4, 0.5, 0.1, 0.3])
    exp_grad = [0.41177732, -0.21262349, 1.6437038, -0.74256516]

    @pytest.mark.torch
    def test_torch(self):
        """Test the torch interface"""
        import torch

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit(coeffs):
            H = qml.ops.LinearCombination(
                coeffs, [qml.Y(0), qml.Y(1) @ qml.Y(2), qml.X(0), qml.X(1) @ qml.X(2)]
            )
            qml.PrepSelPrep(H, control=(3, 4))
            return qml.expval(qml.PauliZ(3) @ qml.PauliZ(4))

        params = torch.tensor(self.params)
        res = torch.autograd.functional.jacobian(circuit, params)
        assert qml.math.shape(res) == (4,)
        assert np.allclose(res, self.exp_grad, atol=1e-5)

    @pytest.mark.autograd
    def test_autograd(self):
        """Test the autograd interface"""

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit(coeffs):
            H = qml.ops.LinearCombination(
                coeffs, [qml.Y(0), qml.Y(1) @ qml.Y(2), qml.X(0), qml.X(1) @ qml.X(2)]
            )
            qml.PrepSelPrep(H, control=(3, 4))
            return qml.expval(qml.PauliZ(3) @ qml.PauliZ(4))

        params = qml.numpy.array(self.params, requires_grad=True)
        res = qml.grad(circuit)(params)

        assert qml.math.shape(res) == (4,)
        assert np.allclose(res, self.exp_grad, atol=1e-5)

    @pytest.mark.jax
    def test_jax(self):
        """Test the jax interface"""
        import jax

        dev = qml.device("default.qubit")

        @qml.qnode(dev)
        def circuit(coeffs):
            H = qml.ops.LinearCombination(
                coeffs, [qml.Y(0), qml.Y(1) @ qml.Y(2), qml.X(0), qml.X(1) @ qml.X(2)]
            )
            qml.PrepSelPrep(H, control=(3, 4))
            return qml.expval(qml.PauliZ(3) @ qml.PauliZ(4))

        res = jax.grad(circuit)(self.params)

        assert qml.math.shape(res) == (4,)
        assert np.allclose(res, self.exp_grad, atol=1e-5)

    @pytest.mark.jax
    def test_jit(self):
        """Test that jax jit works"""
        import jax

        dev = qml.device("default.qubit")

        @jax.jit
        @qml.qnode(dev)
        def circuit(coeffs):
            H = qml.ops.LinearCombination(
                coeffs, [qml.Y(0), qml.Y(1) @ qml.Y(2), qml.X(0), qml.X(1) @ qml.X(2)]
            )
            qml.PrepSelPrep(H, control=(3, 4))
            return qml.expval(qml.PauliZ(3) @ qml.PauliZ(4))

        res = jax.grad(circuit)(self.params)

        assert qml.math.shape(res) == (4,)
        assert np.allclose(res, self.exp_grad, atol=1e-5)
