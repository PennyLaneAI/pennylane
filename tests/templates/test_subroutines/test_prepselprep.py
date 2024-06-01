# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=protected-access
"""
Tests for the PrepSelPrep template.
"""
import copy
import numpy as np
import pytest

import pennylane as qml
from pennylane import numpy as pnp


@pytest.mark.parametrize(
    ("lcu", "control"),
    [
        (qml.ops.LinearCombination([0.25, 0.75], [qml.Z(2), qml.X(1) @ qml.X(2)]), [0]),
        (qml.ops.LinearCombination([-0.25, 0.75j], [qml.Z(2), qml.X(1) @ qml.X(2)]), [0]),
        (qml.ops.LinearCombination([-0.25 + 0.1j, 0.75j], [qml.Z(3), qml.X(2) @ qml.X(3)]), [0, 1]),
    ],
)
def test_standard_checks(lcu, control):
    """Run standard validity tests."""

    op = qml.PrepSelPrep(lcu, control)
    qml.ops.functions.assert_valid(op)


def test_repr():
    """Test the repr method."""
    lcu = qml.dot([0.25, 0.75], [qml.Z(2), qml.X(1) @ qml.X(2)])
    control = [0]

    op = qml.PrepSelPrep(lcu, control)
    assert (
        repr(op)
        == "PrepSelPrep(coeffs=(0.25, 0.75), ops=(Z(2), X(1) @ X(2)), control=<Wires = [0]>)"
    )


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

    @staticmethod
    def manual_circuit(lcu, control):
        """Circuit equivalent to decomposition of PrepSelPrep"""
        coeffs, ops = qml.PrepSelPrep.preprocess_lcu(lcu)
        normalized_coeffs = qml.math.sqrt(qml.math.abs(coeffs)) / qml.math.norm(
            qml.math.sqrt(qml.math.abs(coeffs))
        )

        qml.StatePrep(normalized_coeffs, wires=control)
        qml.Select(ops, control=control)
        qml.adjoint(qml.StatePrep(normalized_coeffs, wires=control))

        return qml.state()

    @staticmethod
    def prepselprep_circuit(lcu, control):
        """PrepSelPrep circuit used for testing"""

        qml.PrepSelPrep(lcu, control)
        return qml.state()

    # Use these circuits in tests
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
               'ancilla',
               qml.matrix(prepselprep, wire_order=['ancilla', 0]),
               qml.matrix(manual, wire_order=['ancilla', 0]),
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
        ],
    )
    def test_decomposition(self, lcu, control, produced_matrix, expected_matrix):
        """Test that the template produces the corrent decomposition"""

        assert np.array_equal(
            produced_matrix(lcu, control),
            expected_matrix(lcu, control),
        )

    @pytest.mark.parametrize(
        ("lcu", "control", "wire_order", "dim"),
        [
            (qml.dot([0.5, -0.5], [qml.Z(1), qml.X(1)]), [0], [0, 1], 2),
            (qml.dot([0.5j, -0.5j], [qml.Z(1), qml.X(1)]), [0], [0, 1], 2),
            (qml.dot([0.5, 0.5], [qml.Identity(0), qml.PauliZ(0)]), "ancilla", ["ancilla", 0], 2),
        ],
    )
    def test_block_encoding(self, lcu, control, wire_order, dim):
        """Test that the decomposition is a block-encoding"""
        dev = qml.device("default.qubit")
        prepselprep = qml.QNode(TestPrepSelPrep.prepselprep_circuit, dev)
        matrix = qml.matrix(lcu)
        block_encoding = qml.matrix(prepselprep, wire_order=wire_order)(lcu, control=control)
        assert qml.math.allclose(matrix, block_encoding[0:dim, 0:dim])

    ops1 = lcu1.terms()[1]
    coeffs1 = lcu1.terms()[0]
    normalized1 = qml.math.sqrt(coeffs1) / qml.math.norm(qml.math.sqrt(coeffs1))

    @pytest.mark.parametrize(
        ("lcu", "control", "results"),
        [
            (
                lcu1,
                [0],
                [
                    qml.MottonenStatePreparation(normalized1, wires=[0]),
                    qml.ctrl(
                        qml.ops.QubitUnitary(
                            qml.math.array([[0.0 + 0j, 0.0 + 0.0j], [0.0 + 0.0j, -1.0 + 0.0j]]),
                            wires=[2],
                        ),
                        0,
                    ),
                    qml.ctrl(
                        qml.ops.QubitUnitary(
                            qml.math.array(
                                [
                                    [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j],
                                    [0.0 + 0.0j, 0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j],
                                    [0.0 + 0.0j, 1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                                    [1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
                                ]
                            ),
                            wires=[1, 2],
                        ),
                        0,
                    ),
                    qml.ops.Adjoint(qml.MottonenStatePreparation(normalized1, wires=[0])),
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
            assert all([a == b] for a, b in zip(val.parameters, results[idx].parameters))

    def test_copy(self):
        """Test the copy function"""

        lcu = qml.dot([1 / 2, 1 / 2], [qml.Identity(1), qml.PauliZ(1)])
        op = qml.PrepSelPrep(lcu, control=0)
        op_copy = copy.copy(op)

        assert qml.equal(op, op_copy)

    def test_flatten_unflatten(self):
        """Test that the class can be correctly flattened and unflattened"""

        lcu = qml.ops.LinearCombination([1 / 2, 1 / 2], [qml.Identity(1), qml.PauliZ(1)])
        lcu_coeffs, lcu_ops = lcu.terms()

        op = qml.PrepSelPrep(lcu, control=0)
        data, metadata = op._flatten()

        data_coeffs = [term.terms()[0][0] for term in data]
        data_ops = [term.terms()[1][0] for term in data]

        assert hash(metadata)

        assert len(data) == len(lcu)
        assert all(coeff1 == coeff2 for coeff1, coeff2 in zip(lcu_coeffs, data_coeffs))
        assert all(op1 == op2 for op1, op2 in zip(lcu_ops, data_ops))

        assert metadata == op.control

        new_op = type(op)._unflatten(*op._flatten())
        assert op.lcu == new_op.lcu
        assert all(coeff1 == coeff2 for coeff1, coeff2 in zip(op.coeffs, new_op.coeffs))
        assert all(qml.equal(op1, op2) for op1, op2 in zip(op.ops, new_op.ops))
        assert op.control == new_op.control
        assert op.wires == new_op.wires
        assert op.target_wires == new_op.target_wires
        assert op is not new_op


class TestErrors:
    """Tests that the template propertly raises errors"""

    def test_control_in_ops(self):
        """Test that using an operation wire as a control wire results in an error"""

        lcu = qml.dot([1 / 2, 1 / 2], [qml.Identity(0), qml.PauliZ(0)])
        with pytest.raises(ValueError, match="Control wires should be different from operation wires."):
            qml.PrepSelPrep(lcu, control=0)

    def test_jit_not_power_2(self):
        """Test that calling jit with a non-power of 2 number of terms results in an error"""

        lcu = qml.dot([1, 2, 3], [qml.Identity(1), qml.PauliZ(1), qml.PauliX(1)])
        with pytest.raises(ValueError, match="Number of terms must be a power of 2."):
            qml.PrepSelPrep(lcu, control=0, jit=True).decomposition()

    def test_jit_complex_coeff(self):
        """Test that calling jit with complex coefficients results in an error"""
        lcu = qml.dot([0.5j, 1], [qml.PauliZ(1), qml.PauliX(1)])
        with pytest.raises(ValueError, match="Coefficients must be positive real numbers."):
            qml.PrepSelPrep(lcu, control=0, jit=True).decomposition()

    def test_jit_negative_coeff(self):
        """Test that calling jit with complex coefficients results in an error"""
        import jax.numpy as jnp

        coeffs = jnp.array([0.5, -1])
        lcu = qml.ops.LinearCombination(coeffs, [qml.PauliZ(1), qml.PauliX(1)])
        with pytest.raises(ValueError, match="Coefficients must be positive real numbers."):
            qml.PrepSelPrep(lcu, control=0, jit=True).decomposition()

class TestInterfaces:
    """Tests that the template is compatible with interfaces used to compute gradients"""

    @pytest.mark.autograd
    def test_autograd(self):
        """Test the autograd interface"""

        dev = qml.device("default.qubit")

        coeffs = pnp.array([1 / 2, 1 / 2], requires_grad=True)
        ops = [qml.Identity(1), qml.PauliZ(1)]

        @qml.qnode(dev, diff_method="finite-diff")
        def prepselprep(coeffs):
            lcu = qml.ops.LinearCombination(coeffs, ops)
            qml.PrepSelPrep(lcu, control=0)
            return qml.expval(qml.Z(1))

        @qml.qnode(dev)
        def manual(coeffs):
            normalized_coeffs = qml.math.sqrt(coeffs) / qml.math.norm(qml.math.sqrt(coeffs))

            qml.StatePrep(normalized_coeffs, wires=0)
            qml.Select(ops, control=0)
            qml.adjoint(qml.StatePrep(normalized_coeffs, wires=0))
            return qml.expval(qml.Z(1))

        grad_prepselprep = qml.grad(prepselprep)(coeffs)
        grad_manual = qml.grad(manual)(coeffs)

        assert qml.math.allclose(grad_prepselprep, grad_manual)

        fd_prepselprep = qml.jacobian(prepselprep)(coeffs)
        assert qml.math.allclose(grad_prepselprep, fd_prepselprep)

    @pytest.mark.jax
    def test_jax(self):
        """Test the jax interface"""
        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit")
        coeffs = jnp.array([1 / 2, 1 / 2])
        ops = [qml.Identity(1), qml.PauliZ(1)]

        @qml.qnode(dev, diff_method="finite-diff")
        def prepselprep(coeffs):
            lcu = qml.ops.LinearCombination(coeffs, ops)
            qml.PrepSelPrep(lcu, control=0)
            return qml.expval(qml.Z(1))

        @qml.qnode(dev)
        def manual(coeffs):
            normalized_coeffs = qml.math.sqrt(coeffs) / qml.math.norm(qml.math.sqrt(coeffs))

            qml.StatePrep(normalized_coeffs, wires=0)
            qml.Select(ops, control=0)
            qml.adjoint(qml.StatePrep(normalized_coeffs, wires=0))
            return qml.expval(qml.Z(1))

        grad_prepselprep = jax.grad(prepselprep)(coeffs)
        grad_manual = jax.grad(manual)(coeffs)

        assert qml.math.allclose(grad_prepselprep, grad_manual)

        fd_prepselprep = jax.jacobian(prepselprep)(coeffs)
        assert qml.math.allclose(grad_prepselprep, fd_prepselprep)

    @pytest.mark.jax
    def test_jax_jit(self):
        """Test the jax interface with jit"""
        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit")
        coeffs = jnp.array([1 / 2, 1 / 2])
        ops = [qml.Identity(1), qml.PauliZ(1)]

        @jax.jit
        @qml.qnode(dev)
        def prepselprep(coeffs):
            lcu = qml.ops.LinearCombination(coeffs, ops)
            qml.PrepSelPrep(lcu, control=0, jit=True)
            return qml.expval(qml.Z(1))

        @jax.jit
        @qml.qnode(dev)
        def manual(coeffs):
            normalized_coeffs = qml.math.sqrt(coeffs) / qml.math.norm(qml.math.sqrt(coeffs))

            qml.StatePrep(normalized_coeffs, wires=0)
            qml.Select(ops, control=0)
            qml.adjoint(qml.StatePrep(normalized_coeffs, wires=0))
            return qml.expval(qml.Z(1))

        grad_prepselprep = jax.grad(prepselprep)(coeffs)
        grad_manual = jax.grad(manual)(coeffs)

        assert qml.math.allclose(grad_prepselprep, grad_manual)

    @pytest.mark.jax
    def test_jit_with_preprocessing(self):
        """Test that jit works with the preprocessed output"""
        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit")
        coeffs = jnp.array([0.5, -0.5, 0.5j])
        ops = [qml.I(2), qml.Y(2), qml.Z(2)]
        lcu = qml.ops.LinearCombination(coeffs, ops)

        preprocessed_coeffs, preprocessed_ops = qml.PrepSelPrep.preprocess_lcu(lcu)
        assert len(preprocessed_coeffs) == len(preprocessed_ops)

        @jax.jit
        @qml.qnode(dev)
        def circuit(coeffs):

            lcu = qml.ops.LinearCombination(coeffs, preprocessed_ops)
            qml.PrepSelPrep(lcu, control=[0, 1], jit=True)
            return qml.state()

        circuit(preprocessed_coeffs)
