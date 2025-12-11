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


@pytest.mark.jax
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
        assert repr(op) == "PrepSelPrep(lcu=0.25 * Z(2) + 0.75 * (X(1) @ X(2)), control=Wires([0]))"


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
    qml.Select(ops, control=control, partial=True)
    qml.adjoint(
        qml.AmplitudeEmbedding(qml.math.sqrt(coeffs), normalize=True, pad_with=0, wires=control)
    )


def prepselprep_circuit(lcu, control):
    """PrepSelPrep circuit used for testing"""
    qml.PrepSelPrep(lcu, control)


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

    @pytest.mark.parametrize(
        ("lcu", "control", "wire_order"),
        [
            (lcu1, 0, [0, 1, 2]),
            (lcu2, "aux", ["aux", 0]),
            (lcu3, [0], [0, 1, 2]),
            (lcu4, [0], [0, 1, 2]),
            (lcu5, [0, 1], [0, 1, 2, 3]),
            (lcu6, [0], [0, 1, 2]),
            (lcu7, [0, 1], [0, 1, 2]),
            (lcu8, [0], [0, 1]),
        ],
    )
    def test_against_manual_circuit(self, lcu, control, wire_order):
        """Test that the template produces the corrent decomposition"""

        assert qml.math.allclose(
            qml.matrix(prepselprep_circuit, wire_order=wire_order)(lcu, control),
            qml.matrix(manual_circuit, wire_order=wire_order)(lcu, control),
        )

    @pytest.mark.parametrize(
        ("lcu", "control", "wire_order", "dim"),
        [
            (qml.dot([0.1, -0.4], [qml.Z(1), qml.X(1)]), [0], [0, 1], 2),
            (qml.dot([0.5, -0.5], [qml.Z(1), qml.X(1)]), [0], [0, 1], 2),
            (qml.dot([0.3, -0.1], [qml.Z(1), qml.X(1)]), [0], [0, 1], 2),
            (qml.dot([0.5j, -0.5j], [qml.Z(2), qml.X(2)]), [0, 1], [0, 1, 2], 2),
            (
                qml.dot([0.5, 0.5], [qml.Identity(0), qml.PauliZ(0)]),
                "auxiliary",
                ["auxiliary", 0],
                2,
            ),
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
        matrix = qml.matrix(lcu)

        coeffs, _ = _get_new_terms(lcu)
        normalization_factor = qml.math.sum(coeffs)
        block_encoding = qml.matrix(prepselprep_circuit, wire_order=wire_order)(
            lcu, control=control
        )

        assert qml.math.allclose(matrix / normalization_factor, block_encoding[0:dim, 0:dim])

    lcu1 = qml.ops.LinearCombination([0.25, 0.75], [qml.Z(2), qml.X(1) @ qml.X(2)])
    ops1 = [
        qml.Z(2) @ qml.GlobalPhase(0, [2]),
        qml.prod(qml.X(1) @ qml.X(2), qml.GlobalPhase(0, [1, 2])),
    ]
    coeffs1 = lcu1.terms()[0]

    @pytest.mark.parametrize(
        ("lcu", "control", "expected"),
        [
            (
                lcu1,
                [0],
                [
                    qml.ops.ChangeOpBasis(
                        qml.AmplitudeEmbedding(
                            qml.math.sqrt(coeffs1), normalize=True, pad_with=0, wires=[0]
                        ),
                        qml.Select(ops1, control=[0]),
                    )
                ],
            )
        ],
    )
    def test_queuing_ops(self, lcu, control, expected):
        """Test that qml.PrepSelPrep queues operations in the correct order."""
        # Test that `compute_decomposition` queues the right ops
        prepselprep = qml.PrepSelPrep(lcu, control=control)
        with qml.queuing.AnnotatedQueue() as q0:
            prepselprep.compute_decomposition(lcu, control)

        # Test that `compute_decomposition` queues the right ops
        with qml.queuing.AnnotatedQueue() as q1:
            prepselprep.decomposition()

        for op0, op1, exp_op in zip(q0.queue, q1.queue, expected, strict=True):
            qml.assert_equal(op0, exp_op)
            qml.assert_equal(op1, exp_op)

        # Test that PrepSelPrep de-queues its input
        with qml.queuing.AnnotatedQueue() as q2:
            op = qml.apply(lcu)
            prepselprep = qml.PrepSelPrep(op, control=control)

        assert len(q2.queue) == 1 and q2.queue[0] == prepselprep

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

    def test_resources(self):
        """Test the registered resources."""

        assert qml.PrepSelPrep.resource_keys == frozenset({"num_control", "op_reps"})

        ops = [qml.X(0), qml.X(1), qml.X(0) @ qml.Y(1)]
        lcu = qml.dot([1, 2, 3], ops)
        op = qml.PrepSelPrep(lcu, (3, 4))

        op_reps = (
            qml.resource_rep(qml.X),
            qml.resource_rep(qml.X),
            qml.resource_rep(qml.ops.Prod, **ops[-1].resource_params),
        )
        assert op.resource_params == {"num_control": 2, "op_reps": op_reps}

    def test_decomposition_new_structure(self):
        """Test that the decomposition is registered into the new pipeline."""

        ops = [qml.X(0), qml.X(1), qml.X(0) @ qml.Y(1)]
        grep = qml.resource_rep(qml.GlobalPhase)
        xrep = qml.resource_rep(qml.X)
        yrep = qml.resource_rep(qml.Y)
        prodrep = qml.resource_rep(qml.ops.Prod, resources={xrep: 1, yrep: 1})
        op_reps = (
            qml.resource_rep(qml.ops.Prod, resources={grep: 1, xrep: 1}),
            qml.resource_rep(qml.ops.Prod, resources={grep: 1, xrep: 1}),
            qml.resource_rep(qml.ops.Prod, resources={grep: 1, prodrep: 1}),
        )
        lcu = qml.dot([1, 4, 9], ops)
        op = qml.PrepSelPrep(lcu, (3, 4))

        decomp = qml.list_decomps(qml.PrepSelPrep)[0]

        resource_obj = decomp.compute_resources(**op.resource_params)
        assert resource_obj.num_gates == 1

        expected_counts = {
            qml.resource_rep(
                qml.Select, op_reps=op_reps, num_control_wires=2, partial=True, num_work_wires=0
            ): 1,
            qml.resource_rep(qml.StatePrep, num_wires=2): 1,
            qml.resource_rep(
                qml.ops.Adjoint, base_class=qml.StatePrep, base_params={"num_wires": 2}
            ): 1,
        }
        expected_counts = {
            qml.resource_rep(
                qml.ops.ChangeOpBasis,
                compute_op=qml.resource_rep(qml.StatePrep, num_wires=2),
                target_op=qml.resource_rep(
                    qml.Select, op_reps=op_reps, num_control_wires=2, partial=True, num_work_wires=0
                ),
                uncompute_op=qml.resource_rep(
                    qml.ops.Adjoint, base_class=qml.StatePrep, base_params={"num_wires": 2}
                ),
            ): 1,
        }

        assert resource_obj.gate_counts == expected_counts

        decomp = qml.list_decomps(qml.PrepSelPrep)[0]

        with qml.queuing.AnnotatedQueue() as q:
            decomp(*op.data, wires=op.wires, **op.hyperparameters)

        q = q.queue[0].decomposition()

        phase_ops = [qml.prod(op, qml.GlobalPhase(0, wires=op.wires)) for op in ops]

        prep = qml.StatePrep(np.array([1, 2, 3]), normalize=True, pad_with=0, wires=(3, 4))
        qml.assert_equal(q[0], prep)
        qml.assert_equal(q[1], qml.Select(phase_ops, (3, 4)))
        qml.assert_equal(q[2], qml.adjoint(prep))


def test_control_in_ops():
    """Test that using an operation wire as a control wire results in an error"""

    lcu = qml.dot([1 / 2, 1 / 2], [qml.Identity(0), qml.PauliZ(0)])
    with pytest.raises(ValueError, match="Control wires should be different from operation wires."):
        qml.PrepSelPrep(lcu, control=0)


class TestInterfaces:
    """Tests that the template is compatible with interfaces used to compute gradients"""

    params = np.array([0.4, 0.5, 0.1, 0.3])
    # TODO: We really shouldn't be hardcoding the expected derivative here [sc-98529]
    exp_grad = [-0.57485039, 0.31253535, -0.717947, 0.48489061]

    @pytest.mark.torch
    def test_torch(self):
        """Test the torch interface"""
        import torch

        dev = qml.device("reference.qubit", wires=5)

        @qml.qnode(dev)
        def circuit(coeffs):
            H = qml.ops.LinearCombination(
                coeffs, [qml.Y(0), qml.Y(1) @ qml.Y(2), qml.X(0), -1 * qml.X(1) @ qml.X(2)]
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

        dev = qml.device("reference.qubit", wires=5)

        @qml.qnode(dev)
        def circuit(coeffs):
            H = qml.ops.LinearCombination(
                coeffs, [qml.Y(0), qml.Y(1) @ qml.Y(2), qml.X(0), -1 * qml.X(1) @ qml.X(2)]
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

        dev = qml.device("reference.qubit", wires=5)

        @qml.qnode(dev)
        def circuit(coeffs):
            H = qml.ops.LinearCombination(
                coeffs, [qml.Y(0), qml.Y(1) @ qml.Y(2), qml.X(0), -1 * qml.X(1) @ qml.X(2)]
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

        dev = qml.device("reference.qubit", wires=5)

        @jax.jit
        @qml.qnode(dev)
        def circuit(coeffs):
            H = qml.ops.LinearCombination(
                coeffs, [qml.Y(0), qml.Y(1) @ qml.Y(2), qml.X(0), -1 * qml.X(1) @ qml.X(2)]
            )
            qml.PrepSelPrep(H, control=(3, 4))
            return qml.expval(qml.PauliZ(3) @ qml.PauliZ(4))

        res = jax.grad(circuit)(self.params)

        assert qml.math.shape(res) == (4,)
        assert np.allclose(res, self.exp_grad, atol=1e-5)
