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

import pennylane as qp


@pytest.mark.jax
@pytest.mark.parametrize(
    ("lcu", "control", "skip_diff"),
    [
        (qp.ops.LinearCombination([0.25, 0.75], [qp.Z(2), qp.X(1) @ qp.X(2)]), [0], False),
        (qp.dot([0.25, 0.75], [qp.Z(2), qp.X(1) @ qp.X(2)]), [0], False),
        (qp.Hamiltonian([0.25, 0.75], [qp.Z(2), qp.X(1) @ qp.X(2)]), [0], False),
        (0.25 * qp.Z(2) - 0.75 * qp.X(1) @ qp.X(2), [0], False),
        (qp.Z(2) + qp.X(1) @ qp.X(2), [0], False),
        (qp.ops.LinearCombination([-0.25, 0.75j], [qp.Z(3), qp.X(2) @ qp.X(3)]), [0, 1], True),
        (
            qp.ops.LinearCombination([-0.25 + 0.1j, 0.75j], [qp.Z(4), qp.X(4) @ qp.X(5)]),
            [0, 1, 2, 3],
            True,
        ),
    ],
)
def test_standard_checks(lcu, control, skip_diff):
    """Run standard validity tests."""

    op = qp.PrepSelPrep(lcu, control)
    # Skip differentiation for test cases that raise NaNs in gradients (known limitation of ``MottonenStatePreparation``).
    qp.ops.functions.assert_valid(op, skip_differentiation=skip_diff)


def test_repr():
    """Test the repr method."""

    lcu = qp.dot([0.25, 0.75], [qp.Z(2), qp.X(1) @ qp.X(2)])
    control = [0]

    op = qp.PrepSelPrep(lcu, control)
    with np.printoptions(legacy="1.21"):
        assert repr(op) == "PrepSelPrep(lcu=0.25 * Z(2) + 0.75 * (X(1) @ X(2)), control=Wires([0]))"


def _get_new_terms(lcu):
    """Compute a new sum of unitaries with positive coefficients"""

    new_coeffs = []
    new_ops = []

    for coeff, op in zip(*lcu.terms()):

        angle = qp.math.angle(coeff)
        new_coeffs.append(qp.math.abs(coeff))

        new_op = op @ qp.GlobalPhase(-angle, wires=op.wires)
        new_ops.append(new_op)

    interface = qp.math.get_interface(lcu.terms()[0])
    new_coeffs = qp.math.array(new_coeffs, like=interface)

    return new_coeffs, new_ops


# Use these circuits in tests
def manual_circuit(lcu, control):
    """Circuit equivalent to decomposition of PrepSelPrep"""
    coeffs, ops = _get_new_terms(lcu)

    qp.AmplitudeEmbedding(qp.math.sqrt(coeffs), normalize=True, pad_with=0, wires=control)
    qp.Select(ops, control=control, partial=True)
    qp.adjoint(
        qp.AmplitudeEmbedding(qp.math.sqrt(coeffs), normalize=True, pad_with=0, wires=control)
    )


def prepselprep_circuit(lcu, control):
    """PrepSelPrep circuit used for testing"""
    qp.PrepSelPrep(lcu, control)


a_set_of_lcus = [
    qp.ops.LinearCombination([0.25, 0.75], [qp.Z(2), qp.X(1) @ qp.X(2)]),
    qp.dot([0.25, 0.75], [qp.Z(2), qp.X(1) @ qp.X(2)]),
    qp.Hamiltonian([0.25, 0.75], [qp.Z(2), qp.X(1) @ qp.X(2)]),
    0.25 * qp.Z(2) - 0.75 * qp.X(1) @ qp.X(2),
    qp.Z(2) + qp.X(1) @ qp.X(2),
    qp.ops.LinearCombination([-0.25, 0.75j], [qp.Z(3), qp.X(2) @ qp.X(3)]),
    qp.ops.LinearCombination([-0.25 + 0.1j, 0.75j], [qp.Z(4), qp.X(4) @ qp.X(5)]),
]


class TestPrepSelPrep:
    """Test the correctness of the decomposition"""

    # Use these LCUs as test input
    lcu1 = qp.dot([0.25, 0.75], [qp.Z(2), qp.X(1) @ qp.X(2)])
    lcu2 = qp.dot([1 / 2, 1 / 2], [qp.Identity(0), qp.PauliZ(0)])

    a = 0.25
    b = 0.75
    A = np.array([[a, 0, 0, b], [0, -a, b, 0], [0, b, a, 0], [b, 0, 0, -a]])
    decomp = qp.pauli_decompose(A)
    coeffs, unitaries = decomp.terms()
    unitaries = [qp.map_wires(op, {0: 1, 1: 2}) for op in unitaries]
    lcu3 = qp.dot(coeffs, unitaries)

    lcu4 = qp.dot([-0.25, -0.75], [qp.Z(2), qp.X(1) @ qp.X(2)])
    lcu5 = qp.dot([1 + 0.25j, 0 - 0.75j], [qp.Z(3), qp.X(2) @ qp.X(3)])
    lcu6 = qp.dot([0.5, -0.5], [qp.Z(1), qp.X(1)])
    lcu7 = qp.dot([0.5, -0.5, 0 + 0.5j], [qp.Z(2), qp.X(2), qp.X(2)])
    lcu8 = qp.dot([0.5, 0.5j], [qp.X(1), qp.Z(1)])

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

        assert qp.math.allclose(
            qp.matrix(prepselprep_circuit, wire_order=wire_order)(lcu, control),
            qp.matrix(manual_circuit, wire_order=wire_order)(lcu, control),
        )

    @pytest.mark.parametrize(
        ("lcu", "control", "wire_order", "dim"),
        [
            (qp.dot([0.1, -0.4], [qp.Z(1), qp.X(1)]), [0], [0, 1], 2),
            (qp.dot([0.5, -0.5], [qp.Z(1), qp.X(1)]), [0], [0, 1], 2),
            (qp.dot([0.3, -0.1], [qp.Z(1), qp.X(1)]), [0], [0, 1], 2),
            (qp.dot([0.5j, -0.5j], [qp.Z(2), qp.X(2)]), [0, 1], [0, 1, 2], 2),
            (
                qp.dot([0.5, 0.5], [qp.Identity(0), qp.PauliZ(0)]),
                "auxiliary",
                ["auxiliary", 0],
                2,
            ),
            (
                qp.dot([0.5, 0.5, 0.5], [qp.PauliX(2), qp.PauliY(2), qp.PauliZ(2)]),
                [0, 1],
                [0, 1, 2],
                2,
            ),
            (qp.dot([0.5, 0.5 + 0.5j], [qp.PauliX(2), qp.PauliY(2)]), [0, 1], [0, 1, 2], 2),
            (qp.dot([1.0, 0.0], [qp.PauliY(1), qp.PauliZ(1)]), [0], [0, 1], 2),
            (qp.dot([0.5, 0.5j], [qp.X(2), qp.Z(2)]), [0, 1], [0, 1, 2], 2),
            (qp.dot([1.0, 1.0, 1.0], [qp.X(2), qp.Y(2), qp.Z(2)]), [0, 1], [0, 1, 2], 2),
            (qp.dot([0.25, 0.75], [qp.I(1) @ qp.Z(2), qp.X(1) @ qp.X(2)]), [0], [0, 1, 2], 4),
            (qp.dot([0.25, 0.75], [qp.I(1) @ qp.Z(2), qp.Y(1) @ qp.Y(2)]), [0], [0, 1, 2], 4),
            (qp.dot([0.25, 0.75], [qp.Z(1), qp.Y(2) @ qp.Y(1)]), [0], [0, 1, 2], 4),
        ],
    )
    def test_block_encoding(self, lcu, control, wire_order, dim):
        """Test that the decomposition is a block-encoding"""
        matrix = qp.matrix(lcu)

        coeffs, _ = _get_new_terms(lcu)
        normalization_factor = qp.math.sum(coeffs)
        block_encoding = qp.matrix(prepselprep_circuit, wire_order=wire_order)(
            lcu, control=control
        )

        assert qp.math.allclose(matrix / normalization_factor, block_encoding[0:dim, 0:dim])

    lcu1 = qp.ops.LinearCombination([0.25, 0.75], [qp.Z(2), qp.X(1) @ qp.X(2)])
    ops1 = [
        qp.Z(2) @ qp.GlobalPhase(0, [2]),
        qp.prod(qp.X(1) @ qp.X(2), qp.GlobalPhase(0, [1, 2])),
    ]
    coeffs1 = lcu1.terms()[0]

    @pytest.mark.parametrize(
        ("lcu", "control", "expected"),
        [
            (
                lcu1,
                [0],
                [
                    qp.ops.ChangeOpBasis(
                        qp.AmplitudeEmbedding(
                            qp.math.sqrt(coeffs1), normalize=True, pad_with=0, wires=[0]
                        ),
                        qp.Select(ops1, control=[0]),
                    )
                ],
            )
        ],
    )
    def test_queuing_ops(self, lcu, control, expected):
        """Test that qp.PrepSelPrep queues operations in the correct order."""
        # Test that `compute_decomposition` queues the right ops
        prepselprep = qp.PrepSelPrep(lcu, control=control)
        with qp.queuing.AnnotatedQueue() as q0:
            prepselprep.compute_decomposition(lcu, control)

        # Test that `compute_decomposition` queues the right ops
        with qp.queuing.AnnotatedQueue() as q1:
            prepselprep.decomposition()

        for op0, op1, exp_op in zip(q0.queue, q1.queue, expected, strict=True):
            qp.assert_equal(op0, exp_op)
            qp.assert_equal(op1, exp_op)

        # Test that PrepSelPrep de-queues its input
        with qp.queuing.AnnotatedQueue() as q2:
            op = qp.apply(lcu)
            prepselprep = qp.PrepSelPrep(op, control=control)

        assert len(q2.queue) == 1 and q2.queue[0] == prepselprep

    def test_copy(self):
        """Test the copy function"""

        lcu = qp.dot([1 / 2, 1 / 2], [qp.Identity(1), qp.PauliZ(1)])
        op = qp.PrepSelPrep(lcu, control=0)
        op_copy = copy.copy(op)

        qp.assert_equal(op, op_copy)

    @pytest.mark.parametrize("lcu", a_set_of_lcus)
    def test_flatten_unflatten(self, lcu):
        """Test that the class can be correctly flattened and unflattened"""

        lcu_coeffs, lcu_ops = lcu.terms()

        op = qp.PrepSelPrep(lcu, control=0)
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
        assert all(qp.equal(op1, op2) for op1, op2 in zip(op.ops, new_op.ops))
        assert op.control == new_op.control
        assert op.wires == new_op.wires
        assert op.target_wires == new_op.target_wires
        assert op is not new_op

    @pytest.mark.parametrize("lcu", a_set_of_lcus)
    def test_label(self, lcu):
        """Test the custom label method of PrepSelPrep."""
        op = qp.PrepSelPrep(lcu, control=0)
        op_with_id = qp.PrepSelPrep(lcu, control=0, id="myID")

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
        c = qp.math.array(op.coeffs)
        assert op.label(cache={"matrices": [0.1, c]}) == "PrepSelPrep(M1)"
        assert op_with_id.label(cache={"matrices": [c, 0.1, 0.6]}) == 'PrepSelPrep(M0,"myID")'

    def test_resources(self):
        """Test the registered resources."""

        assert qp.PrepSelPrep.resource_keys == frozenset({"num_control", "op_reps"})

        ops = [qp.X(0), qp.X(1), qp.X(0) @ qp.Y(1)]
        lcu = qp.dot([1, 2, 3], ops)
        op = qp.PrepSelPrep(lcu, (3, 4))

        op_reps = (
            qp.resource_rep(qp.X),
            qp.resource_rep(qp.X),
            qp.resource_rep(qp.ops.Prod, **ops[-1].resource_params),
        )
        assert op.resource_params == {"num_control": 2, "op_reps": op_reps}

    def test_decomposition_new_structure(self):
        """Test that the decomposition is registered into the new pipeline."""

        ops = [qp.X(0), qp.X(1), qp.X(0) @ qp.Y(1)]
        grep = qp.resource_rep(qp.GlobalPhase)
        xrep = qp.resource_rep(qp.X)
        yrep = qp.resource_rep(qp.Y)
        prodrep = qp.resource_rep(qp.ops.Prod, resources={xrep: 1, yrep: 1})
        op_reps = (
            qp.resource_rep(qp.ops.Prod, resources={grep: 1, xrep: 1}),
            qp.resource_rep(qp.ops.Prod, resources={grep: 1, xrep: 1}),
            qp.resource_rep(qp.ops.Prod, resources={grep: 1, prodrep: 1}),
        )
        lcu = qp.dot([1, 4, 9], ops)
        op = qp.PrepSelPrep(lcu, (3, 4))

        decomp = qp.list_decomps(qp.PrepSelPrep)[0]

        resource_obj = decomp.compute_resources(**op.resource_params)
        assert resource_obj.num_gates == 1

        expected_counts = {
            qp.resource_rep(
                qp.Select, op_reps=op_reps, num_control_wires=2, partial=True, num_work_wires=0
            ): 1,
            qp.resource_rep(qp.StatePrep, num_wires=2): 1,
            qp.resource_rep(
                qp.ops.Adjoint, base_class=qp.StatePrep, base_params={"num_wires": 2}
            ): 1,
        }
        expected_counts = {
            qp.resource_rep(
                qp.ops.ChangeOpBasis,
                compute_op=qp.resource_rep(qp.StatePrep, num_wires=2),
                target_op=qp.resource_rep(
                    qp.Select, op_reps=op_reps, num_control_wires=2, partial=True, num_work_wires=0
                ),
                uncompute_op=qp.resource_rep(
                    qp.ops.Adjoint, base_class=qp.StatePrep, base_params={"num_wires": 2}
                ),
            ): 1,
        }

        assert resource_obj.gate_counts == expected_counts

        decomp = qp.list_decomps(qp.PrepSelPrep)[0]

        with qp.queuing.AnnotatedQueue() as q:
            decomp(*op.data, wires=op.wires, **op.hyperparameters)

        q = q.queue[0].decomposition()

        phase_ops = [qp.prod(op, qp.GlobalPhase(0, wires=op.wires)) for op in ops]

        prep = qp.StatePrep(np.array([1, 2, 3]), normalize=True, pad_with=0, wires=(3, 4))
        qp.assert_equal(q[0], prep)
        qp.assert_equal(q[1], qp.Select(phase_ops, (3, 4)))
        qp.assert_equal(q[2], qp.adjoint(prep))


def test_control_in_ops():
    """Test that using an operation wire as a control wire results in an error"""

    lcu = qp.dot([1 / 2, 1 / 2], [qp.Identity(0), qp.PauliZ(0)])
    with pytest.raises(ValueError, match="Control wires should be different from operation wires."):
        qp.PrepSelPrep(lcu, control=0)


class TestInterfaces:
    """Tests that the template is compatible with interfaces used to compute gradients"""

    params = np.array([0.4, 0.5, 0.1, 0.3])
    # TODO: We really shouldn't be hardcoding the expected derivative here [sc-98529]
    exp_grad = [-0.57485039, 0.31253535, -0.717947, 0.48489061]

    @pytest.mark.torch
    def test_torch(self):
        """Test the torch interface"""
        import torch

        dev = qp.device("reference.qubit", wires=5)

        @qp.qnode(dev)
        def circuit(coeffs):
            H = qp.ops.LinearCombination(
                coeffs, [qp.Y(0), qp.Y(1) @ qp.Y(2), qp.X(0), -1 * qp.X(1) @ qp.X(2)]
            )
            qp.PrepSelPrep(H, control=(3, 4))
            return qp.expval(qp.PauliZ(3) @ qp.PauliZ(4))

        params = torch.tensor(self.params)
        res = torch.autograd.functional.jacobian(circuit, params)
        assert qp.math.shape(res) == (4,)
        assert np.allclose(res, self.exp_grad, atol=1e-5)

    @pytest.mark.autograd
    def test_autograd(self):
        """Test the autograd interface"""

        dev = qp.device("reference.qubit", wires=5)

        @qp.qnode(dev)
        def circuit(coeffs):
            H = qp.ops.LinearCombination(
                coeffs, [qp.Y(0), qp.Y(1) @ qp.Y(2), qp.X(0), -1 * qp.X(1) @ qp.X(2)]
            )
            qp.PrepSelPrep(H, control=(3, 4))
            return qp.expval(qp.PauliZ(3) @ qp.PauliZ(4))

        params = qp.numpy.array(self.params, requires_grad=True)
        res = qp.grad(circuit)(params)

        assert qp.math.shape(res) == (4,)
        assert np.allclose(res, self.exp_grad, atol=1e-5)

    @pytest.mark.jax
    def test_jax(self):
        """Test the jax interface"""
        import jax

        dev = qp.device("reference.qubit", wires=5)

        @qp.qnode(dev)
        def circuit(coeffs):
            H = qp.ops.LinearCombination(
                coeffs, [qp.Y(0), qp.Y(1) @ qp.Y(2), qp.X(0), -1 * qp.X(1) @ qp.X(2)]
            )
            qp.PrepSelPrep(H, control=(3, 4))
            return qp.expval(qp.PauliZ(3) @ qp.PauliZ(4))

        res = jax.grad(circuit)(self.params)

        assert qp.math.shape(res) == (4,)
        assert np.allclose(res, self.exp_grad, atol=1e-5)

    @pytest.mark.jax
    def test_jit(self):
        """Test that jax jit works"""
        import jax

        dev = qp.device("reference.qubit", wires=5)

        @jax.jit
        @qp.qnode(dev)
        def circuit(coeffs):
            H = qp.ops.LinearCombination(
                coeffs, [qp.Y(0), qp.Y(1) @ qp.Y(2), qp.X(0), -1 * qp.X(1) @ qp.X(2)]
            )
            qp.PrepSelPrep(H, control=(3, 4))
            return qp.expval(qp.PauliZ(3) @ qp.PauliZ(4))

        res = jax.grad(circuit)(self.params)

        assert qp.math.shape(res) == (4,)
        assert np.allclose(res, self.exp_grad, atol=1e-5)
