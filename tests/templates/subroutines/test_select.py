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
"""
Tests for the Select template.
"""
# pylint: disable=protected-access,too-many-arguments,import-outside-toplevel, no-self-use
import copy

import numpy as np
import pytest
from scipy.stats import unitary_group

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.templates.subroutines.select import (
    _partial_select,
    _select_decomp_multi_control_work_wire,
    _select_decomp_unary,
)


@pytest.mark.jax
@pytest.mark.parametrize(
    "num_ops, num_controls",
    [(0, 1), (1, 1), (2, 1), (1, 2), (4, 2), (3, 4), (10, 4), (15, 4), (16, 4)],
)
@pytest.mark.parametrize("partial", [True, False])
@pytest.mark.parametrize("work_wires", [None, [5, 6, 7]])
@pytest.mark.parametrize("parametrized", [False, True])
def test_standard_checks(num_ops, num_controls, partial, work_wires, parametrized):
    """Run standard validity tests."""
    if parametrized:
        ops = [qml.RX(0.2, 0) for _ in range(num_ops)]
    else:
        ops = [qml.PauliX(0) for _ in range(num_ops)]
    control = list(range(1, num_controls + 1))

    op = qml.Select(ops, control, work_wires, partial=partial)
    if num_ops > 0:
        assert op.target_wires == qml.wires.Wires(0)
    else:
        assert op.target_wires == qml.wires.Wires([])
    qml.ops.functions.assert_valid(op)


@pytest.mark.unit
def test_repr():
    """Test the repr method."""
    ops = [qml.X(0), qml.Y(0)]
    control = [1]

    op = qml.Select(ops, control)
    assert repr(op) == "Select(ops=(X(0), Y(0)), control=Wires([1]), partial=False)"

    op = qml.Select(ops, control, partial=True)
    assert repr(op) == "Select(ops=(X(0), Y(0)), control=Wires([1]), partial=True)"


@pytest.mark.unit
@pytest.mark.parametrize(
    "K, control, expected",
    [
        (2, ["a"], [[("a",), (0,)], [("a",), (1,)]]),
        (2, ["a", 33], [[(33,), (0,)], [(33,), (1,)]]),
        (3, [7, "aux"], [[(7, "aux"), (0, 0)], [("aux",), (1,)], [(7,), (1,)]]),
        (
            11,
            [0, 1, 2, 3],
            [
                [(0, 1, 2, 3), (0, 0, 0, 0)],
                [(0, 1, 2, 3), (0, 0, 0, 1)],
                [(0, 1, 2, 3), (0, 0, 1, 0)],
                [(1, 2, 3), (0, 1, 1)],
                [(1, 2, 3), (1, 0, 0)],
                [(1, 2, 3), (1, 0, 1)],
                [(1, 2, 3), (1, 1, 0)],
                [(1, 2, 3), (1, 1, 1)],
                [(0, 2, 3), (1, 0, 0)],
                [(0, 3), (1, 1)],
                [(0, 2), (1, 1)],
            ],
        ),
    ],
)
def test_partial_select(K, control, expected):
    """Tests that the _partial_select function produces the correct simplified control
    structure."""
    assert list(_partial_select(K, control)) == expected


@pytest.mark.unit
class TestSelect:
    """Tests that the fundamental methods of Select work properly."""

    def test_copy(self):
        """Test that the copy function of Select works correctly."""
        ops = [qml.X(wires=2), qml.RX(0.2, wires=3), qml.Y(wires=2), qml.SWAP([2, 3])]
        op = qml.Select(ops, control=[0, 1])
        op_copy = copy.copy(op)

        qml.assert_equal(op, op_copy)

    @pytest.mark.parametrize(
        ("ops", "control"),
        [
            ([qml.X(0)], [1]),
            ([qml.X(0), qml.Y(0)], [1]),
            ([qml.RX(0.5, 0), qml.RY(0.7, 1)], [2]),
            ([qml.X(0), qml.I(0), qml.Z(0)], [1, 2]),
            ([qml.RX(0.5, 0), qml.RY(0.7, 1), qml.RZ(0.3, 1), qml.X(2)], [3, 4]),
            ([qml.X(0), qml.I(0), qml.I(0), qml.RX(0.3, 0)], [1, 2]),
            ([qml.X("a"), qml.Z("b"), qml.RX(0.7, "b")], ["c", 1]),
            ([qml.X("a"), qml.RX(0.7, "b")], ["c", 1]),
        ],
    )
    def test_basic_decomposition(self, ops, control):
        """Test the correctness of the Select template decomposition with partial=False.
        Tests both the returned and the queued operations."""
        control_values = [
            list(map(int, np.binary_repr(i, width=len(control)))) for i in range(len(ops))
        ]
        expected_gates = [qml.ctrl(op, control, vals) for op, vals in zip(ops, control_values)]

        select_op = qml.Select(ops, control, partial=False)
        with qml.queuing.AnnotatedQueue() as q0:
            decomp0 = select_op.decomposition()
        decomp_queue0 = qml.tape.QuantumScript.from_queue(q0).operations

        with qml.queuing.AnnotatedQueue() as q1:
            decomp1 = qml.Select.compute_decomposition(ops, control, partial=False)
        decomp_queue1 = qml.tape.QuantumScript.from_queue(q1).operations

        for dec in [decomp0, decomp1, decomp_queue0, decomp_queue1]:
            for op_dec, op_exp in zip(dec, expected_gates, strict=True):
                qml.assert_equal(op_dec, op_exp)

    @pytest.mark.parametrize(
        ("ops", "control", "expected_controls"),
        [
            ([qml.X(0)], [1], [([], [])]),
            ([qml.X(0), qml.Y(0)], [1], [([1], [0]), ([1], [1])]),
            ([qml.RX(0.5, 0), qml.RY(0.7, 1)], [2], [([2], [0]), ([2], [1])]),
            ([qml.X(0), qml.I(0), qml.Z(0)], [1, 2], [([1, 2], [0, 0]), ([2], [1]), ([1], [1])]),
            (
                [qml.RX(0.5, 0), qml.RY(0.7, 1), qml.RZ(0.3, 1), qml.X(2)],
                [3, 4],
                [([3, 4], [0, 0]), ([3, 4], [0, 1]), ([3, 4], [1, 0]), ([3, 4], [1, 1])],
            ),
            (
                [qml.X(0), qml.I(0), qml.I(0), qml.RX(0.3, 0)],
                [1, 2],
                [([1, 2], [0, 0]), ([1, 2], [0, 1]), ([1, 2], [1, 0]), ([1, 2], [1, 1])],
            ),
            (
                [qml.X("a"), qml.Z("b"), qml.RX(0.7, "b")],
                ["c", 1],
                [(["c", 1], [0, 0]), ([1], [1]), (["c"], [1])],
            ),
            ([qml.X("a"), qml.RX(0.7, "b")], ["c", 1], [([1], [0]), ([1], [1])]),
        ],
    )
    def test_basic_decomposition_partial(self, ops, control, expected_controls):
        """Test the correctness of the Select template decomposition with partial=True.
        Tests both the returned and the queued operations."""
        expected_gates = [
            qml.ctrl(op, ctrl, vals) if ctrl else op
            for op, (ctrl, vals) in zip(ops, expected_controls)
        ]

        select_op = qml.Select(ops, control, partial=True)
        with qml.queuing.AnnotatedQueue() as q0:
            decomp0 = select_op.decomposition()
        decomp_queue0 = qml.tape.QuantumScript.from_queue(q0).operations

        with qml.queuing.AnnotatedQueue() as q1:
            decomp1 = qml.Select.compute_decomposition(ops, control, partial=True)
        decomp_queue1 = qml.tape.QuantumScript.from_queue(q1).operations

        for dec in [decomp0, decomp1, decomp_queue0, decomp_queue1]:
            for op_dec, op_exp in zip(dec, expected_gates, strict=True):
                qml.assert_equal(op_dec, op_exp)

    @pytest.mark.parametrize("partial", [False, True])
    def test_new_decomposition_multi_control(self, partial):
        """Test that the multi-control decomposition is properly registered in the new system.
        This test uses two control qubits and four target operators, so that the Select
        template is never a partial Select, and the kwarg ``partial`` has no effect.
        """
        decomp = qml.list_decomps(qml.Select)[0]

        ops = [qml.X(2), qml.X(3), qml.X(4), qml.Y(2)]
        op_reps = (
            qml.resource_rep(qml.X),
            qml.resource_rep(qml.X),
            qml.resource_rep(qml.X),
            qml.resource_rep(qml.Y),
        )
        control = (0, 1)

        resource_obj = decomp.compute_resources(
            op_reps, num_control_wires=2, partial=partial, num_work_wires=0
        )

        assert resource_obj.num_gates == 4

        c_resource = qml.decomposition.resources.controlled_resource_rep

        kwargs = {"base_params": {}, "num_control_wires": 2, "num_work_wires": 0}

        expected_counts = {
            c_resource(base_class=qml.X, **kwargs, num_zero_control_values=2): 1,
            c_resource(base_class=qml.X, **kwargs, num_zero_control_values=1): 2,
            c_resource(base_class=qml.Y, **kwargs, num_zero_control_values=0): 1,
        }
        assert resource_obj.gate_counts == expected_counts

        op = qml.Select(ops, control, partial=partial)
        with qml.queuing.AnnotatedQueue() as q:
            decomp(*op.data, wires=op.wires, **op.hyperparameters)

        decomp_ops = qml.tape.QuantumScript.from_queue(q).operations

        qml.assert_equal(decomp_ops[0], qml.ctrl(qml.X(2), (0, 1), control_values=[0, 0]))
        qml.assert_equal(decomp_ops[1], qml.ctrl(qml.X(3), (0, 1), control_values=[0, 1]))
        qml.assert_equal(decomp_ops[2], qml.ctrl(qml.X(4), (0, 1), control_values=[1, 0]))
        qml.assert_equal(decomp_ops[3], qml.ctrl(qml.Y(2), (0, 1), control_values=[1, 1]))

    @pytest.mark.parametrize("partial", [False, True])
    def test_new_decomposition_multi_control_partial(self, partial):
        """Test that the multi-control decomposition is properly registered in the new system.
        This test uses two control qubits and three target operators, so that the Select
        template is a partial Select, and the kwarg ``partial`` has an effect.
        """

        decomp = qml.list_decomps(qml.Select)[0]

        ops = [qml.X(2), qml.X(3), qml.SWAP([2, 3])]
        op_reps = (
            qml.resource_rep(qml.X),
            qml.resource_rep(qml.X),
            qml.resource_rep(qml.SWAP),
        )
        control = (0, 1)

        resource_obj = decomp.compute_resources(
            op_reps, num_control_wires=2, partial=partial, num_work_wires=0
        )

        assert resource_obj.num_gates == 3

        c_resource = qml.decomposition.resources.controlled_resource_rep

        kwargs22 = {"base_params": {}, "num_control_wires": 2, "num_zero_control_values": 2}
        kwargs21 = {"base_params": {}, "num_control_wires": 2, "num_zero_control_values": 1}
        kwargs10 = {"base_params": {}, "num_control_wires": 1, "num_zero_control_values": 0}

        if partial:
            expected_counts = {
                c_resource(base_class=qml.X, **kwargs22): 1,
                c_resource(base_class=qml.X, **kwargs10): 1,
                c_resource(base_class=qml.SWAP, **kwargs10): 1,
            }
        else:
            expected_counts = {
                c_resource(base_class=qml.X, **kwargs22): 1,
                c_resource(base_class=qml.X, **kwargs21): 1,
                c_resource(base_class=qml.SWAP, **kwargs21): 1,
            }
        assert resource_obj.gate_counts == expected_counts

        op = qml.Select(ops, control, partial=partial)
        with qml.queuing.AnnotatedQueue() as q:
            decomp(*op.data, wires=op.wires, **op.hyperparameters)

        decomp_ops = qml.tape.QuantumScript.from_queue(q).operations

        if partial:
            ctrls = [(0, 1), (1,), (0,)]
            ctrl_vals = [[0, 0], [1], [1]]
        else:
            ctrls = [(0, 1)] * 3
            ctrl_vals = [[0, 0], [0, 1], [1, 0]]
        for decomp_op, op, ctrl, ctrl_val in zip(decomp_ops, ops, ctrls, ctrl_vals, strict=True):
            qml.assert_equal(decomp_op, qml.ctrl(op, ctrl, control_values=ctrl_val))

    @pytest.mark.parametrize("partial", [False, True])
    def test_new_decomposition_multi_control_single_op(self, partial):
        """Test that the multi-control decomposition is properly registered in the new system.
        This test uses a single control wire and just one operator to be applied.
        This is a partial Select, and the kwarg ``partial`` has a notable effect.
        """
        decomp = qml.list_decomps(qml.Select)[0]

        ops = [qml.Z(1)]
        op_reps = (qml.resource_rep(qml.Z),)
        control = (0,)

        resource_obj = decomp.compute_resources(
            op_reps, num_control_wires=1, partial=partial, num_work_wires=0
        )

        assert resource_obj.num_gates == 1

        c_resource = qml.decomposition.resources.controlled_resource_rep

        kwargs = {"base_params": {}, "num_control_wires": 1, "num_work_wires": 0}

        if partial:
            expected_counts = {qml.resource_rep(qml.Z): 1}
        else:
            expected_counts = {c_resource(base_class=qml.Z, **kwargs, num_zero_control_values=1): 1}
        assert resource_obj.gate_counts == expected_counts

        op = qml.Select(ops, control, partial=partial)
        with qml.queuing.AnnotatedQueue() as q:
            decomp(*op.data, wires=op.wires, **op.hyperparameters)

        decomp_ops = qml.tape.QuantumScript.from_queue(q).operations
        assert len(decomp_ops) == 1

        if partial:
            qml.assert_equal(decomp_ops[0], qml.Z(1))
        else:
            qml.assert_equal(decomp_ops[0], qml.ctrl(qml.Z(1), (0,), control_values=[0]))

    def test_resources(self):
        """Test the resources property"""

        assert qml.Select.resource_keys == frozenset(
            ("op_reps", "num_control_wires", "partial", "num_work_wires")
        )

        ops = [qml.X(2), qml.X(3), qml.X(4), qml.Y(2)]

        op = qml.Select(ops, control=(0, 1))

        resources = op.resource_params
        assert resources["num_control_wires"] == 2

        op_reps = (
            qml.resource_rep(qml.X),
            qml.resource_rep(qml.X),
            qml.resource_rep(qml.X),
            qml.resource_rep(qml.Y),
        )

        assert resources["op_reps"] == op_reps


class TestErrorMessages:
    """Test that the correct errors are raised"""

    @pytest.mark.parametrize(
        ("ops", "control", "msg_match"),
        [
            (
                [qml.X(wires=1), qml.Y(wires=0), qml.Z(wires=0)],
                [1, 2],
                "Control wires should be different from operation wires.",
            ),
            (
                [qml.X(wires=2)] * 4,
                [1, 2, 3],
                "Control wires should be different from operation wires.",
            ),
            (
                [qml.X(wires="a"), qml.Y(wires="b")],
                ["a"],
                "Control wires should be different from operation wires.",
            ),
        ],
    )
    def test_control_in_ops(self, ops, control, msg_match):
        """Test an error is raised when a control wire is in one of the ops"""
        with pytest.raises(ValueError, match=msg_match):
            qml.Select(ops, control)

    @pytest.mark.parametrize(
        ("ops", "control", "msg_match"),
        [
            (
                [qml.X(wires=0), qml.Y(wires=0), qml.Z(wires=0)],
                [1],
                r"Not enough control wires \(1\) for the desired number of operations \(3\). At least 2 control wires are required.",
            ),
            (
                [qml.X(wires=0)] * 10,
                [1, 2, 3],
                r"Not enough control wires \(3\) for the desired number of operations \(10\). At least 4 control wires are required.",
            ),
            (
                [qml.X(wires="a"), qml.Y(wires="b"), qml.Z(wires="c")],
                [1],
                r"Not enough control wires \(1\) for the desired number of operations \(3\). At least 2 control wires are required.",
            ),
        ],
    )
    def test_too_many_ops(self, ops, control, msg_match):
        """Test that error is raised if more ops are requested than can fit in control wires"""
        with pytest.raises(ValueError, match=msg_match):
            qml.Select(ops, control)


def select_rx_circuit(angles):
    """Circuit that uses Select for tests."""
    qml.RY(0.6135, 0)
    qml.Select([qml.RX(angles[0], wires=[1]), qml.RY(angles[1], wires=[1])], control=0)
    return qml.expval(qml.Z(wires=1))


def manual_rx_circuit(angles):
    """Circuit that manually creates Select for tests."""
    qml.RY(0.6135, 0)
    qml.ctrl(qml.RX(angles[0], wires=[1]), control=0, control_values=0)
    qml.ctrl(qml.RY(angles[1], wires=[1]), control=0)
    return qml.expval(qml.Z(wires=1))


class TestInterfaces:
    """Tests that the template is compatible with all interfaces, including the computation
    of gradients."""

    @pytest.mark.autograd
    def test_autograd(self):
        """Tests the autograd interface."""
        dev = qml.device("default.qubit", wires=2)

        circuit_default = qml.QNode(manual_rx_circuit, dev)
        circuit_select = qml.QNode(select_rx_circuit, dev)

        input_default = [0.5, 0.2]
        input_grad = pnp.array(input_default, requires_grad=True)

        grad_fn = qml.grad(circuit_default)
        grads = grad_fn(input_grad)

        grad_fn2 = qml.grad(circuit_select)
        grads2 = grad_fn2(input_grad)

        assert qml.math.allclose(grads, grads2)

    @pytest.mark.autograd
    def test_autograd_parameter_shift(self):
        """Tests the autograd interface using the parameter-shift method."""
        dev = qml.device("default.qubit", wires=2)

        circuit_default = qml.QNode(manual_rx_circuit, dev, diff_method="parameter-shift")
        circuit_select = qml.QNode(select_rx_circuit, dev, diff_method="parameter-shift")

        input_default = [0.5, 0.2]
        input_grad = pnp.array(input_default, requires_grad=True)

        grad_fn = qml.grad(circuit_default)
        grads = grad_fn(input_grad)

        grad_fn2 = qml.grad(circuit_select)
        grads2 = grad_fn2(input_grad)

        assert qml.math.allclose(grads, grads2)

    @pytest.mark.tf
    def test_tf(self):
        """Tests the tf interface."""
        import tensorflow as tf

        dev = qml.device("default.qubit", wires=2)

        circuit_default = qml.QNode(manual_rx_circuit, dev)
        circuit_tf = qml.QNode(select_rx_circuit, dev)

        input_default = [0.5, 0.2]
        input_tf = tf.Variable(input_default)

        assert qml.math.allclose(
            qml.matrix(circuit_default)(input_default), qml.matrix(circuit_tf)(input_tf)
        )
        assert qml.math.get_interface(qml.matrix(circuit_tf)(input_tf)) == "tensorflow"

        with tf.GradientTape() as tape:
            res = circuit_default(input_tf)
        grads = tape.gradient(res, [input_tf])

        with tf.GradientTape() as tape2:
            res2 = circuit_tf(input_tf)
        grads2 = tape2.gradient(res2, [input_tf])

        assert qml.math.allclose(grads[0], grads2[0])

    @pytest.mark.torch
    def test_torch(self):
        """Tests the torch interface."""
        import torch

        dev = qml.device("default.qubit", wires=2)

        circuit_default = qml.QNode(manual_rx_circuit, dev)
        circuit_torch = qml.QNode(select_rx_circuit, dev)

        input_default = [0.5, 0.2]
        input_torch = torch.tensor(input_default, requires_grad=True)

        assert qml.math.allclose(
            qml.matrix(circuit_default)(input_default), qml.matrix(circuit_torch)(input_torch)
        )
        assert qml.math.get_interface(qml.matrix(circuit_torch)(input_torch)) == "torch"

        res = circuit_default(input_torch)
        res.backward()
        grads = [input_torch.grad]

        res2 = circuit_torch(input_torch)
        res2.backward()
        grads2 = [input_torch.grad]

        assert qml.math.allclose(grads[0], grads2[0])

    @pytest.mark.jax
    @pytest.mark.slow
    def test_jax(self):
        """Tests the jax interface."""
        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit", wires=2)

        input_default = [0.5, 0.2]
        input_jax = jnp.array(input_default)

        circuit_default = qml.QNode(manual_rx_circuit, dev)
        circuit_jax = qml.QNode(select_rx_circuit, dev)

        assert qml.math.allclose(
            qml.matrix(circuit_default)(input_default), qml.matrix(circuit_jax)(input_jax)
        )
        assert qml.math.get_interface(qml.matrix(circuit_jax)(input_jax)) == "jax"

        grad_fn = jax.grad(circuit_default)
        grads = grad_fn(input_jax)

        grad_fn2 = jax.grad(circuit_jax)
        grads2 = grad_fn2(input_jax)

        assert qml.math.allclose(grads, grads2)

    @pytest.mark.jax
    def test_jax_jit(self):
        """Tests jit within the jax interface."""
        import jax

        dev = qml.device("default.qubit", wires=4)
        ops = [qml.X(2), qml.X(3), qml.Y(2), qml.SWAP([2, 3])]

        @qml.qnode(dev)
        def circuit():
            qml.Select(ops, control=[0, 1])
            return qml.state()

        jit_circuit = jax.jit(circuit)

        assert qml.math.allclose(circuit(), jit_circuit())


num_controls_and_num_ops = (
    [(nc, i) for nc in range(1, 5) for i in range(1, 2**nc + 1)]
    + [(5, 1), (5, 2), (5, 17), (5, 24), (5, 31)]
    + [(6, 3), (6, 8), (6, 27)]
)


@pytest.mark.parametrize("partial", [False, True])
class TestUnaryIterator:
    """Tests for the auxiliary qubit-based unary iterator decomposition of Select."""

    def test_is_registered_with_select(self, partial):
        """Test that the unary iteration decomposition is registered correctly with
        pml.Select."""
        # pylint: disable=unused-argument
        decomp = qml.list_decomps(qml.Select)[1]
        assert decomp is _select_decomp_unary

    @pytest.mark.parametrize(
        "c, K, expected_ops, expected_ops_partial",
        [
            (1, 1, [qml.ctrl(qml.SWAP([0, 1]), ["c0"], [0])], [qml.SWAP([0, 1])]),
            (1, 2, [qml.ctrl(qml.SWAP([0, 1]), ["c0"], [0]), qml.CY(["c0", 1])], None),
            (
                2,
                1,
                [
                    qml.Elbow(["c0", "c1", "w0"], (0, 0)),
                    qml.ctrl(qml.SWAP([0, 1]), ["w0"]),
                    qml.adjoint(qml.Elbow(["c0", "c1", "w0"], (0, 0))),
                ],
                [qml.SWAP([0, 1])],
            ),
            (
                2,
                2,
                [
                    qml.Elbow(["c0", "c1", "w0"], (0, 0)),
                    qml.ctrl(qml.SWAP([0, 1]), ["w0"]),
                    qml.ctrl(qml.X("w0"), control="c0", control_values=[0]),
                    qml.CY(["w0", 1]),
                    qml.adjoint(qml.Elbow(["c0", "c1", "w0"], (0, 1))),
                ],
                [
                    qml.ctrl(qml.SWAP([0, 1]), ["c1"], [0], work_wires=["w0"]),
                    qml.ctrl(qml.Y(1), ["c1"], work_wires=["w0"]),
                ],
            ),
            (
                2,
                3,
                [
                    qml.Elbow(["c0", "c1", "w0"], (0, 0)),
                    qml.ctrl(qml.SWAP([0, 1]), ["w0"]),
                    qml.ctrl(qml.X("w0"), control="c0", control_values=[0]),
                    qml.CY(["w0", 1]),
                    qml.CNOT(["c0", "w0"]),
                    qml.CNOT(["c1", "w0"]),
                    qml.ctrl(qml.CRZ(0.4, [1, 0]), ["w0"], [1], work_wires=["c0", "c1"]),
                    qml.adjoint(qml.Elbow(["c0", "c1", "w0"], (1, 0))),
                ],
                [
                    qml.Elbow(["c0", "c1", "w0"], (0, 0)),
                    qml.ctrl(qml.SWAP([0, 1]), ["w0"]),
                    qml.ctrl(qml.X("w0"), control="c0", control_values=[0]),
                    qml.CY(["w0", 1]),
                    qml.adjoint(qml.Elbow(["c0", "c1", "w0"], (0, 1))),
                    qml.ctrl(qml.CRZ(0.4, [1, 0]), ["c0"]),
                ],
            ),
            (
                2,
                4,
                [
                    qml.Elbow(["c0", "c1", "w0"], (0, 0)),
                    qml.ctrl(qml.SWAP([0, 1]), ["w0"]),
                    qml.ctrl(qml.X("w0"), control="c0", control_values=[0]),
                    qml.CY(["w0", 1]),
                    qml.CNOT(["c0", "w0"]),
                    qml.CNOT(["c1", "w0"]),
                    qml.ctrl(qml.CRZ(0.4, [1, 0]), ["w0"], [1], work_wires=["c0", "c1"]),
                    qml.CNOT(["c0", "w0"]),
                    qml.CNOT(["w0", 0]),
                    qml.adjoint(qml.Elbow(["c0", "c1", "w0"], (1, 1))),
                ],
                None,
            ),
            (
                3,
                7,
                [
                    qml.Elbow(["c0", "c1", "w0"], (0, 0)),
                    qml.Elbow(["w0", "c2", "w1"], (1, 0)),
                    qml.ctrl(qml.SWAP([0, 1]), ["w1"]),
                    qml.CNOT(["w0", "w1"]),
                    qml.CY(["w1", 1]),
                    qml.adjoint(qml.Elbow(["w0", "c2", "w1"], (1, 1))),
                    qml.ctrl(qml.X("w0"), control="c0", control_values=[0]),
                    qml.Elbow(["w0", "c2", "w1"], (1, 0)),
                    qml.ctrl(qml.CRZ(0.4, [1, 0]), ["w1"], work_wires=["c0", "c1", "c2"]),
                    qml.CNOT(["w0", "w1"]),
                    qml.CNOT(["w1", 0]),
                    qml.adjoint(qml.Elbow(["w0", "c2", "w1"], (1, 1))),
                    qml.CNOT(["c0", "w0"]),
                    qml.CNOT(["c1", "w0"]),
                    qml.Elbow(["w0", "c2", "w1"], (1, 0)),
                    qml.CZ(["w1", 1]),
                    qml.CNOT(["w0", "w1"]),
                    qml.ctrl(qml.X(0) @ qml.Z(1), ["w1"], work_wires=["c0", "c1", "c2"]),
                    qml.adjoint(qml.Elbow(["w0", "c2", "w1"])),
                    qml.CNOT(["c0", "w0"]),
                    qml.Elbow(["w0", "c2", "w1"], (1, 0)),
                    qml.CH(["w1", 1]),
                    qml.adjoint(qml.Elbow(["w0", "c2", "w1"], (1, 0))),
                    qml.adjoint(qml.Elbow(["c0", "c1", "w0"], (1, 1))),
                ],
                [
                    qml.Elbow(["c0", "c1", "w0"], (0, 0)),
                    qml.Elbow(["w0", "c2", "w1"], (1, 0)),
                    qml.ctrl(qml.SWAP([0, 1]), ["w1"]),
                    qml.CNOT(["w0", "w1"]),
                    qml.CY(["w1", 1]),
                    qml.adjoint(qml.Elbow(["w0", "c2", "w1"], (1, 1))),
                    qml.ctrl(qml.X("w0"), control="c0", control_values=[0]),
                    qml.Elbow(["w0", "c2", "w1"], (1, 0)),
                    qml.ctrl(qml.CRZ(0.4, [1, 0]), ["w1"], work_wires=["c0", "c1", "w0", "c2"]),
                    qml.CNOT(["w0", "w1"]),
                    qml.CNOT(["w1", 0]),
                    qml.adjoint(qml.Elbow(["w0", "c2", "w1"], (1, 1))),
                    qml.CNOT(["c0", "w0"]),
                    qml.CNOT(["c1", "w0"]),
                    qml.Elbow(["w0", "c2", "w1"], (1, 0)),
                    qml.CZ(["w1", 1]),
                    qml.CNOT(["w0", "w1"]),
                    qml.ctrl(qml.X(0) @ qml.Z(1), ["w1"], work_wires=["c0", "c1", "w0", "c2"]),
                    qml.adjoint(qml.Elbow(["w0", "c2", "w1"])),
                    qml.CNOT(["c0", "w0"]),
                    qml.CH(["w0", 1]),
                    qml.adjoint(qml.Elbow(["c0", "c1", "w0"], (1, 1))),
                ],
            ),
        ],
    )
    def test_expected_operators(self, c, K, expected_ops, expected_ops_partial, partial):
        ops = [
            qml.SWAP([0, 1]),
            qml.Y(1),
            qml.CRZ(0.4, [1, 0]),
            qml.X(0),
            qml.Z(1),
            qml.X(0) @ qml.Z(1),
            qml.H(1),
        ][:K]
        control = [f"c{i}" for i in range(c)]
        work_wires = [f"w{i}" for i in range(c - 1)]
        decomp = _select_decomp_unary(
            ops=ops, control=control, work_wires=work_wires, partial=partial
        )

        if partial and expected_ops_partial is not None:
            expected = expected_ops_partial
        else:
            expected = expected_ops
        for op, exp_op in zip(decomp, expected, strict=True):
            qml.assert_equal(op, exp_op)

    @pytest.mark.parametrize("num_controls", [0, 1, 2, 3])
    def test_no_ops(self, num_controls, partial):
        """Test that the unary iterator does not return any operators for an empty list
        of target operators."""

        control = list(range(num_controls))
        work = list(range(num_controls, 2 * num_controls - 1))

        with qml.queuing.AnnotatedQueue() as q:
            _select_decomp_unary(ops=[], control=control, work_wires=work, partial=partial)

        assert len(q) == 0

    @pytest.mark.parametrize("num_controls, num_ops", num_controls_and_num_ops)
    def test_identity_with_basis_states(self, num_controls, num_ops, partial):
        """Test that the unary iterator is correct by asserting that the identity
        matrix is created by preparing the i-th computational basis state conditioned on the
        i-th basis state in the control qubits."""

        dev = qml.device("default.qubit")

        # Create angle set so that feeding angles[i] into RX on the i-th control wire will
        # yield broadcasted BasisEmbedding (which does not support broadcasting atm)
        angles = [list(map(int, np.binary_repr(i, width=num_controls))) for i in range(num_ops)]
        angles = np.pi * np.array(angles).T
        control = list(range(num_controls))
        work = list(range(num_controls, 2 * num_controls - 1))
        target = list(range(2 * num_controls - 1, 3 * num_controls - 1))

        ops = [qml.BasisEmbedding(i, wires=target) for i in range(num_ops)]

        @qml.qnode(dev)
        def circuit():
            for w, angle in zip(control, angles, strict=True):
                qml.RX(angle, w)
            _select_decomp_unary(ops=ops, control=control, work_wires=work, partial=partial)
            return qml.probs(target)

        probs = circuit()
        assert np.allclose(probs, np.eye(2**num_controls)[:num_ops])

    @pytest.mark.parametrize(
        ("num_ops", "control", "work", "msg_match"),
        [(9, 4, 1, "Can't use this decomposition")],
    )
    def test_operation_and_test_wires_error(
        self, num_ops, control, work, msg_match, partial
    ):  # pylint: disable=too-many-arguments
        """Test that proper errors are raised"""

        wires = qml.registers({"target": num_ops, "control": control, "work": work})
        ops = [qml.BasisEmbedding(i, wires=wires["target"]) for i in range(num_ops)]

        with pytest.raises(ValueError, match=msg_match):
            _select_decomp_unary(
                ops=ops, control=wires["control"], work_wires=wires["work"], partial=partial
            )

    def test_error_too_few_controls(self, partial):
        """Test that an error is raised if too few control wires are given."""

        too_many_ops = [qml.X(0) for _ in range(9)]
        exactly_right_ops = [qml.X(0) for _ in range(8)]
        fewer_ops = [qml.X(0) for _ in range(7)]
        kwargs = {"control": [1, 2, 3], "work_wires": [4, 5], "partial": partial}

        with pytest.raises(ValueError, match="At least 4 control wires are required"):
            _select_decomp_unary(ops=too_many_ops, **kwargs)

        # Test that no error is raised for exactly right number of ops for three controls
        _ = _select_decomp_unary(ops=exactly_right_ops, **kwargs)
        # Test that no error is raised for fewer than exactly right number of ops
        _ = _select_decomp_unary(ops=fewer_ops, **kwargs)

    @pytest.mark.parametrize("num_controls, num_ops", num_controls_and_num_ops)
    def test_comparison_with_select(self, num_controls, num_ops, seed, partial):
        """Test that the unary iterator is correct by comparing it to the standard Select
        decomposition."""

        angles = [list(map(int, np.binary_repr(i, width=num_controls))) for i in range(num_ops)]
        angles = np.pi * np.array(angles).T
        control = list(range(num_controls))
        work = list(range(num_controls, 2 * num_controls - 1))
        target = [2 * num_controls - 1, 2 * num_controls]

        dev = qml.device("default.qubit", wires=control + work + target)
        unitaries = unitary_group.rvs(4, size=num_ops, random_state=seed)
        if num_ops == 1:
            unitaries = np.array([unitaries])
        ops = [qml.QubitUnitary(U, wires=target) for U in unitaries]
        adj_ops = [qml.QubitUnitary(U.conj().T, wires=target) for U in unitaries]

        @qml.qnode(dev)
        def circuit():
            for w, angle in zip(control, angles, strict=True):
                qml.RX(angle, w)
            _select_decomp_unary(ops=ops, control=control, work_wires=work, partial=partial)
            qml.Select(adj_ops, control=control, work_wires=None, partial=partial)
            return qml.probs(target)

        probs = circuit()
        exp = np.eye(4)[0]
        assert np.allclose(probs, exp)


@pytest.mark.parametrize("partial", [False, True])
class TestSelectWithWorkWire:
    """Tests for the 1 work-wire decomposition of Select."""

    @pytest.mark.parametrize("num_controls", [0, 1, 2, 3])
    def test_no_ops(self, num_controls, partial):
        """Test that the decomposition does not return any operators for an empty list
        of target operators."""

        control = list(range(num_controls))
        work = ["a"]

        with qml.queuing.AnnotatedQueue() as q:
            _select_decomp_multi_control_work_wire(
                ops=[], control=control, work_wires=work, partial=partial
            )

        assert len(q) == 0

    @pytest.mark.parametrize("num_controls, num_ops", num_controls_and_num_ops)
    def test_identity_with_basis_states(self, num_controls, num_ops, partial):
        """Test that the decomposition is correct by asserting that the identity
        matrix is created by preparing the i-th computational basis state conditioned on the
        i-th basis state in the control qubits."""

        dev = qml.device("default.qubit")

        # Create angle set so that feeding angles[i] into RX on the i-th control wire will
        # yield broadcasted BasisEmbedding (which does not support broadcasting atm)
        angles = [list(map(int, np.binary_repr(i, width=num_controls))) for i in range(num_ops)]
        angles = np.pi * np.array(angles).T
        control = list(range(num_controls))
        work = ["a"]
        target = list(range(2 * num_controls - 1, 3 * num_controls - 1))

        ops = [qml.BasisEmbedding(i, wires=target) for i in range(num_ops)]

        @qml.qnode(dev)
        def circuit():
            for w, angle in zip(control, angles, strict=True):
                qml.RX(angle, w)
            _select_decomp_multi_control_work_wire(
                ops=ops, control=control, work_wires=work, partial=partial
            )
            return qml.probs(target)

        probs = circuit()
        assert np.allclose(probs, np.eye(2**num_controls)[:num_ops])

    @pytest.mark.parametrize("num_controls, num_ops", num_controls_and_num_ops)
    def test_comparison_with_select(self, num_controls, num_ops, seed, partial):
        """Test that the decomposition is correct by comparing it to the standard Select
        decomposition."""

        angles = [list(map(int, np.binary_repr(i, width=num_controls))) for i in range(num_ops)]
        angles = np.pi * np.array(angles).T
        control = list(range(num_controls))
        work = ["a"]
        target = [2 * num_controls - 1, 2 * num_controls]

        dev = qml.device("default.qubit", wires=control + work + target)
        unitaries = unitary_group.rvs(4, size=num_ops, random_state=seed)
        if num_ops == 1:
            unitaries = np.array([unitaries])
        ops = [qml.QubitUnitary(U, wires=target) for U in unitaries]
        adj_ops = [qml.QubitUnitary(U.conj().T, wires=target) for U in unitaries]

        @qml.qnode(dev)
        def circuit():
            for w, angle in zip(control, angles, strict=True):
                qml.RX(angle, w)
            _select_decomp_multi_control_work_wire(
                ops=ops, control=control, work_wires=work, partial=partial
            )
            qml.Select(adj_ops, control=control, work_wires=None, partial=partial)
            return qml.probs(target)

        probs = circuit()
        exp = np.eye(4)[0]
        assert np.allclose(probs, exp)
