# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Tests for the transform ``qml.transform.split_non_commuting()`` """
# pylint: disable=no-self-use, import-outside-toplevel, no-member, import-error
import pytest
import numpy as np
import pennylane as qml
import pennylane.numpy as pnp

from pennylane.transforms import split_non_commuting

# example tape with 3 commuting groups [[0,3],[1,4],[2,5]]
with qml.queuing.AnnotatedQueue() as q3:
    qml.PauliZ(0)
    qml.Hadamard(0)
    qml.CNOT((0, 1))
    qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
    qml.expval(qml.PauliX(0) @ qml.PauliX(1))
    qml.expval(qml.PauliY(0) @ qml.PauliY(1))
    qml.expval(qml.PauliZ(0))
    qml.expval(qml.PauliX(0))
    qml.expval(qml.PauliY(0))

non_commuting_tape3 = qml.tape.QuantumScript.from_queue(q3)

# example tape with 2 commuting groups [[0,2],[1,3]]
with qml.queuing.AnnotatedQueue() as q2:
    qml.PauliZ(0)
    qml.Hadamard(0)
    qml.CNOT((0, 1))
    qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))
    qml.expval(qml.PauliX(0) @ qml.PauliX(1))
    qml.expval(qml.PauliZ(0))
    qml.expval(qml.PauliX(0))

non_commuting_tape2 = qml.tape.QuantumScript.from_queue(q2)

# list of observables with 2 commuting groups [[1, 3], [0, 2, 4]]
obs_list_2 = [
    qml.PauliZ(0) @ qml.PauliZ(1),
    qml.PauliX(0) @ qml.PauliX(1),
    qml.PauliZ(0),
    qml.PauliX(0),
    qml.PauliZ(1),
]

# list of observables with 3 commuting groups [[0,3], [1,4], [2,5]]
obs_list_3 = [
    qml.PauliZ(0) @ qml.PauliZ(1),
    qml.PauliX(0) @ qml.PauliX(1),
    qml.PauliY(0) @ qml.PauliY(1),
    qml.PauliZ(0),
    qml.PauliX(0),
    qml.PauliY(0),
]

obs_fn = [qml.expval, qml.var]
# measurements that accept observables as arguments
obs_meas_fn = [qml.expval, qml.var, qml.probs, qml.counts, qml.sample]

# measurements that accept wires as arguments
wire_meas_fn = [qml.probs, qml.counts, qml.sample]


class TestUnittestSplitNonCommuting:
    """Unit tests on ``qml.transforms.split_non_commuting()``"""

    @pytest.mark.parametrize("meas_type", obs_meas_fn)
    def test_commuting_group_no_split(self, mocker, meas_type):
        """Testing that commuting groups are not split for all supported measurement types"""
        with qml.queuing.AnnotatedQueue() as q:
            qml.PauliZ(0)
            qml.Hadamard(0)
            qml.CNOT((0, 1))
            meas_type(op=qml.PauliZ(0))
            meas_type(op=qml.PauliZ(0))
            meas_type(op=qml.PauliX(1))
            meas_type(op=qml.PauliZ(2))
            meas_type(op=qml.PauliZ(0) @ qml.PauliZ(3))

        # test transform on tape
        tape = qml.tape.QuantumScript.from_queue(q)
        split, fn = split_non_commuting(tape)

        spy = mocker.spy(qml.math, "concatenate")

        assert len(split) == 1
        assert all(isinstance(t, qml.tape.QuantumScript) for t in split)
        assert fn([0.5]) == 0.5

        # test transform on qscript
        qs = qml.tape.QuantumScript(tape.operations, tape.measurements)
        split, fn = split_non_commuting(qs)

        assert len(split) == 1
        assert all(isinstance(i_qs, qml.tape.QuantumScript) for i_qs in split)
        assert fn([0.5]) == 0.5

        spy.assert_not_called()

    @pytest.mark.parametrize("meas_type", wire_meas_fn)
    def test_wire_commuting_group_no_split(self, mocker, meas_type):
        """Testing that commuting MPs initialized using wires or observables are not split"""
        with qml.queuing.AnnotatedQueue() as q:
            qml.PauliZ(0)
            qml.Hadamard(0)
            qml.CNOT((0, 1))
            meas_type(wires=[0])
            meas_type(wires=[1])
            meas_type(wires=[0, 1])
            meas_type(op=qml.PauliZ(0))
            meas_type(op=qml.PauliZ(0) @ qml.PauliZ(2))

        # test transform on tape
        tape = qml.tape.QuantumScript.from_queue(q)
        split, fn = split_non_commuting(tape)

        spy = mocker.spy(qml.math, "concatenate")

        assert len(split) == 1
        assert all(isinstance(t, qml.tape.QuantumScript) for t in split)
        assert fn([0.5]) == 0.5

        # test transform on qscript
        qs = qml.tape.QuantumScript(tape.operations, tape.measurements)
        split, fn = split_non_commuting(qs)

        assert len(split) == 1
        assert all(isinstance(i_qs, qml.tape.QuantumScript) for i_qs in split)
        assert fn([0.5]) == 0.5

        spy.assert_not_called()

    @pytest.mark.parametrize("meas_type", obs_meas_fn)
    @pytest.mark.parametrize("obs_list, expected", [(obs_list_2, 2), (obs_list_3, 3)])
    def test_non_commuting_group_right_number(self, meas_type, obs_list, expected):
        """Test that the no. of tapes after splitting into commuting groups is of the right size"""

        # create a queue with several measurements of same type but with differnent non-commuting
        # observables
        with qml.queuing.AnnotatedQueue() as q:
            qml.PauliZ(0)
            qml.Hadamard(0)
            qml.CNOT((0, 1))
            for obs in obs_list:
                meas_type(op=obs)

            # if MP type can accept wires, then add two extra measurements using wires and test no.
            # of tapes after splitting commuting groups
            if meas_type in wire_meas_fn:
                meas_type(wires=[0])
                meas_type(wires=[0, 1])

        tape = qml.tape.QuantumScript.from_queue(q)
        split, _ = split_non_commuting(tape)
        assert len(split) == expected

        qs = qml.tape.QuantumScript(tape.operations, tape.measurements)
        split, _ = split_non_commuting(qs)
        assert len(split) == expected

    @pytest.mark.parametrize("meas_type", obs_meas_fn)
    @pytest.mark.parametrize(
        "obs_list, group_coeffs",
        [(obs_list_2, [[1, 3], [0, 2, 4]]), (obs_list_3, [[0, 3], [1, 4], [2, 5]])],
    )
    def test_non_commuting_group_right_reorder(self, meas_type, obs_list, group_coeffs):
        """Test that the output is of the correct order"""
        # create a queue with several measurements of same type but with differnent non-commuting
        # observables
        with qml.queuing.AnnotatedQueue() as q:
            qml.PauliZ(0)
            qml.Hadamard(0)
            qml.CNOT((0, 1))
            for obs in obs_list:
                meas_type(op=obs)

        tape = qml.tape.QuantumScript.from_queue(q)
        _, fn = split_non_commuting(tape)
        assert all(np.array(fn(group_coeffs)) == np.arange(len(tape.measurements)))

        qs = qml.tape.QuantumScript(tape.operations, tape.measurements)
        _, fn = split_non_commuting(qs)
        assert all(np.array(fn(group_coeffs)) == np.arange(len(tape.measurements)))

    @pytest.mark.parametrize("meas_type", wire_meas_fn)
    @pytest.mark.parametrize(
        "obs_list, group_coeffs",
        [(obs_list_2, [[1, 3], [0, 2, 4, 5]]), (obs_list_3, [[1, 4], [2, 5], [0, 3, 6]])],
    )
    def test_wire_non_commuting_group_right_reorder(self, meas_type, obs_list, group_coeffs):
        """Test that the output is of the correct order with wire MPs initialized using a
        combination of wires and observables"""
        # create a queue with several measurements of same type but with differnent non-commuting
        # observables
        with qml.queuing.AnnotatedQueue() as q:
            qml.PauliZ(0)
            qml.Hadamard(0)
            qml.CNOT((0, 1))
            for obs in obs_list:
                meas_type(op=obs)

            # initialize measurements using wires
            meas_type(wires=[0])

        tape = qml.tape.QuantumScript.from_queue(q)
        _, fn = split_non_commuting(tape)
        assert all(np.array(fn(group_coeffs)) == np.arange(len(tape.measurements)))

        qs = qml.tape.QuantumScript(tape.operations, tape.measurements)
        _, fn = split_non_commuting(qs)
        assert all(np.array(fn(group_coeffs)) == np.arange(len(tape.measurements)))

    @pytest.mark.parametrize("meas_type", obs_fn)
    def test_different_measurement_types(self, meas_type):
        """Test that expval, var and sample are correctly reproduced"""
        with qml.queuing.AnnotatedQueue() as q:
            qml.PauliZ(0)
            qml.Hadamard(0)
            qml.CNOT((0, 1))
            meas_type(qml.PauliZ(0) @ qml.PauliZ(1))
            meas_type(qml.PauliX(0) @ qml.PauliX(1))
            meas_type(qml.PauliZ(0))
            meas_type(qml.PauliX(0))
        tape = qml.tape.QuantumScript.from_queue(q)
        the_return_type = tape.measurements[0].return_type
        split, _ = split_non_commuting(tape)
        for new_tape in split:
            for meas in new_tape.measurements:
                assert meas.return_type == the_return_type

        qs = qml.tape.QuantumScript(tape.operations, tape.measurements)
        split, _ = split_non_commuting(qs)
        for new_tape in split:
            for meas in new_tape.measurements:
                assert meas.return_type == the_return_type

    def test_mixed_measurement_types(self):
        """Test that mixing expval and var works correctly."""

        with qml.queuing.AnnotatedQueue() as q:
            qml.Hadamard(0)
            qml.Hadamard(1)
            qml.expval(qml.PauliX(0))
            qml.expval(qml.PauliZ(1))
            qml.var(qml.PauliZ(0))

        tape = qml.tape.QuantumScript.from_queue(q)
        split, _ = split_non_commuting(tape)

        assert len(split) == 2

        with qml.queuing.AnnotatedQueue() as q:
            qml.Hadamard(0)
            qml.Hadamard(1)
            qml.expval(qml.PauliX(0))
            qml.var(qml.PauliZ(0))
            qml.expval(qml.PauliZ(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        split, _ = split_non_commuting(tape)

        assert len(split) == 2
        assert qml.equal(split[0].measurements[0], qml.expval(qml.PauliX(0)))
        assert qml.equal(split[0].measurements[1], qml.expval(qml.PauliZ(1)))
        assert qml.equal(split[1].measurements[0], qml.var(qml.PauliZ(0)))


class TestIntegration:
    """Integration tests for ``qml.transforms.split_non_commuting()``"""

    def test_expval_non_commuting_observables(self):
        """Test expval with multiple non-commuting operators"""
        dev = qml.device("default.qubit", wires=6)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(1)
            qml.Hadamard(0)
            qml.PauliZ(0)
            qml.Hadamard(3)
            qml.Hadamard(5)
            qml.T(5)
            return (
                qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
                qml.expval(qml.PauliX(0)),
                qml.expval(qml.PauliZ(1)),
                qml.expval(qml.PauliX(1) @ qml.PauliX(4)),
                qml.expval(qml.PauliX(3)),
                qml.expval(qml.PauliY(5)),
            )

        res = circuit()
        assert isinstance(res, tuple)
        assert len(res) == 6
        assert all(isinstance(r, np.ndarray) for r in res)
        assert all(r.shape == () for r in res)

        res = qml.math.stack(res)

        assert all(np.isclose(res, np.array([0.0, -1.0, 0.0, 0.0, 1.0, 1 / np.sqrt(2)])))

    def test_expval_non_commuting_observables_qnode(self):
        """Test expval with multiple non-commuting operators as a tranform program on the qnode."""
        dev = qml.device("default.qubit", wires=6)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(1)
            qml.Hadamard(0)
            qml.PauliZ(0)
            qml.Hadamard(3)
            qml.Hadamard(5)
            qml.T(5)
            return (
                qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
                qml.expval(qml.PauliX(0)),
                qml.expval(qml.PauliZ(1)),
                qml.expval(qml.PauliX(1) @ qml.PauliX(4)),
                qml.expval(qml.PauliX(3)),
                qml.expval(qml.PauliY(5)),
            )

        res = split_non_commuting(circuit)()

        assert isinstance(res, tuple)
        assert len(res) == 6
        assert all(isinstance(r, np.ndarray) for r in res)
        assert all(r.shape == () for r in res)

        res = qml.math.stack(res)

        assert all(np.isclose(res, np.array([0.0, -1.0, 0.0, 0.0, 1.0, 1 / np.sqrt(2)])))

    def test_shot_vector_support(self):
        """Test output is correct when using shot vectors"""

        dev = qml.device("default.qubit", wires=6, shots=(10000, (20000, 2), 30000))

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(1)
            qml.Hadamard(0)
            qml.PauliZ(0)
            qml.Hadamard(3)
            qml.Hadamard(5)
            qml.T(5)
            return (
                qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
                qml.expval(qml.PauliX(0)),
                qml.expval(qml.PauliZ(1)),
                qml.expval(
                    qml.PauliY(0) @ qml.PauliY(1) @ qml.PauliZ(3) @ qml.PauliY(4) @ qml.PauliX(5)
                ),
                qml.expval(qml.PauliX(1) @ qml.PauliX(4)),
                qml.expval(qml.PauliX(3)),
                qml.expval(qml.PauliY(5)),
            )

        res = circuit()
        assert isinstance(res, tuple)
        assert len(res) == 4
        assert all(isinstance(shot_res, tuple) for shot_res in res)
        assert all(len(shot_res) == 7 for shot_res in res)
        # pylint:disable=not-an-iterable
        assert all(
            all(list(isinstance(r, np.ndarray) and r.shape == () for r in shot_res))
            for shot_res in res
        )

        res = qml.math.stack([qml.math.stack(r) for r in res])

        assert np.allclose(
            res, np.array([0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1 / np.sqrt(2)]), atol=0.05
        )


# Autodiff tests
exp_res = np.array([0.77015115, -0.47942554, 0.87758256])
exp_grad = np.array(
    [[-4.20735492e-01, -4.20735492e-01], [-8.77582562e-01, 0.0], [-4.79425539e-01, 0.0]]
)


class TestAutodiffSplitNonCommuting:
    """Autodiff tests for all frameworks"""

    @pytest.mark.autograd
    def test_split_with_autograd(self):
        """Test that results after splitting are still differentiable with autograd"""
        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=1)
            return (
                qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
                qml.expval(qml.PauliY(0)),
                qml.expval(qml.PauliZ(0)),
            )

        def cost(params):
            res = circuit(params)
            return qml.math.stack(res)

        params = pnp.array([0.5, 0.5])
        res = cost(params)
        grad = qml.jacobian(cost)(params)
        assert all(np.isclose(res, exp_res))
        assert all(np.isclose(grad, exp_grad).flatten())

    @pytest.mark.jax
    def test_split_with_jax(self):
        """Test that results after splitting are still differentiable with jax"""

        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit.jax", wires=3)

        @qml.qnode(dev)
        def circuit(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=1)
            return (
                qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
                qml.expval(qml.PauliY(0)),
                qml.expval(qml.PauliZ(0)),
            )

        params = jnp.array([0.5, 0.5])
        res = circuit(params)
        grad = jax.jacobian(circuit)(params)
        assert all(np.isclose(res, exp_res))
        assert all(np.isclose(grad, exp_grad, atol=1e-5).flatten())

    @pytest.mark.jax
    def test_split_with_jax_multi_params(self):
        """Test that results after splitting are still differentiable with jax
        with multiple parameters"""

        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit.jax", wires=3)

        @qml.qnode(dev)
        def circuit(x, y):
            qml.RX(x, wires=0)
            qml.RY(y, wires=1)
            return (
                qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
                qml.expval(qml.PauliY(0)),
                qml.expval(qml.PauliZ(0)),
            )

        x = jnp.array(0.5)
        y = jnp.array(0.5)

        res = circuit(x, y)
        grad = jax.jacobian(circuit, argnums=[0, 1])(x, y)

        assert all(np.isclose(res, exp_res))

        assert isinstance(grad, tuple)
        assert len(grad) == 3

        for i, meas_grad in enumerate(grad):
            assert isinstance(meas_grad, tuple)
            assert len(meas_grad) == 2
            assert all(isinstance(g, jnp.ndarray) and g.shape == () for g in meas_grad)

            assert np.allclose(meas_grad, exp_grad[i], atol=1e-5)

    @pytest.mark.jax
    def test_split_with_jax_jit(self):
        """Test that results after splitting are still differentiable with jax-jit"""

        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit", wires=3)

        @jax.jit
        @qml.qnode(dev)
        def circuit(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=1)
            return (
                qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
                qml.expval(qml.PauliY(0)),
                qml.expval(qml.PauliZ(0)),
            )

        params = jnp.array([0.5, 0.5])
        res = circuit(params)
        grad = jax.jacobian(circuit)(params)
        assert all(np.isclose(res, exp_res))
        assert all(np.isclose(grad, exp_grad, atol=1e-5).flatten())

    @pytest.mark.torch
    def test_split_with_torch(self):
        """Test that results after splitting are still differentiable with torch"""

        import torch
        from torch.autograd.functional import jacobian

        dev = qml.device("default.qubit.torch", wires=3)

        @qml.qnode(dev)
        def circuit(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=1)
            return (
                qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
                qml.expval(qml.PauliY(0)),
                qml.expval(qml.PauliZ(0)),
            )

        def cost(params):
            res = circuit(params)
            return qml.math.stack(res)

        params = torch.tensor([0.5, 0.5], requires_grad=True)
        res = cost(params)
        grad = jacobian(cost, (params))
        assert all(np.isclose(res.detach().numpy(), exp_res))
        assert all(np.isclose(grad.detach().numpy(), exp_grad, atol=1e-5).flatten())

    @pytest.mark.tf
    def test_split_with_tf(self):
        """Test that results after splitting are still differentiable with tf"""

        import tensorflow as tf

        dev = qml.device("default.qubit.tf", wires=3)

        @qml.qnode(dev)
        def circuit(params):
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=1)
            return (
                qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
                qml.expval(qml.PauliY(0)),
                qml.expval(qml.PauliZ(0)),
            )

        params = tf.Variable([0.5, 0.5])
        res = circuit(params)
        with tf.GradientTape() as tape:
            loss = circuit(params)
            loss = tf.stack(loss)

        grad = tape.jacobian(loss, params)
        assert all(np.isclose(res, exp_res))
        assert all(np.isclose(grad, exp_grad, atol=1e-5).flatten())
