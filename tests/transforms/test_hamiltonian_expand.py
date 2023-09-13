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
"""
Unit tests for the ``hamiltonian_expand`` transform.
"""
import numpy as np
import pytest

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.queuing import AnnotatedQueue
from pennylane.tape import QuantumScript
from pennylane.transforms import hamiltonian_expand, sum_expand

# Defines the device used for all tests
dev = qml.device("default.qubit", wires=4)

# Defines circuits to be used in queueing/output tests
with AnnotatedQueue() as q_tape1:
    qml.PauliX(0)
    H1 = qml.Hamiltonian([1.5], [qml.PauliZ(0) @ qml.PauliZ(1)])
    qml.expval(H1)
tape1 = QuantumScript.from_queue(q_tape1)

with AnnotatedQueue() as q_tape2:
    qml.Hadamard(0)
    qml.Hadamard(1)
    qml.PauliZ(1)
    qml.PauliX(2)
    H2 = qml.Hamiltonian(
        [1, 3, -2, 1, 1],
        [
            qml.PauliX(0) @ qml.PauliZ(2),
            qml.PauliZ(2),
            qml.PauliX(0),
            qml.PauliX(2),
            qml.PauliZ(0) @ qml.PauliX(1),
        ],
    )
    qml.expval(H2)
tape2 = QuantumScript.from_queue(q_tape2)

H3 = 1.5 * qml.PauliZ(0) @ qml.PauliZ(1) + 0.3 * qml.PauliX(1)

with AnnotatedQueue() as q3:
    qml.PauliX(0)
    qml.expval(H3)


tape3 = QuantumScript.from_queue(q3)
H4 = (
    qml.PauliX(0) @ qml.PauliZ(2)
    + 3 * qml.PauliZ(2)
    - 2 * qml.PauliX(0)
    + qml.PauliZ(2)
    + qml.PauliZ(2)
)
H4 += qml.PauliZ(0) @ qml.PauliX(1) @ qml.PauliY(2)

with AnnotatedQueue() as q4:
    qml.Hadamard(0)
    qml.Hadamard(1)
    qml.PauliZ(1)
    qml.PauliX(2)

    qml.expval(H4)

tape4 = QuantumScript.from_queue(q4)
TAPES = [tape1, tape2, tape3, tape4]
OUTPUTS = [-1.5, -6, -1.5, -8]


class TestHamiltonianExpand:
    """Tests for the hamiltonian_expand transform"""

    def test_ham_with_no_terms_raises(self):
        """Tests that the hamiltonian_expand transform raises an error for a Hamiltonian with no terms."""
        mps = [qml.expval(qml.Hamiltonian([], []))]
        qscript = QuantumScript([], mps)

        with pytest.raises(
            ValueError,
            match="The Hamiltonian in the tape has no terms defined - cannot perform the Hamiltonian expansion.",
        ):
            qml.transforms.hamiltonian_expand(qscript)

    @pytest.mark.parametrize(("tape", "output"), zip(TAPES, OUTPUTS))
    def test_hamiltonians(self, tape, output):
        """Tests that the hamiltonian_expand transform returns the correct value"""

        tapes, fn = hamiltonian_expand(tape)
        results = dev.execute(tapes)
        expval = fn(results)

        assert np.isclose(output, expval)

        qs = QuantumScript(tape.operations, tape.measurements)
        tapes, fn = hamiltonian_expand(qs)
        results = dev.execute(tapes)
        expval = fn(results)
        assert np.isclose(output, expval)

    @pytest.mark.parametrize(("tape", "output"), zip(TAPES, OUTPUTS))
    def test_hamiltonians_no_grouping(self, tape, output):
        """Tests that the hamiltonian_expand transform returns the correct value
        if we switch grouping off"""

        tapes, fn = hamiltonian_expand(tape, group=False)
        results = dev.execute(tapes)
        expval = fn(results)

        assert np.isclose(output, expval)

        qs = QuantumScript(tape.operations, tape.measurements)
        tapes, fn = hamiltonian_expand(qs, group=False)
        results = dev.execute(tapes)
        expval = fn(results)

        assert np.isclose(output, expval)

    def test_grouping_is_used(self):
        """Test that the grouping in a Hamiltonian is used"""
        H = qml.Hamiltonian(
            [1.0, 2.0, 3.0], [qml.PauliZ(0), qml.PauliX(1), qml.PauliX(0)], grouping_type="qwc"
        )
        assert H.grouping_indices is not None

        with AnnotatedQueue() as q:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=2)
            qml.expval(H)

        tape = QuantumScript.from_queue(q)
        tapes, _ = hamiltonian_expand(tape, group=False)
        assert len(tapes) == 2

        qs = QuantumScript(tape.operations, tape.measurements)
        tapes, _ = hamiltonian_expand(qs, group=False)
        assert len(tapes) == 2

    def test_number_of_tapes(self):
        """Tests that the the correct number of tapes is produced"""

        H = qml.Hamiltonian([1.0, 2.0, 3.0], [qml.PauliZ(0), qml.PauliX(1), qml.PauliX(0)])

        with AnnotatedQueue() as q:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=2)
            qml.expval(H)

        tape = QuantumScript.from_queue(q)
        tapes, _ = hamiltonian_expand(tape, group=False)
        assert len(tapes) == 3

        tapes, _ = hamiltonian_expand(tape, group=True)
        assert len(tapes) == 2

    def test_number_of_qscripts(self):
        """Tests the correct number of quantum scripts are produced."""

        H = qml.Hamiltonian([1.0, 2.0, 3.0], [qml.PauliZ(0), qml.PauliX(1), qml.PauliX(0)])
        qs = QuantumScript(measurements=[qml.expval(H)])

        tapes, _ = hamiltonian_expand(qs, group=False)
        assert len(tapes) == 3

        tapes, _ = hamiltonian_expand(qs, group=True)
        assert len(tapes) == 2

    @pytest.mark.parametrize("shots", [None, 100])
    @pytest.mark.parametrize("group", [True, False])
    def test_shots_attribute(self, shots, group):
        """Tests that the shots attribute is copied to the new tapes"""
        H = qml.Hamiltonian([1.0, 2.0, 3.0], [qml.PauliZ(0), qml.PauliX(1), qml.PauliX(0)])

        with AnnotatedQueue() as q:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=2)
            qml.expval(H)

        tape = QuantumScript.from_queue(q, shots=shots)
        new_tapes, _ = hamiltonian_expand(tape, group=group)

        assert all(new_tape.shots == tape.shots for new_tape in new_tapes)

    def test_hamiltonian_error(self):
        """Tests that the script passed to hamiltonian_expand must end with a hamiltonian."""
        qscript = QuantumScript(measurements=[qml.expval(qml.PauliZ(0))])

        with pytest.raises(ValueError, match=r"Passed tape must end in"):
            qml.transforms.hamiltonian_expand(qscript)

    @pytest.mark.autograd
    def test_hamiltonian_dif_autograd(self, tol):
        """Tests that the hamiltonian_expand tape transform is differentiable with the Autograd interface"""

        H = qml.Hamiltonian(
            [-0.2, 0.5, 1], [qml.PauliX(1), qml.PauliZ(1) @ qml.PauliY(2), qml.PauliZ(0)]
        )

        var = pnp.array([0.1, 0.67, 0.3, 0.4, -0.5, 0.7, -0.2, 0.5, 1.0], requires_grad=True)
        output = 0.42294409781940356
        output2 = [
            9.68883500e-02,
            -2.90832724e-01,
            -1.04448033e-01,
            -1.94289029e-09,
            3.50307411e-01,
            -3.41123470e-01,
            0.0,
            -0.43657,
            0.64123,
        ]

        with AnnotatedQueue() as q:
            for _ in range(2):
                qml.RX(np.array(0), wires=0)
                qml.RX(np.array(0), wires=1)
                qml.RX(np.array(0), wires=2)
                qml.CNOT(wires=[0, 1])
                qml.CNOT(wires=[1, 2])
                qml.CNOT(wires=[2, 0])

            qml.expval(H)

        tape = QuantumScript.from_queue(q)

        def cost(x):
            new_tape = tape.bind_new_parameters(x, list(range(9)))
            tapes, fn = hamiltonian_expand(new_tape)
            res = qml.execute(tapes, dev, qml.gradients.param_shift)
            return fn(res)

        assert np.isclose(cost(var), output)

        grad = qml.grad(cost)(var)
        assert len(grad) == len(output2)
        for g, o in zip(grad, output2):
            assert np.allclose(g, o, atol=tol)

    @pytest.mark.tf
    def test_hamiltonian_dif_tensorflow(self):
        """Tests that the hamiltonian_expand tape transform is differentiable with the Tensorflow interface"""

        import tensorflow as tf

        H = qml.Hamiltonian(
            [-0.2, 0.5, 1], [qml.PauliX(1), qml.PauliZ(1) @ qml.PauliY(2), qml.PauliZ(0)]
        )
        var = tf.Variable([[0.1, 0.67, 0.3], [0.4, -0.5, 0.7]], dtype=tf.float64)
        output = 0.42294409781940356
        output2 = [
            9.68883500e-02,
            -2.90832724e-01,
            -1.04448033e-01,
            -1.94289029e-09,
            3.50307411e-01,
            -3.41123470e-01,
        ]

        with tf.GradientTape() as gtape:
            with AnnotatedQueue() as q:
                for _i in range(2):
                    qml.RX(var[_i, 0], wires=0)
                    qml.RX(var[_i, 1], wires=1)
                    qml.RX(var[_i, 2], wires=2)
                    qml.CNOT(wires=[0, 1])
                    qml.CNOT(wires=[1, 2])
                    qml.CNOT(wires=[2, 0])
                qml.expval(H)

            tape = QuantumScript.from_queue(q)
            tapes, fn = hamiltonian_expand(tape)
            res = fn(qml.execute(tapes, dev, qml.gradients.param_shift))

            assert np.isclose(res, output)

            g = gtape.gradient(res, var)
            assert np.allclose(list(g[0]) + list(g[1]), output2)


with AnnotatedQueue() as s_tape1:
    qml.PauliX(0)
    S1 = qml.s_prod(1.5, qml.prod(qml.PauliZ(0), qml.PauliZ(1)))
    qml.expval(S1)
    qml.expval(S1)
    qml.state()

with AnnotatedQueue() as s_tape2:
    qml.Hadamard(0)
    qml.Hadamard(1)
    qml.PauliZ(1)
    qml.PauliX(2)
    S2 = qml.sum(
        qml.prod(qml.PauliX(0), qml.PauliZ(2)),
        qml.s_prod(3, qml.PauliZ(2)),
        qml.s_prod(-2, qml.PauliX(0)),
        qml.PauliX(2),
        qml.prod(qml.PauliZ(0), qml.PauliX(1)),
    )
    qml.expval(S2)
    qml.probs(op=qml.PauliZ(0))
    qml.expval(S2)

S3 = qml.sum(
    qml.s_prod(1.5, qml.prod(qml.PauliZ(0), qml.PauliZ(1))), qml.s_prod(0.3, qml.PauliX(1))
)

with AnnotatedQueue() as s_tape3:
    qml.PauliX(0)
    qml.expval(S3)
    qml.probs(wires=[1, 3])
    qml.expval(qml.PauliX(1))
    qml.expval(S3)
    qml.probs(op=qml.PauliY(0))


S4 = qml.sum(
    qml.prod(qml.PauliX(0), qml.PauliZ(2)),
    qml.s_prod(3, qml.PauliZ(2)),
    qml.s_prod(-2, qml.PauliX(0)),
    qml.PauliZ(2),
    qml.PauliZ(2),
    qml.prod(qml.PauliZ(0), qml.PauliX(1), qml.PauliY(2)),
)

with AnnotatedQueue() as s_tape4:
    qml.Hadamard(0)
    qml.Hadamard(1)
    qml.PauliZ(1)
    qml.PauliX(2)

    qml.expval(S4)
    qml.expval(qml.PauliX(2))
    qml.expval(S4)
    qml.expval(qml.PauliX(2))

s_qscript1 = QuantumScript.from_queue(s_tape1)
s_qscript2 = QuantumScript.from_queue(s_tape2)
s_qscript3 = QuantumScript.from_queue(s_tape3)
s_qscript4 = QuantumScript.from_queue(s_tape4)

SUM_QSCRIPTS = [s_qscript1, s_qscript2, s_qscript3, s_qscript4]
SUM_OUTPUTS = [
    [
        -1.5,
        -1.5,
        np.array(
            [
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                1.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
            ]
        ),
    ],
    [-6, np.array([0.5, 0.5]), -6],
    [-1.5, np.array([1.0, 0.0, 0.0, 0.0]), 0.0, -1.5, np.array([0.5, 0.5])],
    [-8, 0, -8, 0],
]


class TestSumExpand:
    """Tests for the sum_expand transform"""

    def test_observables_on_same_wires(self):
        """Test that even if the observables are on the same wires, if they are different operations, they are separated.
        This is testing for a case that gave rise to a bug that occured due to a problem in MeasurementProcess.hash.
        """
        obs1 = qml.prod(qml.PauliX(0), qml.PauliX(1))
        obs2 = qml.prod(qml.PauliX(0), qml.PauliY(1))

        circuit = QuantumScript(measurements=[qml.expval(obs1), qml.expval(obs2)])
        batch, _ = sum_expand(circuit)
        assert len(batch) == 2
        assert qml.equal(batch[0][0], qml.expval(obs1))
        assert qml.equal(batch[1][0], qml.expval(obs2))

    @pytest.mark.parametrize(("qscript", "output"), zip(SUM_QSCRIPTS, SUM_OUTPUTS))
    def test_sums(self, qscript, output):
        """Tests that the sum_expand transform returns the correct value"""
        processed, _ = dev.preprocess()[0]([qscript])
        assert len(processed) == 1
        qscript = processed[0]
        tapes, fn = sum_expand(qscript)
        results = dev.execute(tapes)
        expval = fn(results)

        assert all(qml.math.allclose(o, e) for o, e in zip(output, expval))

    @pytest.mark.parametrize(("qscript", "output"), zip(SUM_QSCRIPTS, SUM_OUTPUTS))
    def test_sums_no_grouping(self, qscript, output):
        """Tests that the sum_expand transform returns the correct value
        if we switch grouping off"""
        processed, _ = dev.preprocess()[0]([qscript])
        assert len(processed) == 1
        qscript = processed[0]
        tapes, fn = sum_expand(qscript, group=False)
        results = dev.execute(tapes)
        expval = fn(results)

        assert all(qml.math.allclose(o, e) for o, e in zip(output, expval))

    def test_grouping(self):
        """Test the grouping functionality"""
        S = qml.sum(qml.PauliZ(0), qml.s_prod(2, qml.PauliX(1)), qml.s_prod(3, qml.PauliX(0)))

        with AnnotatedQueue() as q:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=2)
            qml.expval(S)

        qscript = QuantumScript.from_queue(q)

        tapes, _ = sum_expand(qscript, group=True)
        assert len(tapes) == 2

    def test_number_of_qscripts(self):
        """Tests the correct number of quantum scripts are produced."""

        S = qml.sum(qml.PauliZ(0), qml.s_prod(2, qml.PauliX(1)), qml.s_prod(3, qml.PauliX(0)))
        qs = QuantumScript(measurements=[qml.expval(S)])

        tapes, _ = sum_expand(qs, group=False)
        assert len(tapes) == 3

        tapes, _ = sum_expand(qs, group=True)
        assert len(tapes) == 2

    @pytest.mark.parametrize("shots", [None, 100])
    @pytest.mark.parametrize("group", [True, False])
    def test_shots_attribute(self, shots, group):
        """Tests that the shots attribute is copied to the new tapes"""
        H = qml.Hamiltonian([1.0, 2.0, 3.0], [qml.PauliZ(0), qml.PauliX(1), qml.PauliX(0)])

        with AnnotatedQueue() as q:
            qml.Hadamard(wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PauliX(wires=2)
            qml.expval(H)

        tape = QuantumScript.from_queue(q, shots=shots)
        new_tapes, _ = sum_expand(tape, group=group)

        assert all(new_tape.shots == tape.shots for new_tape in new_tapes)

    def test_non_sum_tape(self):
        """Test that the ``sum_expand`` function returns the input tape if it does not
        contain a single measurement with the expectation value of a Sum."""

        with AnnotatedQueue() as q:
            qml.expval(qml.PauliZ(0))

        tape = QuantumScript.from_queue(q)

        tapes, fn = sum_expand(tape)

        assert len(tapes) == 1
        assert isinstance(list(tapes[0])[0].obs, qml.PauliZ)
        # Old return types return a list for a single value:
        # e.g. qml.expval(qml.PauliX(0)) = [1.23]
        res = [1.23]
        assert fn(res) == 1.23

    def test_multiple_sum_tape(self):
        """Test that the ``sum_expand`` function can expand tapes with multiple sum observables"""

    @pytest.mark.autograd
    def test_sum_dif_autograd(self, tol):
        """Tests that the sum_expand tape transform is differentiable with the Autograd interface"""
        S = qml.sum(
            qml.s_prod(-0.2, qml.PauliX(1)),
            qml.s_prod(0.5, qml.prod(qml.PauliZ(1), qml.PauliY(2))),
            qml.s_prod(1, qml.PauliZ(0)),
        )

        var = pnp.array([0.1, 0.67, 0.3, 0.4, -0.5, 0.7, -0.2, 0.5, 1], requires_grad=True)
        output = 0.42294409781940356
        output2 = [
            9.68883500e-02,
            -2.90832724e-01,
            -1.04448033e-01,
            -1.94289029e-09,
            3.50307411e-01,
            -3.41123470e-01,
            0.0,
            -4.36578753e-01,
            6.41233474e-01,
        ]

        with AnnotatedQueue() as q:
            for _ in range(2):
                qml.RX(np.array(0), wires=0)
                qml.RX(np.array(0), wires=1)
                qml.RX(np.array(0), wires=2)
                qml.CNOT(wires=[0, 1])
                qml.CNOT(wires=[1, 2])
                qml.CNOT(wires=[2, 0])

            qml.expval(S)

        qscript = QuantumScript.from_queue(q)

        def cost(x):
            new_qscript = qscript.bind_new_parameters(x, list(range(9)))
            tapes, fn = sum_expand(new_qscript)
            res = qml.execute(tapes, dev, qml.gradients.param_shift)
            return fn(res)

        assert np.isclose(cost(var), output)

        grad = qml.grad(cost)(var)
        assert len(grad) == len(output2)
        for g, o in zip(grad, output2):
            assert np.allclose(g, o, atol=tol)

    @pytest.mark.tf
    def test_sum_dif_tensorflow(self):
        """Tests that the sum_expand tape transform is differentiable with the Tensorflow interface"""

        import tensorflow as tf

        S = qml.sum(
            qml.s_prod(-0.2, qml.PauliX(1)),
            qml.s_prod(0.5, qml.prod(qml.PauliZ(1), qml.PauliY(2))),
            qml.s_prod(1, qml.PauliZ(0)),
        )
        var = tf.Variable([[0.1, 0.67, 0.3], [0.4, -0.5, 0.7]], dtype=tf.float64)
        output = 0.42294409781940356
        output2 = [
            9.68883500e-02,
            -2.90832724e-01,
            -1.04448033e-01,
            -1.94289029e-09,
            3.50307411e-01,
            -3.41123470e-01,
        ]

        with tf.GradientTape() as gtape:
            with AnnotatedQueue() as q:
                for _i in range(2):
                    qml.RX(var[_i, 0], wires=0)
                    qml.RX(var[_i, 1], wires=1)
                    qml.RX(var[_i, 2], wires=2)
                    qml.CNOT(wires=[0, 1])
                    qml.CNOT(wires=[1, 2])
                    qml.CNOT(wires=[2, 0])
                qml.expval(S)

            qscript = QuantumScript.from_queue(q)
            tapes, fn = sum_expand(qscript)
            res = fn(qml.execute(tapes, dev, qml.gradients.param_shift))

            assert np.isclose(res, output)

            g = gtape.gradient(res, var)
            assert np.allclose(list(g[0]) + list(g[1]), output2)

    @pytest.mark.jax
    def test_sum_dif_jax(self, tol):
        """Tests that the sum_expand tape transform is differentiable with the Jax interface"""
        import jax
        from jax import numpy as jnp

        S = qml.sum(
            qml.s_prod(-0.2, qml.PauliX(1)),
            qml.s_prod(0.5, qml.prod(qml.PauliZ(1), qml.PauliY(2))),
            qml.s_prod(1, qml.PauliZ(0)),
        )

        var = jnp.array([0.1, 0.67, 0.3, 0.4, -0.5, 0.7, -0.2, 0.5, 1])
        output = 0.42294409781940356
        output2 = [
            9.68883500e-02,
            -2.90832724e-01,
            -1.04448033e-01,
            -1.94289029e-09,
            3.50307411e-01,
            -3.41123470e-01,
            0.0,
            -4.36578753e-01,
            6.41233474e-01,
        ]

        with AnnotatedQueue() as q:
            for _ in range(2):
                qml.RX(np.array(0), wires=0)
                qml.RX(np.array(0), wires=1)
                qml.RX(np.array(0), wires=2)
                qml.CNOT(wires=[0, 1])
                qml.CNOT(wires=[1, 2])
                qml.CNOT(wires=[2, 0])

            qml.expval(S)

        qscript = QuantumScript.from_queue(q)

        def cost(x):
            new_qscript = qscript.bind_new_parameters(x, list(range(9)))
            tapes, fn = sum_expand(new_qscript)
            res = qml.execute(tapes, dev, qml.gradients.param_shift)
            return fn(res)

        assert np.isclose(cost(var), output)

        grad = jax.grad(cost)(var)
        assert len(grad) == len(output2)
        for g, o in zip(grad, output2):
            assert np.allclose(g, o, atol=tol)
