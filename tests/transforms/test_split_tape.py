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

import numpy as np
import pytest

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.ops import Hamiltonian
from pennylane.queuing import AnnotatedQueue
from pennylane.tape import QuantumScript
from pennylane.transforms import split_tape

dev = qml.device("default.qubit", wires=4)
"""Defines the device used for all tests"""

H1 = qml.Hamiltonian([1.5, 1.5], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)])
S1 = qml.s_prod(1.5, qml.prod(qml.PauliZ(0), qml.PauliZ(1))) + qml.s_prod(
    1.5, qml.prod(qml.PauliZ(0), qml.PauliZ(1))
)

"""Defines circuits to be used in queueing/output tests"""
tape1 = QuantumScript(
    [qml.PauliX(0)], [qml.expval(H1), qml.expval(S1), qml.expval(S1), qml.state()]
)

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
S2 = qml.op_sum(
    qml.prod(qml.PauliX(0), qml.PauliZ(2)),
    qml.s_prod(3, qml.PauliZ(2)),
    qml.s_prod(-2, qml.PauliX(0)),
    qml.PauliX(2),
    qml.prod(qml.PauliZ(0), qml.PauliX(1)),
)
tape2 = QuantumScript(
    [qml.Hadamard(0), qml.Hadamard(1), qml.PauliZ(1), qml.PauliX(2)],
    [qml.expval(H2), qml.expval(S2), qml.probs(op=qml.PauliZ(0)), qml.expval(S2)],
)

H3 = 1.5 * qml.PauliZ(0) @ qml.PauliZ(1) + 0.3 * qml.PauliX(1)
S3 = qml.op_sum(
    qml.s_prod(1.5, qml.prod(qml.PauliZ(0), qml.PauliZ(1))), qml.s_prod(0.3, qml.PauliX(1))
)

tape3 = QuantumScript(
    [qml.PauliX(0)],
    [
        qml.expval(H3),
        qml.expval(S3),
        qml.probs(wires=1),
        qml.expval(qml.PauliX(1)),
        qml.expval(S3),
        qml.probs(wires=[0, 1]),
    ],
)
H4 = (
    qml.PauliX(0) @ qml.PauliZ(2)
    + 3 * qml.PauliZ(2)
    - 2 * qml.PauliX(0)
    + qml.PauliZ(2)
    + qml.PauliZ(2)
    + qml.PauliZ(0) @ qml.PauliX(1) @ qml.PauliY(2)
)
S4 = (
    qml.prod(qml.PauliX(0), qml.PauliZ(2))
    + qml.s_prod(3, qml.PauliZ(2))
    - qml.s_prod(2, qml.PauliX(0))
    + qml.PauliZ(2)
    + qml.PauliZ(2)
    + qml.prod(qml.PauliZ(0), qml.PauliX(1), qml.PauliY(2))
)

tape4 = QuantumScript(
    [qml.Hadamard(0), qml.Hadamard(1), qml.PauliZ(1), qml.PauliX(2)],
    [
        qml.expval(H4),
        qml.expval(S4),
        qml.expval(qml.PauliX(2)),
        qml.expval(S4),
        qml.expval(qml.PauliX(2)),
    ],
)
tape5 = QuantumScript(
    [qml.PauliX(0)],
    [qml.expval(H1), qml.expval(H3), qml.expval(S1), qml.expval(S3)],
)
tape6 = QuantumScript(
    [qml.Hadamard(0), qml.Hadamard(1), qml.PauliZ(1), qml.PauliX(2)],
    [qml.expval(H2), qml.expval(H4), qml.expval(S2), qml.expval(S4)],
)
TAPES = [tape1, tape2, tape3, tape4, tape5, tape6]
OUTPUTS = [
    [
        -3.0,
        -3.0,
        -3.0,
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
    [-6, -6, np.array([0.5, 0.5]), -6],
    [-1.5, -1.5, np.array([1.0, 0.0]), 0.0, -1.5, np.array([0.0, 0.0, 1.0, 0.0])],
    [-8, -8, 0, -8, 0],
    [-3.0, -1.5, -3.0, -1.5],
    [-6, -8, -6, -8],
]


class TestSplitTape:
    """Tests for the split_tape transform"""

    @pytest.mark.parametrize(("tape", "output"), zip(TAPES, OUTPUTS))
    def test_tapes(self, tape, output):
        """Tests that the split_tape transform returns the correct value"""
        tapes, fn = split_tape(tape, group=True)
        tapes = [q.expand() for q in tapes]
        results = dev.batch_execute(tapes)
        expval = fn(results)

        assert all(qml.math.allclose(o, e) for o, e in zip(output, expval))

    @pytest.mark.parametrize(("tape", "output"), zip(TAPES, OUTPUTS))
    def test_no_grouping(self, tape, output):
        """Tests that the split_tape transform returns the correct value
        if we switch grouping off"""

        tapes, fn = split_tape(tape, group=False)
        tapes = [q.expand() for q in tapes]
        results = dev.batch_execute(tapes)
        expval = fn(results)

        assert all(qml.math.allclose(o, e) for o, e in zip(output, expval))

    @pytest.mark.parametrize(("tape", "output"), zip(TAPES, OUTPUTS))
    def test_hamiltonian_grouping_indices_and_grouping(self, tape, output):
        """Tests that the split_tape transform returns the correct value
        if we use ``Hamiltonian.grouping_indices`` and ``group=True``"""

        for m in tape.measurements:
            if isinstance(m.obs, Hamiltonian):
                m.obs.compute_grouping()

        tapes, fn = split_tape(tape, group=True)
        tapes = [q.expand() for q in tapes]
        results = dev.batch_execute(tapes)
        expval = fn(results)

        assert all(qml.math.allclose(o, e) for o, e in zip(output, expval))

    @pytest.mark.parametrize(("tape", "output"), zip(TAPES, OUTPUTS))
    def test_hamiltonian_grouping_indices_and_no_grouping(self, tape, output):
        """Tests that the split_tape transform returns the correct value
        if we use ``Hamiltonian.grouping_indices`` and ``group=False``"""

        for m in tape.measurements:
            if isinstance(m.obs, Hamiltonian):
                m.obs.compute_grouping()

        tapes, fn = split_tape(tape, group=False)
        tapes = [q.expand() for q in tapes]
        results = dev.batch_execute(tapes)
        expval = fn(results)

        assert all(qml.math.allclose(o, e) for o, e in zip(output, expval))

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
        tapes, _ = split_tape(tape, group=False)
        assert len(tapes) == 2

    def test_number_of_tapes_with_grouping(self):
        """Test that that the correct number of tapes are generated."""
        H = qml.Hamiltonian([1.0, 2.0, 3.0], [qml.PauliZ(0), qml.PauliX(1), qml.PauliX(0)])

        with AnnotatedQueue() as q:
            qml.expval(H)
            qml.expval(qml.PauliX(0))
            qml.expval(qml.op_sum(qml.PauliX(0), qml.PauliZ(0), qml.PauliX(1)))
            qml.expval(H)
            qml.probs(op=qml.PauliX(0))
            qml.probs(op=qml.PauliZ(0) @ qml.PauliX(1))

        tape = QuantumScript.from_queue(q)
        tapes, _ = split_tape(tape, group=True)
        assert len(tapes) == 2

        grouped_measurements = [
            [
                qml.expval(qml.PauliZ(0)),
                qml.expval(qml.PauliX(1)),
                qml.probs(op=qml.PauliZ(0) @ qml.PauliX(1)),
            ],
            [qml.expval(qml.PauliX(0)), qml.probs(op=qml.PauliX(0))],
        ]
        for m1_list, m2_list in zip([tape.measurements for tape in tapes], grouped_measurements):
            assert all(qml.equal(m1, m2) for m1, m2 in zip(m1_list, m2_list))

        # When using ``Hamiltonian.grouping_indices``, we generate tapes for the previously computed
        # groups, and then use the ``group_observables`` method with the remaining pauli observables.
        # Consequently, we will obtain 4 tapes: 2 for the Hamiltonian groups and 2 for the
        # remaining ``probs(PauliZ(0) @ PauliX(1))`` and ``probs(PauliX(0)) measurements.
        # NOTE: The other expectation values have observables that are present in the Hamiltonian
        # group, and thus we avoid measuring them twice.
        H.compute_grouping()

        tape = QuantumScript.from_queue(q)
        tapes, _ = split_tape(tape, group=True)
        assert len(tapes) == 4

        grouped_measurements = [
            [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(1))],
            [qml.expval(qml.PauliX(0))],
            [qml.probs(op=qml.PauliX(0))],
            [qml.probs(op=qml.PauliZ(0) @ qml.PauliX(1))],
        ]
        for m1_list, m2_list in zip([tape.measurements for tape in tapes], grouped_measurements):
            assert all(qml.equal(m1, m2) for m1, m2 in zip(m1_list, m2_list))

    def test_number_of_tapes(self):
        """Tests the correct number of quantum tapes are produced."""

        H = qml.Hamiltonian([1.0, 2.0, 3.0], [qml.PauliZ(0), qml.PauliX(1), qml.PauliX(0)])
        S = qml.op_sum(qml.PauliZ(0), qml.s_prod(2, qml.PauliX(1)), qml.s_prod(3, qml.PauliX(0)))

        qs = QuantumScript(measurements=[qml.expval(H), qml.expval(S)])

        tapes, _ = split_tape(qs)
        assert len(tapes) == 2

    def test_non_ham_and_non_sum_tape(self):
        """Test that the ``split_tape`` function returns the input tape if it does not
        contain a single measurement with the expectation value of a Sum or a Hamiltonian."""

        tape = QuantumScript(measurements=[qml.expval(qml.PauliZ(0))])
        tapes, fn = split_tape(tape)

        assert len(tapes) == 1
        assert isinstance(list(tapes[0])[0].obs, qml.PauliZ)
        # Old return types return a list for a single value:
        # e.g. qml.expval(qml.PauliX(0)) = [1.23]
        res = [1.23]
        assert fn(res) == 1.23

    @pytest.mark.autograd
    def test_dif_autograd(self, tol):
        """Tests that the split_tape tape transform is differentiable with the Autograd interface"""

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

        with AnnotatedQueue() as q_tape:
            for _ in range(2):
                qml.RX(np.array(0), wires=0)
                qml.RX(np.array(0), wires=1)
                qml.RX(np.array(0), wires=2)
                qml.CNOT(wires=[0, 1])
                qml.CNOT(wires=[1, 2])
                qml.CNOT(wires=[2, 0])

            qml.expval(H)

        tape = QuantumScript.from_queue(q_tape)

        def cost(x):
            tape.set_parameters(x, trainable_only=False)
            tapes, fn = split_tape(tape)
            res = qml.execute(tapes, dev, qml.gradients.param_shift)
            return fn(res)

        assert np.allclose(cost(var), output)

        grad = qml.grad(cost)(var)
        assert len(grad) == len(output2)
        for g, o in zip(grad, output2):
            assert np.allclose(g, o, atol=tol)

    @pytest.mark.tf
    def test_dif_tensorflow(self):
        """Tests that the split_tape tape transform is differentiable with the Tensorflow interface"""

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
            with AnnotatedQueue() as q_tape:
                for i in range(2):
                    qml.RX(var[i, 0], wires=0)
                    qml.RX(var[i, 1], wires=1)
                    qml.RX(var[i, 2], wires=2)
                    qml.CNOT(wires=[0, 1])
                    qml.CNOT(wires=[1, 2])
                    qml.CNOT(wires=[2, 0])
                qml.expval(H)
            tape = QuantumScript.from_queue(q_tape)
            tapes, fn = split_tape(tape)
            res = fn(qml.execute(tapes, dev, qml.gradients.param_shift, interface="tf"))

            assert np.allclose(res, output)

            g = gtape.gradient(res, var)
            assert np.allclose(list(g[0]) + list(g[1]), output2)

    @pytest.mark.jax
    def test_sum_dif_jax(self, tol):
        """Tests that the split_tape transform is differentiable with the Jax interface"""
        import jax
        from jax import numpy as jnp

        S = qml.op_sum(
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
            0.0,
            0.0,
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

        tape = QuantumScript.from_queue(q)

        def cost(x):
            tape.set_parameters(x, trainable_only=False)
            tapes, fn = split_tape(tape)
            res = qml.execute(tapes, dev, qml.gradients.param_shift, interface="jax")
            return fn(res)

        assert np.isclose(cost(var), output)

        grad = jax.grad(cost)(var)
        assert len(grad) == len(output2)
        for g, o in zip(grad, output2):
            assert np.allclose(g, o, atol=tol)

    def test_hamiltonian_expand_deprecated(self):
        tape = QuantumScript()
        with pytest.warns(UserWarning, match="hamiltonian_expand function is deprecated"):
            _ = qml.transforms.hamiltonian_expand(tape)

        with pytest.warns(UserWarning, match="hamiltonian_expand function is deprecated"):
            from pennylane.transforms import hamiltonian_expand

        assert hamiltonian_expand is split_tape

    def test_sum_expand_deprecated(self):
        tape = QuantumScript()
        with pytest.warns(UserWarning, match="sum_expand function is deprecated"):
            _ = qml.transforms.sum_expand(tape)

        with pytest.warns(UserWarning, match="sum_expand function is deprecated"):
            from pennylane.transforms import sum_expand

        assert sum_expand is split_tape
