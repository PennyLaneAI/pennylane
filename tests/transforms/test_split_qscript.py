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
from pennylane.queuing import AnnotatedQueue
from pennylane.tape import QuantumScript
from pennylane.transforms import split_qscript

dev = qml.device("default.qubit", wires=4)
"""Defines the device used for all tests"""

H1 = qml.Hamiltonian([1.5, 1.5], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliZ(0) @ qml.PauliZ(1)])
S1 = qml.s_prod(1.5, qml.prod(qml.PauliZ(0), qml.PauliZ(1)))

"""Defines circuits to be used in queueing/output tests"""
with AnnotatedQueue() as q_qscript1:
    qml.PauliX(0)
    qml.expval(H1)
    qml.expval(S1)
    qml.expval(S1)
    qml.state()
qscript1 = QuantumScript.from_queue(q_qscript1)

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
with AnnotatedQueue() as q_qscript2:
    qml.Hadamard(0)
    qml.Hadamard(1)
    qml.PauliZ(1)
    qml.PauliX(2)
    qml.expval(H2)
    qml.expval(S2)
    qml.probs(op=qml.PauliZ(0))
    qml.expval(S2)
qscript2 = QuantumScript.from_queue(q_qscript2)

H3 = 1.5 * qml.PauliZ(0) @ qml.PauliZ(1) + 0.3 * qml.PauliX(1)
S3 = qml.op_sum(
    qml.s_prod(1.5, qml.prod(qml.PauliZ(0), qml.PauliZ(1))), qml.s_prod(0.3, qml.PauliX(1))
)

with AnnotatedQueue() as q_qscript3:
    qml.PauliX(0)
    qml.expval(H3)
    qml.expval(S3)
    qml.probs(wires=[1, 3])
    qml.expval(qml.PauliX(1))
    qml.expval(S3)
    qml.probs(op=qml.PauliY(0))


qscript3 = QuantumScript.from_queue(q_qscript3)
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

with AnnotatedQueue() as q_qscript4:
    qml.Hadamard(0)
    qml.Hadamard(1)
    qml.PauliZ(1)
    qml.PauliX(2)

    qml.expval(H4)
    qml.expval(S4)
    qml.expval(qml.PauliX(2))
    qml.expval(S4)
    qml.expval(qml.PauliX(2))

qscript4 = QuantumScript.from_queue(q_qscript4)
QSCRIPTS = [qscript1, qscript2, qscript3, qscript4]
OUTPUTS = [
    [
        -3.0,
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
    [-6, -6, np.array([0.5, 0.5]), -6],
    [-1.5, -1.5, np.array([1.0, 0.0, 0.0, 0.0]), 0.0, -1.5, np.array([0.5, 0.5])],
    [-8, -8, 0, -8, 0],
]


class TestSplitQscript:
    """Tests for the split_qscript transform"""

    @pytest.mark.parametrize(("qscript", "output"), zip(QSCRIPTS, OUTPUTS))
    def test_qscripts(self, qscript, output):
        """Tests that the split_qscript transform returns the correct value"""
        qscripts, fn = split_qscript(qscript)
        qscripts = [q.expand() for q in qscripts]
        results = dev.batch_execute(qscripts)
        expval = fn(results)

        assert all(qml.math.allclose(o, e) for o, e in zip(output, expval))

    @pytest.mark.parametrize(("qscript", "output"), zip(QSCRIPTS, OUTPUTS))
    def test_no_grouping(self, qscript, output):
        """Tests that the split_qscript transform returns the correct value
        if we switch grouping off"""

        qscripts, fn = split_qscript(qscript, group=False)
        qscripts = [q.expand() for q in qscripts]
        results = dev.batch_execute(qscripts)
        expval = fn(results)

        assert all(qml.math.allclose(o, e) for o, e in zip(output, expval))

    def test_number_of_qscripts(self):
        """Tests the correct number of quantum scripts are produced."""

        H = qml.Hamiltonian([1.0, 2.0, 3.0], [qml.PauliZ(0), qml.PauliX(1), qml.PauliX(0)])
        S = qml.op_sum(qml.PauliZ(0), qml.s_prod(2, qml.PauliX(1)), qml.s_prod(3, qml.PauliX(0)))

        qs = QuantumScript(measurements=[qml.expval(H), qml.expval(S)])

        qscripts, _ = split_qscript(qs, group=False)
        assert len(qscripts) == 3

        qscripts, _ = split_qscript(qs, group=True)
        assert len(qscripts) == 2

    def test_non_ham_and_non_sum_qscript(self):
        """Test that the ``split_qscript`` function returns the input qscript if it does not
        contain a single measurement with the expectation value of a Sum or a Hamiltonian."""

        qscript = QuantumScript(measurements=[qml.expval(qml.PauliZ(0))])
        qscripts, fn = split_qscript(qscript)

        assert len(qscripts) == 1
        assert isinstance(list(qscripts[0])[0].obs, qml.PauliZ)
        # Old return types return a list for a single value:
        # e.g. qml.expval(qml.PauliX(0)) = [1.23]
        res = [1.23] if qml.active_return() else [[1.23]]
        assert fn(res) == 1.23

    @pytest.mark.autograd
    def test_dif_autograd(self, tol):
        """Tests that the split_qscript qscript transform is differentiable with the Autograd interface"""

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

        with AnnotatedQueue() as q_qscript:
            for _ in range(2):
                qml.RX(np.array(0), wires=0)
                qml.RX(np.array(0), wires=1)
                qml.RX(np.array(0), wires=2)
                qml.CNOT(wires=[0, 1])
                qml.CNOT(wires=[1, 2])
                qml.CNOT(wires=[2, 0])

            qml.expval(H)

        qscript = QuantumScript.from_queue(q_qscript)

        def cost(x):
            qscript.set_parameters(x, trainable_only=False)
            qscripts, fn = split_qscript(qscript)
            res = qml.execute(qscripts, dev, qml.gradients.param_shift)
            return fn(res)

        assert np.allclose(cost(var), output)

        grad = qml.grad(cost)(var)
        assert len(grad) == len(output2)
        for g, o in zip(grad, output2):
            assert np.allclose(g, o, atol=tol)

    @pytest.mark.tf
    def test_dif_tensorflow(self):
        """Tests that the split_qscript qscript transform is differentiable with the Tensorflow interface"""

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

        with tf.Gradientqscript() as gqscript:
            with AnnotatedQueue() as q_qscript:
                for i in range(2):
                    qml.RX(var[i, 0], wires=0)
                    qml.RX(var[i, 1], wires=1)
                    qml.RX(var[i, 2], wires=2)
                    qml.CNOT(wires=[0, 1])
                    qml.CNOT(wires=[1, 2])
                    qml.CNOT(wires=[2, 0])
                qml.expval(H)
            qscript = QuantumScript.from_queue(q_qscript)
            qscripts, fn = split_qscript(qscript)
            res = fn(qml.execute(qscripts, dev, qml.gradients.param_shift, interface="tf"))

            assert np.allclose(res, output)

            g = gqscript.gradient(res, var)
            assert np.allclose(list(g[0]) + list(g[1]), output2)

    @pytest.mark.jax
    def test_sum_dif_jax(self, tol):
        """Tests that the split_qscript transform is differentiable with the Jax interface"""
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

        qscript = QuantumScript.from_queue(q)

        def cost(x):
            qscript.set_parameters(x, trainable_only=False)
            qscripts, fn = split_qscript(qscript)
            res = qml.execute(qscripts, dev, qml.gradients.param_shift, interface="jax")
            return fn(res)

        assert np.isclose(cost(var), output)

        grad = jax.grad(cost)(var)
        assert len(grad) == len(output2)
        for g, o in zip(grad, output2):
            assert np.allclose(g, o, atol=tol)
