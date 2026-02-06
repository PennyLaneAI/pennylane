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
Unit tests for the iterative_qpe function
"""
import numpy as np
import pytest

import pennylane as qp


class TestIQPE:
    """Test to check that the iterative quantum phase estimation function works as expected."""

    @pytest.mark.parametrize("mcm_method", ["deferred", "tree-traversal"])
    @pytest.mark.parametrize("phi", (1.0, 2.0, 3.0))
    def test_compare_qpe(self, mcm_method, phi):
        """Test to check that the results obtained are equivalent to those of QuantumPhaseEstimation"""

        dev = qp.device("default.qubit")

        @qp.qnode(dev, mcm_method=mcm_method)
        def circuit_iterative():
            # Initial state
            qp.PauliX(wires=[0])

            # Iterative QPE
            measurements = qp.iterative_qpe(qp.RZ(phi, wires=[0]), aux_wire=[1], iters=3)

            return qp.probs(op=measurements)

        output = circuit_iterative()

        @qp.qnode(dev)
        def circuit_qpe():
            # Initial state
            qp.PauliX(wires=[0])

            # Iterative QPE
            qp.QuantumPhaseEstimation(qp.RZ(phi, wires=[0]), estimation_wires=[1, 2, 3])

            return qp.probs(wires=[1, 2, 3])

        assert np.allclose(np.round(output, 2), np.round(circuit_qpe(), 2))

    @pytest.mark.jax
    def test_check_gradients_jax(self):
        """Test to check that the gradients are correct comparing with the expanded circuit using JAX"""

        import jax

        dev = qp.device("default.qubit")

        @qp.qnode(dev)
        def circuit(theta):
            meas = qp.iterative_qpe(qp.RZ(theta, wires=[0]), [1], iters=2)
            return qp.expval(meas[0])

        @qp.qnode(dev)
        def manual_circuit(phi):
            qp.Hadamard(wires=[1])
            qp.ctrl(qp.RZ(phi, wires=[0]) ** 2, control=[1])
            qp.Hadamard(wires=[1])
            qp.CNOT(wires=[1, 2])
            qp.CNOT(wires=[2, 1])
            qp.Hadamard(wires=[1])
            qp.ctrl(qp.RZ(phi, wires=[0]), control=[1])
            qp.ctrl(qp.PhaseShift(-np.pi / 2, wires=[1]), control=[2])
            qp.Hadamard(wires=[1])
            qp.CNOT(wires=[1, 3])
            qp.CNOT(wires=[3, 1])

            return qp.expval(qp.Hermitian([[0, 0], [0, 1]], wires=3))

        phi = jax.numpy.array(1.0)
        assert jax.numpy.isclose(jax.grad(circuit)(phi), jax.grad(manual_circuit)(phi))

    @pytest.mark.torch
    def test_check_gradients_torch(self):
        """Test to check that the gradients are correct comparing with the expanded circuit using PyTorch"""

        import torch

        dev = qp.device("default.qubit")

        @qp.qnode(dev)
        def circuit(theta):
            meas = qp.iterative_qpe(qp.RZ(theta, wires=[0]), [1], iters=2)
            return qp.expval(meas[0])

        @qp.qnode(dev)
        def manual_circuit(phi):
            qp.Hadamard(wires=[1])
            qp.ctrl(qp.RZ(phi, wires=[0]) ** 2, control=[1])
            qp.Hadamard(wires=[1])
            qp.CNOT(wires=[1, 2])
            qp.CNOT(wires=[2, 1])
            qp.Hadamard(wires=[1])
            qp.ctrl(qp.RZ(phi, wires=[0]), control=[1])
            qp.ctrl(qp.PhaseShift(-np.pi / 2, wires=[1]), control=[2])
            qp.Hadamard(wires=[1])
            qp.CNOT(wires=[1, 3])
            qp.CNOT(wires=[3, 1])

            return qp.expval(qp.Hermitian([[0, 0], [0, 1]], wires=3))

        phi = torch.tensor(1.0, requires_grad=True)
        assert torch.isclose(torch.func.grad(circuit)(phi), torch.func.grad(manual_circuit)(phi))

    @pytest.mark.tf
    def test_check_gradients_tf(self):
        """Test to check that the gradients are correct comparing with the expanded circuit using TensorFlow"""

        import tensorflow as tf

        def grad(f):
            def wrapper(x):
                with tf.GradientTape() as tape:
                    y = f(x)

                return tape.gradient(y, x)

            return wrapper

        dev = qp.device("default.qubit")

        @qp.qnode(dev)
        def circuit(theta):
            meas = qp.iterative_qpe(qp.RZ(theta, wires=[0]), [1], iters=2)
            return qp.expval(meas[0])

        @qp.qnode(dev)
        def manual_circuit(phi):
            qp.Hadamard(wires=[1])
            qp.ctrl(qp.RZ(phi, wires=[0]) ** 2, control=[1])
            qp.Hadamard(wires=[1])
            qp.CNOT(wires=[1, 2])
            qp.CNOT(wires=[2, 1])
            qp.Hadamard(wires=[1])
            qp.ctrl(qp.RZ(phi, wires=[0]), control=[1])
            qp.ctrl(qp.PhaseShift(-np.pi / 2, wires=[1]), control=[2])
            qp.Hadamard(wires=[1])
            qp.CNOT(wires=[1, 3])
            qp.CNOT(wires=[3, 1])

            return qp.expval(qp.Hermitian([[0, 0], [0, 1]], wires=3))

        phi = tf.Variable(1.0)
        assert np.isclose(grad(circuit)(phi), grad(manual_circuit)(phi))

    @pytest.mark.parametrize("iters", (1, 2, 3, 4))
    def test_size_return(self, iters):
        """Test to check that the size of the returned list is correct"""

        dev = qp.device("default.qubit")

        @qp.set_shots(1)
        @qp.qnode(dev, mcm_method="one-shot")
        def circuit():
            m = qp.iterative_qpe(qp.RZ(1.0, wires=[0]), [1], iters=iters)
            return [qp.sample(op=meas) for meas in m]

        assert len(circuit()) == iters

    @pytest.mark.parametrize("wire", (1, "a", "abc", 6))
    def test_wires_args(self, wire):
        """Test to check that all types of wires are working"""

        with qp.tape.QuantumTape() as tape:
            qp.iterative_qpe(qp.RZ(1.0, wires=[0]), wire, iters=3)

        assert wire in tape.wires

    @pytest.mark.parametrize("phi", (1.2, 2.3, 3.4))
    def test_measurement_processes_probs(self, phi):
        """Test to check that the measurement process prob works correctly"""

        dev = qp.device("default.qubit")

        @qp.qnode(dev)
        def circuit_qpe():
            # Initial state
            qp.PauliX(wires=[0])

            # Iterative QPE
            qp.QuantumPhaseEstimation(qp.RZ(phi, wires=[0]), estimation_wires=[1, 2, 3])

            return [qp.probs(wires=i) for i in [1, 2, 3]]

        @qp.qnode(dev)
        def circuit_iterative():
            # Initial state
            qp.PauliX(wires=[0])

            # Iterative QPE
            measurements = qp.iterative_qpe(qp.RZ(phi, wires=[0]), aux_wire=[1], iters=3)

            return [qp.probs(op=i) for i in measurements]

        assert np.allclose(circuit_qpe(), circuit_iterative())

    @pytest.mark.parametrize("phi", (1.2, 2.3, 3.4))
    def test_measurement_processes_expval(self, phi):
        """Test to check that the measurement process expval works correctly"""

        dev = qp.device("default.qubit")

        @qp.qnode(dev)
        def circuit_qpe():
            # Initial state
            qp.PauliX(wires=[0])

            # Iterative QPE
            qp.QuantumPhaseEstimation(qp.RZ(phi, wires=[0]), estimation_wires=[1, 2, 3])

            # We will use the projector as an observable
            return [qp.expval(qp.Hermitian([[0, 0], [0, 1]], wires=i)) for i in [1, 2, 3]]

        @qp.qnode(dev)
        def circuit_iterative():
            # Initial state
            qp.PauliX(wires=[0])

            # Iterative QPE
            measurements = qp.iterative_qpe(qp.RZ(phi, wires=[0]), aux_wire=[1], iters=3)

            return [qp.expval(op=i) for i in measurements]

        assert np.allclose(circuit_qpe(), circuit_iterative())


@pytest.mark.slow
@pytest.mark.capture
def test_capture_execution(seed):
    """Test that iterative qpe can be captured and executed.

    While this is a rather bad test:
    * the captured jaxpr has too many classical instructions for
    easy verification of its contents
    * The captured jaxpr cannot be used with CollectOpsandMeas as it converts mcm integers to
    measurement values, which are incompatible with the scatter operation used in
    `measurements = measurements.at[iters - i - 1].set(m)`
    * Evaluating jaxpr currently uses single-branch-statistics, which gives incorrect results for a
    a single execution.


    """
    import jax

    def f(x):
        qp.X(0)
        return qp.iterative_qpe(qp.RZ(x, wires=[0]), aux_wire=1, iters=3)

    x = jax.numpy.array(2.0)

    jaxpr = jax.make_jaxpr(f)(1.5)

    dev = qp.device("default.qubit", wires=5, seed=seed)

    # hack for single-branch statistics
    samples = qp.math.vstack([dev.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, x) for _ in range(5000)])
    probs_capture = qp.probs(wires=(0, 1, 2)).process_samples(
        samples, wire_order=qp.wires.Wires((0, 1, 2))
    )

    qp.capture.disable()

    @qp.qnode(dev)
    def normal_qnode(x):
        meas = f(x)
        return qp.probs(op=meas)

    probs_normal = normal_qnode(x)

    assert qp.math.allclose(probs_capture, probs_normal, atol=0.02)
