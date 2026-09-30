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

from functools import partial

import jax
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


@pytest.mark.capture
class TestCaptureIQPE:
    """Tests the capture of the function as a subroutine in jaxpr."""

    def test_capture_as_single_subroutine_eqn(self, recwarn):
        """Test that the rounds are captured into one subroutine."""

        def circuit(phi, iters):
            return qp.iterative_qpe(qp.RZ(phi, wires=[0]), aux_wire=1, iters=iters)

        for iters in (3, 6):
            fixed_circuit = partial(circuit, iters=iters)
            cjaxpr = jax.make_jaxpr(fixed_circuit)(2)

            assert len(cjaxpr.eqns) == 1
            assert cjaxpr.eqns[0].primitive == qp.capture.primitives.quantum_subroutine_prim
            assert len(cjaxpr.jaxpr.outvars) == iters

        assert not [w for w in recwarn if issubclass(w.category, qp.exceptions.CaptureWarning)]

    @pytest.mark.parametrize("iters", (2, 3, 4))
    def test_subroutine_body_matches_uncaptured(self, iters):
        """Test that the captured body matches the legacy tape implementation."""

        # CAPTURE
        jaxpr = jax.make_jaxpr(
            lambda phi: qp.iterative_qpe(qp.RZ(phi, wires=[0]), aux_wire=1, iters=iters)
        )(2.0)
        cjaxpr = jaxpr.eqns[0].params["jaxpr"]
        captured = qp.tape.plxpr_to_tape(cjaxpr.jaxpr, cjaxpr.consts, 2.0, 0, 1)

        # LEGACY TAPE
        qp.capture.disable()
        with qp.queuing.AnnotatedQueue() as q:
            qp.iterative_qpe(qp.RZ(2.0, wires=[0]), aux_wire=1, iters=iters)
        expected = qp.tape.QuantumScript.from_queue(q)
        qp.capture.enable()

        assert [type(op) for op in captured.operations] == [type(op) for op in expected.operations]

    def test_subroutine_is_shared_if_different_dyn_args(self):
        """Test that two calls share one subroutine body."""

        def circuit(phi):
            qp.iterative_qpe(qp.RZ(phi, wires=[0]), aux_wire=1, iters=3)
            qp.iterative_qpe(qp.RZ(phi, wires=[2]), aux_wire=3, iters=3)

        eqns = jax.make_jaxpr(circuit)(2.0).eqns

        assert len(eqns) == 2
        # Shared impl
        assert eqns[0].params["jaxpr"] is eqns[1].params["jaxpr"]

    def test_subroutine_is_not_shared_if_different_static_args(self):
        """Test that two calls with diff iters do not share one subroutine body."""

        def circuit(phi):
            qp.iterative_qpe(qp.RZ(phi, wires=[0]), aux_wire=3, iters=1)
            qp.iterative_qpe(qp.RZ(phi, wires=[2]), aux_wire=3, iters=3)

        eqns = jax.make_jaxpr(circuit)(2.0).eqns

        assert len(eqns) == 2
        # Shared impl
        assert eqns[0].params["jaxpr"] is not eqns[1].params["jaxpr"]

    @pytest.mark.parametrize("aux_wire", (1, [1], qp.wires.Wires([1])))
    def test_different_aux_wire_containers(self, aux_wire):
        """Test different wire inputs can be used."""

        jaxpr = jax.make_jaxpr(
            lambda phi: qp.iterative_qpe(qp.RZ(phi, wires=[0]), aux_wire=aux_wire, iters=3)
        )(2.0)

        assert len(jaxpr.eqns) == 1
        assert jaxpr.eqns[0].primitive == qp.capture.primitives.quantum_subroutine_prim

    def test_legacy_op_can_be_used_as_base(self):
        """Ensure a legacy operator still works fine under capture."""

        class DummyOp(qp.core.Operator):  # pylint: disable=too-few-public-methods
            pass

        def circuit(phi):
            return qp.iterative_qpe(DummyOp(phi, 0), 1, 3)

        cjaxpr = jax.make_jaxpr(circuit)(0.5)

        assert cjaxpr.eqns[-1].primitive == qp.capture.primitives.quantum_subroutine_prim
        # op is captured as data into subroutine
        assert cjaxpr.eqns[-2].outvars[0] in cjaxpr.eqns[-1].invars

    @pytest.mark.catalyst
    def test_qjit_integration(self):
        """Test that this subroutine can be used with QJIT."""
        num_iters = 3

        @qp.qjit(capture=True, target="mlir", collect_decomp_rules=False)
        @qp.set_shots(10)
        @qp.qnode(qp.device("null.qubit", wires=2))
        def c():
            return qp.sample(qp.iterative_qpe(qp.RX(0.5, 0), 1, num_iters))

        specs = qp.specs(c, level="all-mlir")()
        # NOTE: PauliX comes from the aux_wire reset
        expected_operations = {
            "C(Pow2)": num_iters,
            "Hadamard": 2 * num_iters,
            "MidCircuitMeasure": num_iters,
            "PhaseShift": num_iters,
            "PauliX": num_iters,
        }
        assert specs.resources["Before MLIR Passes"].quantum_operations == expected_operations
