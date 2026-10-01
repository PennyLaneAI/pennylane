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

import itertools
from functools import partial

import jax
import numpy as np
import pytest

import pennylane as qp
from pennylane.capture.base_interpreter import FlattenedInterpreter
from pennylane.capture.primitives import measure_prim
from pennylane.ops.mid_measure import MidMeasure


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

    # NOTE: This is an alternative to using 'plxpr_to_tape' which does not work for this specific inner for loop.
    @pytest.mark.parametrize("iters", (1, 2, 3, 4))
    def test_subroutine_body_structure(self, iters):
        """Test the structure of the captured body: each round is a Hadamard, a controlled power
        of the base, one ``for_loop`` of conditional phase corrections, a Hadamard and a
        measurement with reset, all acting on the auxiliary wire.

        NOTE: Drafted with the help of Jukebox

        """

        jaxpr = jax.make_jaxpr(
            lambda phi: qp.iterative_qpe(qp.RZ(phi, wires=[0]), aux_wire=1, iters=iters)
        )(2.0)
        body = jaxpr.eqns[0].params["jaxpr"].jaxpr
        aux = body.invars[-1]

        # ignore the array bookkeeping ('broadcast_in_dim', 'concatenate') of the outcomes
        eqns = [e for e in body.eqns if e.primitive.name in ("operator", "for_loop", "measure")]
        meas_outvars = []
        for i in range(iters):
            expected = ["Hadamard", "Pow2", "for_loop", "Hadamard", "measure"]

            # no for loop if there are no measurements
            if i == 0:
                expected.remove("for_loop")

            rnd, eqns = eqns[: len(expected)], eqns[len(expected) :]
            names = [
                e.params["op_cls"].__name__ if e.primitive.name == "operator" else e.primitive.name
                for e in rnd
            ]
            assert names == expected

            _, ctrl_pow, *_, meas = rnd
            assert ctrl_pow.params["n_ctrls"] == 1
            assert ctrl_pow.params["z"][0] == (2 ** (iters - i - 1),)
            assert ctrl_pow.invars[-2] is aux  # control wire

            assert meas.params["reset"] is True
            assert all(e.invars[-1] is aux for e in (rnd[0], rnd[-2], meas))

            if i > 0:
                loop = rnd[2]
                start, stop, step, *_ = loop.invars
                assert (start.val, stop.val, step.val) == (0, i, 1)
                # previous outcomes and the aux wire are the only loop arguments
                assert loop.invars[-1] is aux

                # the loop body only does classical indexing plus a single conditional
                body_eqns = loop.params["jaxpr_body_fn"].eqns
                quantum = [
                    e for e in body_eqns if e.primitive.name in ("operator", "measure", "cond")
                ]
                (cond_eqn,) = quantum
                assert cond_eqn.primitive.name == "cond"
                true_branch, false_branch = cond_eqn.params["jaxpr_branches"]
                assert not false_branch.eqns
                (phase,) = (
                    e for e in true_branch.eqns if e.primitive.name in ("operator", "measure")
                )
                assert phase.params["op_cls"] is qp.PhaseShift
                # PhaseShift acts on the aux wire threaded through the loop and the cond
                loop_body = loop.params["jaxpr_body_fn"]
                assert cond_eqn.invars[-1] is loop_body.constvars[-1]
                assert phase.invars[-1] is true_branch.constvars[-1]

            meas_outvars.append(meas.outvars[0])

        assert not eqns
        # the most recent outcome is returned first
        assert body.outvars == meas_outvars[::-1]

    @pytest.mark.parametrize("iters", (1, 2, 3, 4))
    def test_subroutine_body_matches_uncaptured(self, iters):
        """Test that, for every combination of mid-circuit measurement outcomes, the captured
        program applies exactly the same gates as the uncaptured circuit.

        The uncaptured circuit is checked numerically against ``qp.QuantumPhaseEstimation`` in
        ``TestIQPE``, so an exact match here makes the captured circuit correct too.

        NOTE: Drafted with the help of Jukebox
        """
        phi, wire, aux_wire = 0.7, 3, 5

        # ---------------------------------------------------------
        # PHASE 1: GENERATE BOTH REPRESENTATIONS
        # ---------------------------------------------------------

        jaxpr = jax.make_jaxpr(
            lambda p: qp.iterative_qpe(qp.RZ(p, wires=[wire]), aux_wire=aux_wire, iters=iters)
        )(phi)

        qp.capture.disable()
        try:
            with qp.queuing.AnnotatedQueue() as q:
                mcm_values = qp.iterative_qpe(
                    qp.RZ(phi, wires=[wire]), aux_wire=aux_wire, iters=iters
                )
            uncaptured = qp.tape.QuantumScript.from_queue(q).operations
        finally:
            qp.capture.enable()

        # 'iterative_qpe' returns the most recent outcome first (it uses insert(0, m)).
        # We need to map each symbolic MeasurementValue back to its chronological iteration
        # step (0 to iters-1) so we can feed it the correct forced bit later.
        rounds = {mv.measurements[0]: i for i, mv in enumerate(reversed(mcm_values))}

        # ---------------------------------------------------------
        # PHASE 2: DEFINE THE JAXPR INTERPRETER
        # ---------------------------------------------------------

        class _FixedOutcomes(FlattenedInterpreter):
            """Evaluates a plxpr with each mid-circuit measurement returning a fixed outcome, and records
            the gates applied and the measurements made."""

            def __init__(self, outcomes):
                super().__init__()
                # 'outcomes' is a predefined sequence of 1s and 0s (e.g., (1, 0, 1))
                self.outcomes = iter(outcomes)
                self.ops = []  # List to store the flat sequence of gates actually executed

            def interpret_operation(self, op):
                # For standard quantum gates (Hadamard, PhaseShift), just record them.
                self.ops.append(op)
                return op

        # Intercept JAX primitive measurement nodes. Instead of doing quantum math,
        # record the measurement intent as a tuple, and return the next hardcoded bit.
        @_FixedOutcomes.register_primitive(measure_prim)
        def _fixed_measure(self, wires, reset, postselect):
            self.ops.append(("measure", qp.wires.Wires(int(wires)), reset, postselect))
            return jax.numpy.int32(next(self.outcomes))

        # ---------------------------------------------------------
        # PHASE 3: SIMULATE EVERY POSSIBLE MEASUREMENT PATH
        # ---------------------------------------------------------

        # itertools.product((0,1), repeat=iters) generates all possible bitstrings.
        # e.g., for iters=2: (0,0), (0,1), (1,0), (1,1)
        for outcomes in itertools.product((0, 1), repeat=iters):

            # 1. EVALUATE CAPTURED CIRCUIT
            captured = _FixedOutcomes(outcomes)
            # This walks the JAXpr graph. When it hits a 'cond', it uses the forced bits
            # returned by _fixed_measure to decide which branch to take.
            returned = captured.eval(jaxpr.jaxpr, jaxpr.consts, phi)

            # Verify the function returns the measurements in reverse-chronological order
            assert [int(r) for r in returned] == list(outcomes[::-1])

            # 2. EVALUATE UNCAPTURED CIRCUIT
            expected = []
            for op in uncaptured:
                if isinstance(op, MidMeasure):
                    # Record measurements as tuples to match the captured formatting
                    expected.append(("measure", op.wires, op.reset, op.postselect))
                elif isinstance(op, qp.ops.Conditional):
                    # For uncaptured conditionals, classically evaluate the boolean condition
                    # using our mocked 'outcomes' bits.
                    bits = [outcomes[rounds[m]] for m in op.meas_val.measurements]
                    # If the condition evaluates to True, append the underlying gate (e.g. PhaseShift)
                    if op.meas_val.processing_fn(*bits):
                        expected.append(op.base)
                else:
                    # Append standard unconditional gates (Hadamards, C-Unitary)
                    expected.append(op)

            # 3. ASSERT EXACT MATCH
            # Ensure both systems decided to apply the exact same number and sequence of gates
            # under this specific measurement path.
            assert len(captured.ops) == len(expected)
            for actual, op in zip(captured.ops, expected, strict=True):
                if isinstance(op, tuple):
                    assert actual == op
                else:
                    qp.assert_equal(actual, op, check_interface=False, check_trainability=False)

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
    @pytest.mark.parametrize("num_iters", (3, 6))
    def test_qjit_integration(self, num_iters):
        """Test that this subroutine can be used with QJIT."""

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
            "PhaseShift": num_iters * (num_iters - 1) // 2,
            "PauliX": num_iters,
        }
        assert specs.resources["Before MLIR Passes"].quantum_operations == expected_operations
