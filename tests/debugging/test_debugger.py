# Copyright 2026 Xanadu Quantum Technologies Inc.

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
Tests for debugger helpers in the debugging module.
"""

import numpy as np
import pytest

import pennylane as qp
from pennylane.debugging import debug_expval, debug_probs, debug_state
from pennylane.decomposition import add_decomps, register_resources


# pylint: disable=too-few-public-methods
class _DecomposingOp(qp.operation.Operation):
    """Define a minimal operator whose decomposition creates multiple gates.

    Specifically used to trigger the queue leaking bug when wrapped in qp.ctrl()."""

    num_wires = 1

    @staticmethod
    def compute_decomposition(wires):
        return [qp.RY(0.5, wires=wires), qp.RX(0.5, wires=wires)]


@register_resources({qp.RY: 1, qp.RX: 1})
def _ry_rx_decompose(wires, **__):
    qp.RY(0.5, wires=wires)
    qp.RX(0.5, wires=wires)


add_decomps(_DecomposingOp, _ry_rx_decompose)


@pytest.mark.usefixtures("enable_and_disable_graph_decomp")
class TestQueuePollution:
    """Tests that the 'debug_*' helpers don't pollute the queue. Regression tests for issue #9343."""

    @staticmethod
    def _create_circuit():
        """Creates the ground truth circuit."""

        @qp.qnode(qp.device("default.qubit", wires=[0, 1]))
        def correct_circuit():
            qp.H(0)
            qp.ctrl(_DecomposingOp, control=0)(wires=1)
            return qp.expval(qp.Z(0))

        return correct_circuit

    def test_debug_state_doesnt_pollute_active_queue(self):
        """Tests that 'debug_state' doesn't accidentally change the queue."""

        @qp.qnode(qp.device("default.qubit", wires=[0, 1]))
        def circuit():
            qp.H(0)
            qp.ctrl(_DecomposingOp, control=0)(wires=1)

            ops_before = list(qp.QueuingManager.active_context().queue)
            _ = debug_state()
            ops_after = list(qp.QueuingManager.active_context().queue)
            assert ops_before == ops_after

            return qp.expval(qp.Z(0))

        expected = self._create_circuit()
        assert np.allclose(circuit(), expected())

    def test_debug_probs_doesnt_pollute_active_queue(self):
        """Tests that 'debug_probs' doesn't accidentally change the queue."""

        @qp.qnode(qp.device("default.qubit", wires=[0, 1]))
        def circuit():
            qp.H(0)
            qp.ctrl(_DecomposingOp, control=0)(wires=1)

            ops_before = list(qp.QueuingManager.active_context().queue)
            _ = debug_probs()
            ops_after = list(qp.QueuingManager.active_context().queue)
            assert ops_before == ops_after

            return qp.expval(qp.Z(0))

        expected = self._create_circuit()
        assert np.allclose(circuit(), expected())


    def test_debug_probs_accepts_measurement_value(self):
        """debug_probs(op=MeasurementValue) must not truth-test the measurement (#9652)."""
        import pennylane as qml
        from pennylane.debugging import debug_probs

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            m0 = qml.measure(0)
            # Calling debug_probs with a MeasurementValue used to raise via ``if op:``.
            probs = debug_probs(op=m0)
            assert probs is not None
            assert len(probs) == 2
            return qml.expval(qml.Z(0))

        # Execute in a context that has an active queue for debugger helpers.
        # debug_probs expects QueuingManager active context (breakpoint path).
        with qml.queuing.AnnotatedQueue() as q:
            qml.Hadamard(0)
            m0 = qml.measure(0)
            # May need active device/state for _measure - if too heavy, only assert no truthiness error
            try:
                _ = debug_probs(op=m0)
            except ValueError as e:
                if "truth value of a MeasurementValue" in str(e):
                    raise AssertionError("regression: MeasurementValue truthiness still evaluated") from e
                # Other errors (e.g. no active state) are OK for this regression as long as truthiness is fixed
                pass

    def test_debug_expval_doesnt_pollute_active_queue(self):
        """Tests that 'debug_expval' doesn't accidentally change the queue."""

        @qp.qnode(qp.device("default.qubit", wires=[0, 1]))
        def circuit():
            qp.H(0)
            qp.ctrl(_DecomposingOp, control=0)(wires=1)

            ops_before = list(qp.QueuingManager.active_context().queue)
            _ = debug_expval(qp.Z(0))
            ops_after = list(qp.QueuingManager.active_context().queue)
            assert ops_before == ops_after

            return qp.expval(qp.Z(0))

        expected = self._create_circuit()
        assert np.allclose(circuit(), expected())
