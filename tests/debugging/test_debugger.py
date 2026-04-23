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
Unit tests for debugger helpers in the debugging module.
"""

import pennylane as qp
from pennylane.debugging import debug_expval, debug_probs, debug_state


class _DecomposingOp(qp.operation.Operation):
    """Define a minimal operator whose decomposition creates multiple gates.

    Specifically used to trigger the queue leaking bug when wrapped in qp.ctrl()."""

    num_wires = 1

    @staticmethod
    def compute_decomposition(wires):
        return [qp.RY(0.5, wires=wires), qp.RX(0.5, wires=wires)]


class TestQueuePollution:
    """Tests that the 'debug_*' helpers don't pollute the queue. Regression tests for issue #9343."""

    def test_debug_state_doesnt_pollute_active_queue(self):
        """Tests that 'debug_state' doesn't accidentally change the queue."""

        @qp.qnode(qp.device("default.qubit", wires=[0, 1]))
        def circuit():
            qp.H(0)
            qp.ctrl(_DecomposingOp, control=0)(wires=1)

            ops_before = len(qp.QueuingManager.active_context().queue)
            _ = debug_state()
            ops_after = len(qp.QueuingManager.active_context().queue)
            assert ops_before == ops_after

            return qp.expval(qp.Z(0))

        _ = circuit()

    def test_debug_probs_doesnt_pollute_active_queue(self):
        """Tests that 'debug_probs' doesn't accidentally change the queue."""

        @qp.qnode(qp.device("default.qubit", wires=[0, 1]))
        def circuit():
            qp.H(0)
            qp.ctrl(_DecomposingOp, control=0)(wires=1)

            ops_before = len(qp.QueuingManager.active_context().queue)
            _ = debug_probs()
            ops_after = len(qp.QueuingManager.active_context().queue)
            assert ops_before == ops_after

            return qp.expval(qp.Z(0))

        _ = circuit()

    def test_debug_expval_doesnt_pollute_active_queue(self):
        """Tests that 'debug_expval' doesn't accidentally change the queue."""

        @qp.qnode(qp.device("default.qubit", wires=[0, 1]))
        def circuit():
            qp.H(0)
            qp.ctrl(_DecomposingOp, control=0)(wires=1)

            ops_before = len(qp.QueuingManager.active_context().queue)
            _ = debug_expval(qp.Z(0))
            ops_after = len(qp.QueuingManager.active_context().queue)
            assert ops_before == ops_after

            return qp.expval(qp.Z(0))

        _ = circuit()
