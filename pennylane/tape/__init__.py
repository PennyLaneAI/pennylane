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
This subpackage contains various quantum tapes, which track, queue,
validate, execute, and differentiate quantum circuits.
"""
import contextlib
from unittest import mock

from .circuit_graph import TapeCircuitGraph
from .measure import expval, var, sample, state, probs, MeasurementProcess
from .operation import mock_operations
from .queuing import AnnotatedQueue, Queue, QueuingContext
from .qnode import QNode, qnode
from .tapes import QuantumTape, QubitParamShiftTape, CVParamShiftTape, ReversibleTape


_mock_stack = None


def enable_tape():
    global _mock_stack

    if _mock_stack is not None:
        return

    mocks = [mock.patch("pennylane.qnode", qnode), mock.patch("pennylane.QNode", QNode)]

    with contextlib.ExitStack() as stack:
        for m in mocks:
            stack.enter_context(m)

        _mock_stack = stack.pop_all()


def disable_tape():
    if _mock_stack is None:
        raise ValueError("Tape mode is not currently enabled.")

    _mock_stack.close()
