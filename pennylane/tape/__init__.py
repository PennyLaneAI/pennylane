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
from .operation import mock_operations
from .measure import expval, var, sample, state, probs, MeasurementProcess
from .queuing import AnnotatedQueue, Queue, QueuingContext
from .tapes import QuantumTape, QubitParamShiftTape, CVParamShiftTape, ReversibleTape
from .qnode import QNode, qnode


_mock_stack = []


def enable_tape():
    """Enable tape mode"""
    if _mock_stack:
        return

    mocks = [mock.patch("pennylane.qnode", qnode), mock.patch("pennylane.QNode", QNode)]

    with contextlib.ExitStack() as stack:
        for m in mocks:
            stack.enter_context(m)

        _mock_stack.append(stack.pop_all())


def disable_tape():
    """disable tape mode"""
    if not _mock_stack:
        raise ValueError("Tape mode is not currently enabled.")

    _mock_stack.pop().close()
