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
import warnings

from .circuit_graph import TapeCircuitGraph
from .queuing import AnnotatedQueue, Queue, QueuingContext
from .measure import MeasurementProcess, state
from .qnode import QNode, qnode
from .tapes import (
    QuantumTape,
    JacobianTape,
    QubitParamShiftTape,
    CVParamShiftTape,
    ReversibleTape,
)


_mock_stack = []


def enable_tape():
    """Enable tape mode.

    Tape mode is an experimental new mode of PennyLane. QNodes created in tape mode have support for
    in-QNode classical processing, differentiable quantum decompositions, returning the quantum
    state, less restrictive QNode signatures, and various other improvements.

    For more details on tape mode, see :mod:`pennylane.tape`.

    **Example**

    Simply call this function at the beginning of your script or session.

    >>> qml.enable_tape()

    All subsequent QNodes will be created using tape mode, and can take
    advantage of the various tape mode features:

    >>> dev = qml.device("default.qubit", wires=1)
    >>> @qml.qnode(dev)
    ... def circuit(x, y):
    ...     qml.RX(np.sin(x) * y, wires=0)
    ...     return qml.expval(qml.PauliZ(0))
    >>> print(circuit(0.5, 0.1))
    0.9988509758748578
    >>> qml.grad(circuit)(0.5, 0.1)
    (array(-0.00420574), array(-0.02297608)))
    >>> type(circuit)
    pennylane.tape.qnode.QNode

    Tape mode can be disabled by calling :func:`~.disable_tape`.
    """
    if _mock_stack:
        return

    mocks = [mock.patch("pennylane.qnode", qnode), mock.patch("pennylane.QNode", QNode)]

    with contextlib.ExitStack() as stack:
        for m in mocks:
            stack.enter_context(m)

        _mock_stack.append(stack.pop_all())


def disable_tape():
    """Disable tape mode.

    This function may be called at any time after :func:`~.enable_tape` has been executed
    in order to disable tape mode.

    Tape mode is an experimental new mode of PennyLane. QNodes created in tape mode have support for
    in-QNode classical processing, differentiable quantum decompositions, returning the quantum
    state, less restrictive QNode signatures, and various other improvements.

    For more details on tape mode, see :mod:`~.tape`.
    """
    if not _mock_stack:
        warnings.warn("Tape mode is not currently enabled.", UserWarning)
    else:
        _mock_stack.pop().close()
