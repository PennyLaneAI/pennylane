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
import inspect
from unittest import mock
import warnings

import pennylane as qml

from . import measure
from . import transforms
from .circuit_graph import TapeCircuitGraph
from .queuing import AnnotatedQueue, Queue, QueuingContext
from .measure import MeasurementProcess, state, density_matrix
from .qnode import QNode, qnode, draw
from .tapes import (
    QuantumTape,
    JacobianTape,
    QubitParamShiftTape,
    CVParamShiftTape,
    ReversibleTape,
)


_mock_stack = []


class TapeOperationRecorder(QuantumTape):
    """A template and quantum function inspector,
    allowing easy introspection of operators that have been
    applied without requiring a QNode.

    **Example**:

    The OperationRecorder is a context manager. Executing templates
    or quantum functions stores resulting applied operators in the
    recorder, which can then be printed.

    >>> weights = qml.init.strong_ent_layers_normal(n_layers=1, n_wires=2)
    >>>
    >>> with OperationRecorder() as rec:
    >>>    qml.templates.layers.StronglyEntanglingLayers(*weights, wires=[0, 1])
    >>>
    >>> print(rec)
    Operations
    ==========
    Rot(-0.10832656163640327, 0.14429091013664083, -0.010835826725765343, wires=[0])
    Rot(-0.11254523669444501, 0.0947222564914006, -0.09139600968423377, wires=[1])
    CNOT(wires=[0, 1])
    CNOT(wires=[1, 0])

    Alternatively, the :attr:`~.OperationRecorder.queue` attribute can be used
    to directly access the applied :class:`~.Operation` and :class:`~.Observable`
    objects.

    Attributes:
        queue (List[Operator]): list of operators applied within
            the OperatorRecorder context, includes operations and observables
        operations (List[Operation]): list of operations applied within
            the OperatorRecorder context
        observables (List[Observable]): list of observables applied within
            the OperatorRecorder context
    """

    def __init__(self):
        super().__init__()

    def _process_queue(self):
        super()._process_queue()

        for obj, info in self._queue.items():
            QueuingContext.append(obj, **info)

        QueuingContext.remove(self)

    def __str__(self):
        output = ""
        output += "Operations\n"
        output += "==========\n"
        for op in self.operations:
            output += repr(op) + "\n"

        output += "\n"
        output += "Observables\n"
        output += "==========\n"
        for op in self.observables:
            output += repr(op) + "\n"

        return output

    @property
    def queue(self):
        return self.operations + self.observables


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

    mocks = [
        mock.patch("pennylane.qnode", qnode),
        mock.patch("pennylane.QNode", QNode),
        mock.patch("pennylane.expval", measure.expval),
        mock.patch("pennylane.var", measure.var),
        mock.patch("pennylane.probs", measure.probs),
        mock.patch("pennylane.sample", measure.sample),
        mock.patch("pennylane._queuing.OperationRecorder", TapeOperationRecorder),
    ]

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


def tape_mode_active():
    """Returns whether tape mode is enabled."""
    return inspect.isclass(qml.QNode) and issubclass(qml.QNode, qml.tape.QNode)
