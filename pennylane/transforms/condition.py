# Copyright 2022 Xanadu Quantum Technologies Inc.

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
Contains the condition transform.
"""
from copy import copy
from functools import wraps
from typing import Type

from pennylane.measurements import MeasurementValue
from pennylane.operation import AnyWires, Operation
from pennylane.transforms import make_tape


class ConditionalTransformError(ValueError):
    """Error for using qml.cond incorrectly"""


class Conditional(Operation):
    """A Conditional Operation.

    Unless you are a Pennylane plugin developer, **you should NOT directly use this class**,
    instead, use the :func:`qml.cond <.cond>` function.

    The ``Conditional`` class is a container class that defines an operation
    that should by applied relative to a single measurement value.

    Support for executing ``Conditional`` operations is device-dependent. If a
    device doesn't support mid-circuit measurements natively, then the QNode
    will apply the :func:`defer_measurements` transform.

    Args:
        expr (MeasurementValue): the measurement outcome value to consider
        then_op (Operation): the PennyLane operation to apply conditionally
        do_queue (bool): indicates whether the operator should be
            recorded when created in a tape context
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified
    """

    num_wires = AnyWires

    def __init__(
        self,
        expr: MeasurementValue[bool],
        then_op: Type[Operation],
        do_queue=True,
        id=None,
    ):
        self.meas_val = expr
        self.then_op = then_op
        super().__init__(wires=then_op.wires, do_queue=do_queue, id=id)


def cond(condition, true_fn, false_fn=None):
    """Condition a quantum operation on the results of mid-circuit qubit measurements.

    Support for using :func:`~.cond` is device-dependent. If a device doesn't
    support mid-circuit measurements natively, then the QNode will apply the
    :func:`defer_measurements` transform.

    Args:
        condition (.MeasurementValue[bool]): a conditional expression involving a mid-circuit
           measurement value (see :func:`.pennylane.measure`)
        true_fn (callable): The quantum function of PennyLane operation to
            apply if ``condition`` is ``True``
        false_fn (callable): The quantum function of PennyLane operation to
            apply if ``condition`` is ``False``

    Returns:
        function: A new function that applies the conditional equivalent of ``true_fn``. The returned
        function takes the same input arguments as ``true_fn``.

    **Example**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=3)

        first_par = 0.1
        sec_par = 0.3

        @qml.qnode(dev)
        def qnode():
            m_0 = qml.measure(0)
            qml.cond(m_0, qml.RY)(first_par, wires=1)

            m_1 = qml.measure(2)
            qml.cond(m_0, qml.RZ)(sec_par, wires=1)
            return qml.expval(qml.PauliZ(1))
    """
    if callable(true_fn):
        # We assume that the callable is an operation or a quantum function

        with_meas_err = (
            "Only quantum functions that contain no measurements can be applied conditionally."
        )

        @wraps(true_fn)
        def wrapper(*args, **kwargs):
            # We assume that the callable is a quantum function

            # 1. Apply true_fn conditionally
            tape = make_tape(true_fn)(*args, **kwargs)

            if tape.measurements:
                raise ConditionalTransformError(with_meas_err)

            for op in tape.operations:
                Conditional(condition, op)

            if false_fn is not None:
                # 2. Apply false_fn conditionally
                else_tape = make_tape(false_fn)(*args, **kwargs)

                if else_tape.measurements:
                    raise ConditionalTransformError(with_meas_err)

                inverted_m = copy(condition)
                inverted_m = ~inverted_m

                for op in else_tape.operations:
                    Conditional(inverted_m, op)

    else:
        raise ConditionalTransformError(
            "Only operations and quantum functions with no measurements can be applied conditionally."
        )

    return wrapper
