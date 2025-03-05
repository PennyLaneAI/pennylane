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
from functools import wraps
from typing import Callable

import pennylane as qml
from pennylane.compiler import compiler
from pennylane.measurements import MeasurementValue, MidMeasureMP
from pennylane.ops.op_math.condition import CondCallable, Conditional


def cond_meas(
    condition,
    true_fn: Callable,
    false_fn: Callable,
):
    """Quantum-compatible if-else conditional --- condition mid-circuit qubit measurements
    on parameters such as the results of mid-circuit qubit measurements.

    .. note::

        With the Python interpreter, support for :func:`~.cond`
        is device-dependent. If a device doesn't
        support mid-circuit measurements natively, then the QNode will
        apply the :func:`defer_measurements` transform.

    .. note::

        This function is currently not compatible with :func:`~.qjit`, or with
        :func:`.pennylane.capture.enabled`.

    Args:
        condition (Union[.MeasurementValue, bool]): a conditional expression that may involve a mid-circuit
           measurement value (see :func:`.pennylane.measure`).
        true_fn (callable): The quantum function or PennyLane operation to
            apply if ``condition`` is ``True``
        false_fn (callable): The quantum function or PennyLane operation to
            apply if ``condition`` is ``False``

    Returns:
        function: A new function that applies the conditional equivalent of ``true_fn``. The returned
        function takes the same input arguments as ``true_fn``.

    **Example**

    .. code-block:: python3

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def qnode(x, y):
            qml.Hadamard(0)
            m_0 = qml.measure(0)
            qml.cond(m_0, qml.RY)(x, wires=1)

            qml.Hadamard(2)
            qml.RY(-np.pi/2, wires=[2])
            m_1 = qml.measure(2)
            qml.cond(m_1 == 0, qml.RX)(y, wires=1)
            return qml.expval(qml.Z(1))

    .. code-block :: pycon

        >>> first_par = np.array(0.3)
        >>> sec_par = np.array(1.23)
        >>> qnode(first_par, sec_par)
        tensor(0.32677361, requires_grad=True)

    .. note::

        If the first argument of ``cond_meas`` is a measurement value (e.g., ``m_0``
        in ``qml.cond(m_0, qml.RY)``), then ``m_0 == 1`` is considered
        internally.

    .. warning::

        Expressions with boolean logic flow using operators like ``and``,
        ``or`` and ``not`` are not supported as the ``condition`` argument.

        While such statements may not result in errors, they may result in
        incorrect behaviour.
    """

    if compiler.active_compiler() or qml.capture.enabled():
        raise NotImplementedError("The `cond_meas` function is not compatible with `qjit`")

    if qml.capture.enabled():
        raise NotImplementedError("The `cond_meas` function is not program capture")

    if not isinstance(condition, MeasurementValue):
        # The condition is not a mid-circuit measurement - we can simplify immediately
        return CondCallable(condition, true_fn, false_fn)

    if callable(true_fn):

        @wraps(true_fn)
        def wrapper(*args, **kwargs):
            # We assume that the callable is a measure function

            with qml.QueuingManager.stop_recording():
                true_meas = true_fn(*args, **kwargs).measurements[0]
                false_meas = false_fn(*args, **kwargs).measurements[0]

            _validate_measurements(true_meas, false_meas)

            for op in true_meas.diagonalizing_gates():
                Conditional(condition, op)

            for op in false_meas.diagonalizing_gates():
                Conditional(~condition, op)

            mp = MidMeasureMP(
                true_meas.wires, reset=true_meas.reset, postselect=true_meas.postselect
            )

            return MeasurementValue([mp], processing_fn=lambda v: v)

    else:
        raise ValueError("Only measurement functions can be applied conditionally by `cond_meas`.")

    return wrapper


def _validate_measurements(true_meas, false_meas):
    """Takes a pair of MCMs (representing a true and false functions for the conditional) and
    confirms that they have the expected type ,and 'match' except for the measurement basis"""
    print(true_meas, false_meas)

    if not (isinstance(true_meas, MidMeasureMP) and isinstance(false_meas, MidMeasureMP)):
        raise ValueError(
            "Only measurement functions that create a mid-circuit measurement"
            " and return a measurement value can be used in `cond_meas`"
        )

    if not (
        true_meas.wires == false_meas.wires
        and true_meas.reset == false_meas.reset
        and true_meas.postselect == false_meas.postselect
    ):
        raise ValueError(
            "When applying a mid-circuit measurement in `cond_meas`, the `wire`, "
            "`postselect` and `reset` behaviour must be consistent for both "
            "branches of the conditional. Only the basis of the measurement (defined "
            "by measurement type or by `plane` and `angle`) can vary."
        )
