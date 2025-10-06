# Copyright 2025 Xanadu Quantum Technologies Inc.

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
from collections.abc import Callable

from pennylane import capture
from pennylane.capture.autograph import wraps
from pennylane.measurements import MeasurementValue, MidMeasureMP
from pennylane.ops.op_math.condition import CondCallable, Conditional, cond
from pennylane.queuing import QueuingManager


def cond_measure(
    condition: MeasurementValue | bool,
    true_fn: Callable,
    false_fn: Callable,
):
    """Perform a mid-circuit measurement where the basis of the measurement is conditional on the
    supplied expression. This conditional expression may involve the results of other mid-circuit
    qubit measurements.

    Args:
        condition (Union[.MeasurementValue, bool]): a conditional expression that may involve a mid-circuit
           measurement value (see :func:`.pennylane.measure`).
        true_fn (callable): The quantum function or PennyLane operation to
            apply if ``condition`` is ``True``. The callable must create a single mid-circuit measurement.
        false_fn (callable): The quantum function or PennyLane operation to
            apply if ``condition`` is ``False``. The callable must create a single mid-circuit measurement.

    .. note::
        The mid-circuit measurements applied on the two branches must both be applied to the same
        wire, and they must have the same settings for `reset` and `postselection`. The two
        branches can differ only in regard to the measurement basis of the applied measurement.

    Returns:
        function: A new function that applies the conditional measurements. The returned
        function takes the same input arguments as ``true_fn`` and ``false_fn``.

    **Example**

    .. code-block:: python

        from pennylane.ftqc import cond_measure, diagonalize_mcms, measure_x, measure_y
        from functools import partial

        dev = qml.device("default.qubit", wires=3)

        @diagonalize_mcms
        @qml.set_shots(shots=1_000)
        @qml.qnode(dev, mcm_method="one-shot")
        def qnode(x, y):
            qml.RY(x, 0)
            qml.Hadamard(1)

            m0 = qml.measure(0)
            m2 = cond_measure(m0, measure_x, measure_y)(1)

            qml.Hadamard(2)
            qml.cond(m2 == 0, qml.RY)(y, wires=2)
            return qml.expval(qml.X(2))


    >>> print(qnode(np.pi/3, np.pi/2)) # doctest: +SKIP
    0.3914

    .. note::

        If the first argument of ``cond_measure`` is a measurement value (e.g., ``m_0``
        in ``qml.cond(m_0, measure_x, measure_y)``), then ``m_0 == 1`` is considered
        internally.

    .. warning::

        Expressions with boolean logic flow using operators like ``and``,
        ``or`` and ``not`` are not supported as the ``condition`` argument.

        While such statements may not result in errors, they may result in
        incorrect behaviour.
    """
    if capture.enabled():
        cond(condition, true_fn, false_fn)

    if not isinstance(condition, MeasurementValue):
        # The condition is not a mid-circuit measurement - we can simplify immediately
        return CondCallable(condition, true_fn, false_fn)

    if callable(true_fn) and callable(false_fn):

        # We assume this callable is a measurement function that returns a MeasurementValue
        # containing a single mid-circuit measurement. If this isn't the case, getting the
        # measurements will return None, and it will be caught in _validate_measurements.

        @wraps(true_fn)
        def wrapper(*args, **kwargs):

            with QueuingManager.stop_recording():
                true_meas_return = true_fn(*args, **kwargs)
                false_meas_return = false_fn(*args, **kwargs)

                true_meas = getattr(true_meas_return, "measurements", [None])[0]
                false_meas = getattr(false_meas_return, "measurements", [None])[0]

            _validate_measurements(true_meas, false_meas)

            Conditional(condition, true_meas)
            Conditional(~condition, false_meas)

            return MeasurementValue([true_meas, false_meas], processing_fn=lambda v1, v2: v1 or v2)

    else:
        raise ValueError(
            "Only measurement functions can be applied conditionally by `cond_measure`."
        )

    return wrapper


def _validate_measurements(true_meas, false_meas):
    """Takes a pair of variables that are expected to be mid-circuit measurements
    (representing a true and false functions for the conditional) and confirms that
    they have the expected type, and 'match' except for the measurement basis"""

    if not (isinstance(true_meas, MidMeasureMP) and isinstance(false_meas, MidMeasureMP)):
        raise ValueError(
            "Only measurement functions that return a measurement value can be used in `cond_measure`"
        )

    if not (
        true_meas.wires == false_meas.wires
        and true_meas.reset == false_meas.reset
        and true_meas.postselect == false_meas.postselect
    ):
        raise ValueError(
            "When applying a mid-circuit measurement in `cond_measure`, the `wire`, "
            "`postselect` and `reset` behaviour must be consistent for both "
            "branches of the conditional. Only the basis of the measurement (defined "
            "by measurement type or by `plane` and `angle`) can vary."
        )
