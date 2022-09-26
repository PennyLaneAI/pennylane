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
        self, expr: MeasurementValue[bool], then_op: Type[Operation], do_queue=True, id=None,
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

        @qml.qnode(dev)
        def qnode(x, y):
            qml.Hadamard(0)
            m_0 = qml.measure(0)
            qml.cond(m_0, qml.RY)(x, wires=1)

            qml.Hadamard(2)
            qml.RY(-np.pi/2, wires=[2])
            m_1 = qml.measure(2)
            qml.cond(m_1 == 0, qml.RX)(y, wires=1)
            return qml.expval(qml.PauliZ(1))

    .. code-block :: pycon

        >>> first_par = np.array(0.3, requires_grad=True)
        >>> sec_par = np.array(1.23, requires_grad=True)
        >>> qnode(first_par, sec_par)
        tensor(0.32677361, requires_grad=True)

    .. note::

        If the first argument of ``cond`` is a measurement value (e.g., ``m_0``
        in ``qml.cond(m_0, qml.RY)``), then ``m_0 == 1`` is considered
        internally.

    .. warning::

        The following are not supported as the ``condition`` argument:

        * Expressions that contain multiple measurement values;
        * Expressions with boolean logic flow using operators like ``and``,
          ``or`` and ``not``.

        While such statements may not result in errors, they may result in
        incorrect behaviour.


    .. details::
        :title: Usage Details

        **Conditional quantum functions**

        The ``cond`` transform allows conditioning quantum functions too:

        .. code-block:: python3

            dev = qml.device("default.qubit", wires=2)

            def qfunc(par, wires):
                qml.Hadamard(wires[0])
                qml.RY(par, wires[0])

            @qml.qnode(dev)
            def qnode(x):
                qml.Hadamard(0)
                m_0 = qml.measure(0)
                qml.cond(m_0, qfunc)(x, wires=[1])
                return qml.expval(qml.PauliZ(1))

        .. code-block :: pycon

            >>> par = np.array(0.3, requires_grad=True)
            >>> qnode(par)
            tensor(0.3522399, requires_grad=True)

        **Passing two quantum functions**

        In the qubit model, single-qubit measurements may result in one of two
        outcomes. Such measurement outcomes may then be used to create
        conditional expressions.

        According to the truth value of the conditional expression passed to
        ``cond``, the transform can apply a quantum function in both the
        ``True`` and ``False`` case:

        .. code-block:: python3

            dev = qml.device("default.qubit", wires=2)

            def qfunc1(x, wires):
                qml.Hadamard(wires[0])
                qml.RY(x, wires[0])

            def qfunc2(x, wires):
                qml.Hadamard(wires[0])
                qml.RZ(x, wires[0])

            @qml.qnode(dev)
            def qnode1(x):
                qml.Hadamard(0)
                m_0 = qml.measure(0)
                qml.cond(m_0, qfunc1, qfunc2)(x, wires=[1])
                return qml.expval(qml.PauliZ(1))

        .. code-block :: pycon

            >>> par = np.array(0.3, requires_grad=True)
            >>> qnode1(par)
            tensor(-0.1477601, requires_grad=True)

        The previous QNode is equivalent to using ``cond`` twice, inverting the
        conditional expression in the second case using the ``~`` unary
        operator:

        .. code-block:: python3

            @qml.qnode(dev)
            def qnode2(x):
                qml.Hadamard(0)
                m_0 = qml.measure(0)
                qml.cond(m_0, qfunc1)(x, wires=[1])
                qml.cond(~m_0, qfunc2)(x, wires=[1])
                return qml.expval(qml.PauliZ(1))

        .. code-block :: pycon

            >>> qnode2(par)
            tensor(-0.1477601, requires_grad=True)

        **Quantum functions with different signatures**

        It may be that the two quantum functions passed to ``qml.cond`` have
        different signatures. In such a case, ``lambda`` functions taking no
        arguments can be used with Python closure:

        .. code-block:: python3

            dev = qml.device("default.qubit", wires=2)

            def qfunc1(x, wire):
                qml.Hadamard(wire)
                qml.RY(x, wire)

            def qfunc2(x, y, z, wire):
                qml.Hadamard(wire)
                qml.Rot(x, y, z, wire)

            @qml.qnode(dev)
            def qnode(a, x, y, z):
                qml.Hadamard(0)
                m_0 = qml.measure(0)
                qml.cond(m_0, lambda: qfunc1(a, wire=1), lambda: qfunc2(x, y, z, wire=1))()
                return qml.expval(qml.PauliZ(1))

        .. code-block :: pycon

            >>> par = np.array(0.3, requires_grad=True)
            >>> x = np.array(1.2, requires_grad=True)
            >>> y = np.array(1.1, requires_grad=True)
            >>> z = np.array(0.3, requires_grad=True)
            >>> qnode(par, x, y, z)
            tensor(-0.30922805, requires_grad=True)
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

                inverted_condition = ~condition

                for op in else_tape.operations:
                    Conditional(inverted_condition, op)

    else:
        raise ConditionalTransformError(
            "Only operations and quantum functions with no measurements can be applied conditionally."
        )

    return wrapper
