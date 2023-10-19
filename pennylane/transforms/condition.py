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
from pennylane.tape import make_qscript
from pennylane.compiler import compiler


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
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified
    """

    num_wires = AnyWires

    def __init__(
        self,
        expr: MeasurementValue[bool],
        then_op: Type[Operation],
        id=None,
    ):
        self.meas_val = expr
        self.then_op = then_op
        super().__init__(wires=then_op.wires, id=id)


def cond(condition, true_fn, false_fn=None):
    """A :func:`~.qjit` compatible decorator for if-else conditionals in PennyLane/Catalyst.

    This can be used to condition a quantum operation on the results of mid-circuit qubit measurements
    in the interpreted mode while acts like the traditional if-else conditional when is called inside a
    QJIT decorated workflow. This means that each execution path, 'if' and 'else' branchs, is provided
    as a separate function. All functions will be traced during compilation, but only one of them will
    be executed at runtime, depending on the value of one or more Boolean predicates. The JAX equivalent
    is the ``jax.lax.cond`` function, but this version is optimized to work with quantum programs in PennyLane.
    This version also supports an 'else if' construct which the JAX version does not. However to use `else if`
    you need to use `catalyst.cond <https://docs.pennylane.ai/projects/catalyst/en/latest/code/api/catalyst.cond.html>`__


    In the interpreted mode, support for using :func:`~.cond` is device-dependent. If a device doesn't
    support mid-circuit measurements natively, then the QNode will apply the :func:`defer_measurements` transform.

    Args:
        condition (.MeasurementValue): a conditional expression involving a mid-circuit
           measurement value (see :func:`.pennylane.measure`)
        true_fn (callable): The quantum function of PennyLane operation to
            apply if ``condition`` is ``True``
        false_fn (callable): The quantum function of PennyLane operation to
            apply if ``condition`` is ``False``

    Returns:
        function: A new function that applies the conditional equivalent of ``true_fn``. The returned
        function takes the same input arguments as ``true_fn``.

    **Example**

    In the interpreted mode,

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

        Expressions with boolean logic flow using operators like ``and``,
        ``or`` and ``not`` are not supported as the ``condition`` argument.

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

    In the compilation mode,

    .. code-block:: python

        dev = qml.device("lightning.qubit", wires=1)

        @qml.qjit
        @qml.qnode(dev)
        def circuit(x: float):
            def ansatz_true():
                qml.RX(x, wires=0)
                qml.Hadamard(wires=0)

            def ansatz_false():
                qml.RY(x, wires=0)

            qml.cond(x > 1.4, ansatz_true, ansatz_false)

            return qml.expval(qml.PauliZ(0))

    >>> circuit(1.4)
    array(0.16996714)
    >>> circuit(1.6)
    array(0.)

    Additional 'else-if' clauses can also be included via the ``else_if`` method:

    .. code-block:: python

        @qml.qjit
        @qml.qnode(dev)
        def circuit(x):

            @catalyst.cond(x > 2.7)
            def cond_fn():
                qml.RX(x, wires=0)

            @cond_fn.else_if(x > 1.4)
            def cond_elif():
                qml.RY(x, wires=0)

            @cond_fn.otherwise
            def cond_else():
                qml.RX(x ** 2, wires=0)

            cond_fn()

            eturn qml.probs(wires=0)

    The conditional function is permitted to also return values.
    Any value that is supported by JAX JIT compilation is supported as a return
    type. Please see the
    `catalyst.cond <https://docs.pennylane.ai/projects/catalyst/en/latest/code/api/catalyst.cond.html>`__
    page for examples.

    """
    if compiler.active("catalyst"):
        catalyst_compiler = compiler.AvailableCompilers.names_entrypoints["catalyst"]
        ops_loader = catalyst_compiler["ops"].load()
        cond_func = ops_loader.cond(condition)(true_fn)
        if false_fn:
            cond_func.otherwise(false_fn)
        return cond_func()

    if callable(true_fn):
        # We assume that the callable is an operation or a quantum function
        with_meas_err = (
            "Only quantum functions that contain no measurements can be applied conditionally."
        )

        @wraps(true_fn)
        def wrapper(*args, **kwargs):
            # We assume that the callable is a quantum function

            # 1. Apply true_fn conditionally
            qscript = make_qscript(true_fn)(*args, **kwargs)

            if qscript.measurements:
                raise ConditionalTransformError(with_meas_err)

            for op in qscript.operations:
                Conditional(condition, op)

            if false_fn is not None:
                # 2. Apply false_fn conditionally
                else_qscript = make_qscript(false_fn)(*args, **kwargs)

                if else_qscript.measurements:
                    raise ConditionalTransformError(with_meas_err)

                inverted_condition = ~condition

                for op in else_qscript.operations:
                    Conditional(inverted_condition, op)

    else:
        raise ConditionalTransformError(
            "Only operations and quantum functions with no measurements can be applied conditionally."
        )

    return wrapper
