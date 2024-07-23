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

from pennylane import QueuingManager
from pennylane.compiler import compiler
from pennylane.operation import AnyWires, Operation, Operator
from pennylane.ops.op_math.symbolicop import SymbolicOp
from pennylane.tape import make_qscript


class ConditionalTransformError(ValueError):
    """Error for using qml.cond incorrectly"""


class Conditional(SymbolicOp, Operation):
    """A Conditional Operation.

    Unless you are a Pennylane plugin developer, **you should NOT directly use this class**,
    instead, use the :func:`qml.cond <.cond>` function.

    The ``Conditional`` class is a container class that defines an operation
    that should be applied relative to a single measurement value.

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

    def __init__(self, expr, then_op: Type[Operation], id=None):
        self.hyperparameters["meas_val"] = expr
        self._name = f"Conditional({then_op.name})"
        super().__init__(then_op, id=id)
        if self.grad_recipe is None:
            self.grad_recipe = [None] * self.num_params

    def label(self, decimals=None, base_label=None, cache=None):
        return self.base.label(decimals=decimals, base_label=base_label, cache=cache)

    @property
    def meas_val(self):
        """the measurement outcome value to consider from `expr` argument"""
        return self.hyperparameters["meas_val"]

    @property
    def num_params(self):
        return self.base.num_params

    @property
    def ndim_params(self):
        return self.base.ndim_params

    def map_wires(self, wire_map):
        meas_val = self.meas_val.map_wires(wire_map)
        then_op = self.base.map_wires(wire_map)
        return Conditional(meas_val, then_op=then_op)

    def matrix(self, wire_order=None):
        return self.base.matrix(wire_order=wire_order)

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_diagonalizing_gates(self):
        return self.base.has_diagonalizing_gates

    def diagonalizing_gates(self):
        return self.base.diagonalizing_gates()

    def eigvals(self):
        return self.base.eigvals()

    @property
    def has_adjoint(self):
        return self.base.has_adjoint

    def adjoint(self):
        return Conditional(self.meas_val, self.base.adjoint())


def cond(condition, true_fn, false_fn=None, elifs=()):
    """Quantum-compatible if-else conditionals --- condition quantum operations
    on parameters such as the results of mid-circuit qubit measurements.

    This method is restricted to simply branching on mid-circuit measurement
    results when it is not used with the :func:`~.qjit` decorator.

    When used with the :func:`~.qjit` decorator, this function allows for general
    if-elif-else constructs. All ``true_fn``, ``false_fn`` and ``elifs`` branches
    will be captured by Catalyst, the just-in-time (JIT) compiler, with the executed
    branch determined at runtime. For more details, please see :func:`catalyst.cond`.

    .. note::

        With the Python interpreter, support for :func:`~.cond`
        is device-dependent. If a device doesn't
        support mid-circuit measurements natively, then the QNode will
        apply the :func:`defer_measurements` transform.

    .. note::
        When used with :func:`~.qjit`, this function only supports
        the Catalyst compiler. See :func:`catalyst.cond` for more details.

        Please see the Catalyst :doc:`quickstart guide <catalyst:dev/quick_start>`,
        as well as the :doc:`sharp bits and debugging tips <catalyst:dev/sharp_bits>`.

    Args:
        condition (Union[.MeasurementValue, bool]): a conditional expression involving a mid-circuit
           measurement value (see :func:`.pennylane.measure`). This can only be of type ``bool`` when
           decorated by :func:`~.qjit`.
        true_fn (callable): The quantum function or PennyLane operation to
            apply if ``condition`` is ``True``
        false_fn (callable): The quantum function or PennyLane operation to
            apply if ``condition`` is ``False``
        elifs (List(Tuple(bool, callable))): A list of (bool, elif_fn) clauses. Can only
            be used when decorated by :func:`~.qjit`.

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

    In just-in-time (JIT) mode using the :func:`~.qjit` decorator,

    .. code-block:: python3

        dev = qml.device("lightning.qubit", wires=1)

        @qml.qjit
        @qml.qnode(dev)
        def circuit(x: float):
            def ansatz_true():
                qml.RX(x, wires=0)
                qml.Hadamard(wires=0)

            def ansatz_false():
                qml.RY(x, wires=0)

            qml.cond(x > 1.4, ansatz_true, ansatz_false)()

            return qml.expval(qml.Z(0))

    >>> circuit(1.4)
    array(0.16996714)
    >>> circuit(1.6)
    array(0.)

    Additional 'else-if' clauses can also be included via the ``elif`` argument:

    .. code-block:: python3

        @qml.qjit
        @qml.qnode(dev)
        def circuit(x):

            def true_fn():
                qml.RX(x, wires=0)

            def elif_fn():
                qml.RY(x, wires=0)

            def false_fn():
                qml.RX(x ** 2, wires=0)

            qml.cond(x > 2.7, true_fn, false_fn, ((x > 1.4, elif_fn),))()
            return qml.expval(qml.Z(0))

    >>> circuit(1.2)
    array(0.13042371)

    .. details::
        :title: Usage Details

        **Conditional quantum functions**

        The ``cond`` transform allows conditioning quantum functions too:

        .. code-block:: python3

            dev = qml.device("default.qubit")

            def qfunc(par, wires):
                qml.Hadamard(wires[0])
                qml.RY(par, wires[0])

            @qml.qnode(dev)
            def qnode(x):
                qml.Hadamard(0)
                m_0 = qml.measure(0)
                qml.cond(m_0, qfunc)(x, wires=[1])
                return qml.expval(qml.Z(1))

        .. code-block :: pycon

            >>> par = np.array(0.3, requires_grad=True)
            >>> qnode(par)
            tensor(0.3522399, requires_grad=True)

        **Postprocessing multiple measurements into a condition**

        The Boolean condition for ``cond`` may consist of arithmetic expressions
        of one or multiple mid-circuit measurements:

        .. code-block:: python3

            def cond_fn(mcms):
                first_term = np.prod(mcms)
                second_term = (2 ** np.arange(len(mcms))) @ mcms
                return (1 - first_term) * (second_term > 3)

            @qml.qnode(dev)
            def qnode(x):
                ...
                mcms = [qml.measure(w) for w in range(4)]
                qml.cond(cond_fn(mcms), qml.RX)(x, wires=4)
                ...
                return qml.expval(qml.Z(1))

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
                return qml.expval(qml.Z(1))

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
                return qml.expval(qml.Z(1))

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
                return qml.expval(qml.Z(1))

        .. code-block :: pycon

            >>> par = np.array(0.3, requires_grad=True)
            >>> x = np.array(1.2, requires_grad=True)
            >>> y = np.array(1.1, requires_grad=True)
            >>> z = np.array(0.3, requires_grad=True)
            >>> qnode(par, x, y, z)
            tensor(-0.30922805, requires_grad=True)
    """
    if active_jit := compiler.active_compiler():
        available_eps = compiler.AvailableCompilers.names_entrypoints
        ops_loader = available_eps[active_jit]["ops"].load()
        cond_func = ops_loader.cond(condition)(true_fn)

        # Optional 'elif' branches
        for cond_val, elif_fn in elifs:
            cond_func.else_if(cond_val)(elif_fn)

        # Optional 'else' branch
        if false_fn:
            cond_func.otherwise(false_fn)

        return cond_func

    if elifs:
        raise ConditionalTransformError("'elif' branches are not supported in interpreted mode.")

    if callable(true_fn):
        # We assume that the callable is an operation or a quantum function
        with_meas_err = (
            "Only quantum functions that contain no measurements can be applied conditionally."
        )

        @wraps(true_fn)
        def wrapper(*args, **kwargs):
            # We assume that the callable is a quantum function

            recorded_ops = [a for a in args if isinstance(a, Operator)] + [
                k for k in kwargs.values() if isinstance(k, Operator)
            ]

            # This will dequeue all operators passed in as arguments to the qfunc that is
            # being conditioned. These are queued incorrectly due to be fully constructed
            # before the wrapper function is called.
            if recorded_ops and QueuingManager.recording():
                for op in recorded_ops:
                    QueuingManager.remove(op)

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
