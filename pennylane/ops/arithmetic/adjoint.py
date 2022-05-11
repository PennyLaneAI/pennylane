# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

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
This submodule defines an operation modifier that indicates the adjoint of an operator.
"""
from functools import wraps

from pennylane.operation import Operator, Operation, AnyWires, AdjointUndefinedError
from pennylane.queuing import QueuingContext, QueuingError
from pennylane.tape import QuantumTape, stop_recording
from pennylane.math import transpose, conj


class Adjoint(Operator):
    """
    The Adjoint of an operator.

    Args:
        base (~.operation.Operator): The operator that is adjointed.

    .. seealso:: :func:`~.adjoint`, :meth:`~.operation.Operator.adjoint`

    **Example:**

    >>> op = Adjoint(qml.S(0))
    >>> op.name
    'Adjoint(S)'
    >>> qml.matrix(op)
    array([[1.-0.j, 0.-0.j],
       [0.-0.j, 0.-1.j]])
    >>> qml.generator(Adjoint(qml.RX(1.0, wires=0)))
    (PauliX(wires=[0]), 0.5)
    >>> Adjoint(qml.RX(1.234, wires=0)).data
    [1.234]

    """

    num_wires = AnyWires

    def __copy__(self):
        # this method needs to be overwritten becuase the base must be copied too.
        cls = self.__class__
        copied_op = cls.__new__(cls)
        copied_base = self.base.__copy__()
        copied_op.base = copied_base
        copied_op._hyperparameters = {"base": copied_base}
        for attr, value in vars(self).items():
            if attr not in {"data", "base", "_hyperparameters"}:
                setattr(copied_op, attr, value)

        return copied_op

    def __init__(self, base=None, do_queue=True, id=None):
        self.base = base
        self.hyperparameters["base"] = base
        self._id = id
        self.queue_idx = None

        self._name = f"Adjoint({self.base.name})"

        if do_queue:
            self.queue()

    @property
    def data(self):
        """Trainable parameters that the operator depends on."""
        return self.base.data

    @data.setter
    def data(self, new_data):
        """Allows us to set base operation parameters."""
        self.base.data = new_data

    @property
    def parameters(self):
        return self.base.parameters

    @property
    def num_params(self):
        return self.base.num_params

    @property
    def wires(self):
        return self.base.wires

    def queue(self, context=QueuingContext):
        try:
            context.update_info(self.base, owner=self)
        except QueuingError:
            self.base.queue(context=context)
            context.update_info(self.base, owner=self)

        context.append(self, owns=self.base)

        return self

    def label(self, decimals=None, base_label=None, cache=None):
        return self.base.label(decimals, base_label, cache=cache) + "†"

    @staticmethod
    def compute_matrix(*params, base=None):
        base_matrix = base.compute_matrix(*params, **base.hyperparameters)
        return transpose(conj(base_matrix))

    @staticmethod
    def compute_decomposition(*params, wires, base=None):
        try:
            return [base.adjoint()]
        except AdjointUndefinedError:
            base_decomp = base.compute_decomposition(*params, wires, **base.hyperparameters)
            return [Adjoint(op) for op in reversed(base_decomp)]

    def sparse_matrix(self, wires=None):
        base_matrix = self.base.sparse_matrix(wires=wires)
        return transpose(conj(base_matrix))

    def eigvals(self):
        # Cannot define ``compute_eigvals`` because Hermitian only defines ``get_eigvals``
        return [conj(x) for x in self.base.eigvals()]

    @staticmethod
    def compute_diagonalizing_gates(*params, wires, base=None):
        return base.compute_diagonalizing_gates(*params, wires, **base.hyperparameters)

    @property
    def has_matrix(self):
        return self.base.has_matrix

    def adjoint(self):
        return self.base

    @property
    def _queue_category(self):
        """Used for sorting objects into their respective lists in `QuantumTape` objects.

        This property is a temporary solution that should not exist long-term and should not be
        used outside of ``QuantumTape._process_queue``.

        Returns ``_queue_cateogory`` for base operator.
        """
        return self.base._queue_category

    def generator(self):
        if isinstance(self.base, Operation):  # stand in for being unitary and inverse=adjoint
            return -1.0 * self.base.generator()
        return super().generator()

    ## Operation specific properties ##########################################

    @property
    def basis(self):
        return self.base.basis

    @property
    def control_wires(self):
        return self.base.control_wires

    def single_qubit_rot_angles(self):
        omega, theta, phi = self.base.single_qubit_rot_angles()
        return [-phi, -theta, -omega]

    @property
    def grad_method(self):
        return self.base.grad_method

    @property
    def grad_recipe(self):
        return self.base.grad_recipe

    def get_parameter_shift(self, idx):
        return self.base.get_parameter_shift(idx)

    @property
    def parameter_frequencies(self):
        return self.base.parameter_frequencies


def adjoint(fn):
    """Create a function that applies the adjoint (inverse) of the provided operation or template.

    This transform can be used to apply the adjoint of an arbitrary sequence of operations.

    Args:
        fn (function): A single operator or a quantum function that
            applies quantum operations.

    Returns:
        function: A new function that will apply the same operations but adjointed and in reverse order.

    **Example**

    The adjoint transforms can be used within a QNode to apply the adjoint of
    any quantum function. Consider the following quantum function, that applies two
    operations:

    .. code-block:: python3

        def my_ops(a, wire):
            qml.RX(a, wires=wire)
            qml.SX(wire)

    We can create a QNode that applies this quantum function,
    followed by the adjoint of this function:

    .. code-block:: python3

        dev = qml.device('default.qubit', wires=1)

        @qml.qnode(dev)
        def circuit(a):
            my_ops(a, wire=0)
            qml.adjoint(my_ops)(a, wire=0)
            return qml.expval(qml.PauliZ(0))

    Printing this out, we can see that the inverse quantum
    function has indeed been applied:

    >>> print(qml.draw(circuit)(0.2))
    0: ──RX(0.20)──SX──SX†──RX(-0.20)─┤  <Z>

    If a "shortcut" exists to easily compute the adjoint, see :meth:`~.operation.Operator.adjoint`, then this
    shortcut is eagerly used.  Otherwise, the operator is wrapped in an :class:`~.ops.arithmetic.Adjoint` class
    for later handling. The adjoint function can also be applied directly to templates and operations:

    >>> qml.adjoint(qml.RX)(0.123, wires=0)
    RX(-0.123, wires=[0])
    >>> qml.adjoint(qml.QFT)(wires=(0,1,2,3))
    Adjoint(QFT)(wires=[0, 1, 2, 3])

    .. details::
        :title: Usage Details

        **Adjoint of a function**

        Here, we apply the ``subroutine`` function, and then apply its adjoint.
        Notice that in addition to adjointing all of the operations, they are also
        applied in reverse construction order.

        .. code-block:: python3

            def subroutine(wire):
                qml.RX(0.123, wires=wire)
                qml.RY(0.456, wires=wire)

            dev = qml.device('default.qubit', wires=1)
            @qml.qnode(dev)
            def circuit():
                subroutine(0)
                qml.adjoint(subroutine)(0)
                return qml.expval(qml.PauliZ(0))

        This creates the following circuit:

        >>> print(qml.draw(circuit)())
        0: ──RX(0.12)──T──T†──RX(-0.12)─┤  <Z>

        **Single operation**

        You can also easily adjoint a single operation just by wrapping it with ``adjoint``:

        .. code-block:: python3

            dev = qml.device('default.qubit', wires=1)
            @qml.qnode(dev)
            def circuit():
                qml.RX(0.123, wires=0)
                qml.adjoint(qml.RX)(0.123, wires=0)
                return qml.expval(qml.PauliZ(0))

        This creates the following circuit:

        >>> print(qml.draw(circuit)())
        0: ──RX(0.12)──RX(-0.12)─┤  <Z>
    """
    if not callable(fn):
        raise ValueError(
            f"The object {fn} of type {type(fn)} is not callable. "
            "This error might occur if you apply adjoint to a list "
            "of operations instead of a function or template."
        )

    @wraps(fn)
    def wrapper(*args, **kwargs):
        with stop_recording(), QuantumTape() as tape:
            fn(*args, **kwargs)

        adjoint_ops = [Adjoint(op) for op in reversed(tape)]

        return adjoint_ops[0] if len(adjoint_ops) == 1 else adjoint_ops

    return wrapper
