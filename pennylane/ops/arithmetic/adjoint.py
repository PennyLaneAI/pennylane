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


# pylint: disable=too-many-public-methods
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

    # pylint: disable=attribute-defined-outside-init
    def __copy__(self):
        # this method needs to be overwritten becuase the base must be copied too.
        cls = self.__class__
        copied_op = cls.__new__(cls)
        copied_base = self.base.__copy__()
        copied_op._hyperparameters = {"base": copied_base}
        for attr, value in vars(self).items():
            if attr not in {"data", "base", "_hyperparameters"}:
                setattr(copied_op, attr, value)

        return copied_op

    # pylint: disable=super-init-not-called
    def __init__(self, base=None, do_queue=True, id=None):
        self.hyperparameters["base"] = base
        self._id = id
        self.queue_idx = None

        self._name = f"Adjoint({self.base.name})"

        if do_queue:
            self.queue()

    @property
    def base(self):
        return self.hyperparameters['base']

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

    # pylint: disable=arguments-differ
    @staticmethod
    def compute_matrix(*params, base=None):
        base_matrix = base.compute_matrix(*params, **base.hyperparameters)
        return transpose(conj(base_matrix))

    # pylint: disable=arguments-differ
    @staticmethod
    def compute_decomposition(*params, wires, base=None):
        try:
            return [base.adjoint()]
        except AdjointUndefinedError:
            base_decomp = base.compute_decomposition(*params, wires, **base.hyperparameters)
            return [Adjoint(op) for op in reversed(base_decomp)]

    # pylint: disable=arguments-differ
    @staticmethod
    def compute_sparse_matrix(*params, base=None):

        base_matrix = base.compute_sparse_matrix(*params, **base.hyperparameters)
        return transpose(conj(base_matrix))

    def eigvals(self):
        # Cannot define ``compute_eigvals`` because Hermitian only defines ``get_eigvals``
        return [conj(x) for x in self.base.eigvals()]

    # pylint: disable=arguments-differ
    @staticmethod
    def compute_diagonalizing_gates(*params, wires, base=None):
        return base.compute_diagonalizing_gates(*params, wires, **base.hyperparameters)

    # pylint: disable=arguments-renamed
    @property
    def has_matrix(self):
        return self.base.has_matrix

    # pylint: disable=arguments-differ
    def adjoint(self):
        return self.base

    @property
    def _queue_category(self):
        """Used for sorting objects into their respective lists in `QuantumTape` objects.

        This property is a temporary solution that should not exist long-term and should not be
        used outside of ``QuantumTape._process_queue``.

        Returns ``_queue_cateogory`` for base operator.
        """
        return self.base._queue_category  # pylint: disable=protected-access

    def generator(self):
        if isinstance(self.base, Operation):  # stand in for being unitary and inverse=adjoint
            return -1.0 * self.base.generator()
        return super().generator()

    ## Operation specific properties ##########################################

    @property
    def basis(self):
        """str or None: The target operation for controlled gates.
        target operation. If not ``None``, should take a value of ``"X"``, ``"Y"``,
        or ``"Z"``.

        For example, ``X`` and ``CNOT`` have ``basis = "X"``, whereas
        ``ControlledPhaseShift`` and ``RZ`` have ``basis = "Z"``.
        """
        return self.base.basis

    @property
    def control_wires(self):
        r"""Control wires of the operator.

        For operations that are not controlled,
        this is an empty ``Wires`` object of length ``0``.

        Returns:
            Wires: The control wires of the operation.
        """
        return self.base.control_wires

    def single_qubit_rot_angles(self):
        r"""The parameters required to implement a single-qubit gate as an
        equivalent ``Rot`` gate, up to a global phase.

        Returns:
            tuple[float, float, float]: A list of values :math:`[\phi, \theta, \omega]`
            such that :math:`RZ(\omega) RY(\theta) RZ(\phi)` is equivalent to the
            original operation.
        """
        omega, theta, phi = self.base.single_qubit_rot_angles()
        return [-phi, -theta, -omega]

    @property
    def grad_method(self):
        """Gradient computation method.

        * ``'A'``: analytic differentiation using the parameter-shift method.
        * ``'F'``: finite difference numerical differentiation.
        * ``None``: the operation may not be differentiated.

        Default is ``'F'``, or ``None`` if the Operation has zero parameters.
        """
        return self.base.grad_method

    @property
    def grad_recipe(self):
        r"""tuple(Union(list[list[float]], None)) or None: Gradient recipe for the
        parameter-shift method.

        This is a tuple with one nested list per operation parameter. For
        parameter :math:`\phi_k`, the nested list contains elements of the form
        :math:`[c_i, a_i, s_i]` where :math:`i` is the index of the
        term, resulting in a gradient recipe of

        .. math:: \frac{\partial}{\partial\phi_k}f = \sum_{i} c_i f(a_i \phi_k + s_i).

        If ``None``, the default gradient recipe containing the two terms
        :math:`[c_0, a_0, s_0]=[1/2, 1, \pi/2]` and :math:`[c_1, a_1,
        s_1]=[-1/2, 1, -\pi/2]` is assumed for every parameter.
        """
        return self.base.grad_recipe

    def get_parameter_shift(self, idx):
        r"""Multiplier and shift for the given parameter, based on its gradient recipe.

        Args:
            idx (int): parameter index within the operation

        Returns:
            list[[float, float, float]]: list of multiplier, coefficient, shift for each term in the gradient recipe

        Note that the default value for ``shift`` is None, which is replaced by the
        default shift :math:`\pi/2`.
        """
        return self.base.get_parameter_shift(idx)

    @property
    def parameter_frequencies(self):
        r"""Returns the frequencies for each operator parameter with respect
        to an expectation value of the form
        :math:`\langle \psi | U(\mathbf{p})^\dagger \hat{O} U(\mathbf{p})|\psi\rangle`.

        These frequencies encode the behaviour of the operator :math:`U(\mathbf{p})`
        on the value of the expectation value as the parameters are modified.
        For more details, please see the :mod:`.pennylane.fourier` module.

        Returns:
            list[tuple[int or float]]: Tuple of frequencies for each parameter.
            Note that only non-negative frequency values are returned.

        **Example**

        >>> op = qml.CRot(0.4, 0.1, 0.3, wires=[0, 1])
        >>> op.parameter_frequencies
        [(0.5, 1), (0.5, 1), (0.5, 1)]

        For operators that define a generator, the parameter frequencies are directly
        related to the eigenvalues of the generator:

        >>> op = qml.ControlledPhaseShift(0.1, wires=[0, 1])
        >>> op.parameter_frequencies
        [(1,)]
        >>> gen = qml.generator(op, format="observable")
        >>> gen_eigvals = qml.eigvals(gen)
        >>> qml.gradients.eigvals_to_frequencies(tuple(gen_eigvals))
        (1.0,)

        For more details on this relationship, see :func:`.eigvals_to_frequencies`.
        """
        return self.base.parameter_frequencies


def adjoint(fn, lazy=True):
    """Create a function that applies the adjoint of the provided operation or template.

    This transform can be used to apply the adjoint of an arbitrary sequence of operations.

    Args:
        fn (function): A single operator or a quantum function that
            applies quantum operations.

    Keyword Args:
        lazy=True (bool): If the transform is behaving lazily, all operations are wrapped in a `Adjoint` class
            and handled later. If ``lazy=False``, operation-specific adjoint decompositions are first attempted.

    Returns:
        function: A new function that will apply the same operations but adjointed and in reverse order.

    .. note::

        While the adjoint and inverse are identical for Unitary gates, not all possible operators are Unitary.
        This transform can also act on Channels and Hamiltonians, for which the inverse and adjoint are different.

    .. seealso:: :class:`~.ops.arithmetic.Adjoint` and :meth:`~.operation.Operator.adjoint`

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
    0: ──RX(0.20)──SX──SX†──RX(0.20)†─┤  <Z>

    .. details::
        :title: Usage Details

        **Adjoint of a function**

        Here, we apply the ``subroutine`` function, and then apply its adjoint.
        Notice that in addition to adjointing all of the operations, they are also
        applied in reverse construction order. Some `Adjoint` gates like those wrapping ``SX``, ``S``, and
        ``T`` are natively supported by ``default.qubit``. Other gates will be expanded either using a custom
        adjoint decomposition defined in :meth:`~.operation.Operator.adjoint`.

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
        0: ──RX(0.12)──S──S†──RX(0.12)†─┤  <Z>
        >>> print(qml.draw(circuit, expansion_strategy="device")())
        0: ──RX(0.12)──S──S†──RX(-0.12)─┤  <Z>

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
        0: ──RX(0.12)──RX(0.12)†─┤  <Z>


        :title: Developer details

        **Lazy Evaluation**

        When ``lazy=False``, the function first attempts operation-specific decomposition of the
        adjoint via the :meth:`.operation.Operator.adjoint` method. Only if an Operator doesn't have
        an :meth:`.operation.Operator.adjoint` method is the object wrapped with the :class:`~.ops.arithmetic.Adjoint`
        wrapper class.

        >>> qml.adjoint(qml.PauliZ, lazy=False)(0)
        PauliZ(wires=[0])
        >>> qml.adjoint(qml.RX, lazy=False)(1.0, wires=0)
        RX(-1.0, wires=[0])
        >>> qml.adjoint(qml.S, lazy=False)(0)
        Adjoint(S)(wires=[0])

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

        if lazy:
            adjoint_ops = [Adjoint(op) for op in reversed(tape.operations)]
            return adjoint_ops[0] if len(adjoint_ops) == 1 else adjoint_ops

        def _op_adjoint(op):
            try:
                return op.adjoint()
            except AdjointUndefinedError:
                return Adjoint(op)

        adjoint_ops = [_op_adjoint(op) for op in reversed(tape.operations)]
        return adjoint_ops[0] if len(adjoint_ops) == 1 else adjoint_ops

    return wrapper
