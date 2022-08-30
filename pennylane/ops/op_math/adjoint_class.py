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
This submodule defines the symbolic operation that indicates the adjoint of an operator.
"""
import pennylane as qml
from pennylane.math import conj, transpose
from pennylane.operation import AdjointUndefinedError, Observable, Operation

from .symbolicop import SymbolicOp


# pylint: disable=no-member
class AdjointOperation(Operation):
    """This mixin class is dynamically added to an ``Adjoint`` instance if the provided base class
    is an ``Operation``.

    .. warning::
        This mixin class should never be initialized independent of ``Adjoint``.

    Overriding the dunder method ``__new__`` in ``Adjoint`` allows us to customize the creation of
    an instance and dynamically add in parent classes.

    .. note:: Once the ``Operation`` class does not contain any unique logic any more, this mixin
    class can be removed.
    """

    # This inverse behavior only needs to temporarily patch behavior until in-place inversion is
    # removed.

    @property
    def _inverse(self):
        return self.base._inverse  # pylint: disable=protected-access

    @_inverse.setter
    def _inverse(self, boolean):
        self.base._inverse = boolean  # pylint: disable=protected-access
        # refresh name as base_name got updated.
        self._name = f"Adjoint({self.base.name})"

    def inv(self):
        self.base.inv()
        # refresh name as base_name got updated.
        self._name = f"Adjoint({self.base.name})"
        return self

    @property
    def base_name(self):
        return self._name

    @property
    def name(self):
        return self._name

    # pylint: disable=missing-function-docstring
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

    # pylint: disable=missing-function-docstring
    @property
    def grad_recipe(self):
        return self.base.grad_recipe

    def get_parameter_shift(self, idx):
        return self.base.get_parameter_shift(idx)

    @property
    def parameter_frequencies(self):
        return self.base.parameter_frequencies

    def generator(self):
        return -1.0 * self.base.generator()


# pylint: disable=too-many-public-methods
class Adjoint(SymbolicOp):
    """
    The Adjoint of an operator.

    Args:
        base (~.operation.Operator): The operator that is adjointed.

    .. seealso:: :func:`~.adjoint`, :meth:`~.operation.Operator.adjoint`

    This is a *developer*-facing class, and the :func:`~.adjoint` transform should be used to
    construct instances
    of this class.

    **Example**

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

    .. details::
        :title: Developer Details

    This class mixes in parent classes based on the inheritance tree of the provided ``Operator``.
    For example, when provided an ``Operation``, the instance will inherit from ``Operation`` and
    the ``AdjointOperation`` mixin.

    >>> op = Adjoint(qml.RX(1.234, wires=0))
    >>> isinstance(op, qml.operation.Operation)
    True
    >>> isinstance(op, AdjointOperation)
    True
    >>> op.grad_method
    'A'

    If the base class is an ``Observable`` instead, the ``Adjoint`` will be an ``Observable`` as
    well.

    >>> op = Adjoint(1.0 * qml.PauliX(0))
    >>> isinstance(op, qml.operation.Observable)
    True
    >>> isinstance(op, qml.operation.Operation)
    False
    >>> Adjoint(qml.PauliX(0)) @ qml.PauliY(1)
    Adjoint(PauliX)(wires=[0]) @ PauliY(wires=[1])

    """

    _operation_type = None  # type if base inherits from operation and not observable
    _operation_observable_type = None  # type if base inherits from both operation and observable
    _observable_type = None  # type if base inherits from observable and not operation

    # pylint: disable=unused-argument
    def __new__(cls, base=None, do_queue=True, id=None):
        """Mixes in parents based on inheritance structure of base.

        Though all the types will be named "Adjoint", their *identity* and location in memory will
        be different based on ``base``'s inheritance.  We cache the different types in private class
        variables so that:

        >>> Adjoint(op).__class__ is Adjoint(op).__class__
        True
        >>> type(Adjoint(op)) == type(Adjoint(op))
        True
        >>> Adjoint(qml.RX(1.2, wires=0)).__class__ is Adjoint._operation_type
        True
        >>> Adjoint(qml.PauliX(0)).__class__ is Adjoint._operation_observable_type
        True

        """

        if isinstance(base, Operation):
            if isinstance(base, Observable):
                if cls._operation_observable_type is None:
                    class_bases = (AdjointOperation, Adjoint, SymbolicOp, Observable, Operation)
                    cls._operation_observable_type = type(
                        "Adjoint", class_bases, dict(cls.__dict__)
                    )
                return object.__new__(cls._operation_observable_type)

            # not an observable
            if cls._operation_type is None:
                class_bases = (AdjointOperation, Adjoint, SymbolicOp, Operation)
                cls._operation_type = type("Adjoint", class_bases, dict(cls.__dict__))
            return object.__new__(cls._operation_type)

        if isinstance(base, Observable):
            if cls._observable_type is None:
                class_bases = (Adjoint, SymbolicOp, Observable)
                cls._observable_type = type("Adjoint", class_bases, dict(cls.__dict__))
            return object.__new__(cls._observable_type)

        return object.__new__(Adjoint)

    def __init__(self, base=None, do_queue=True, id=None):
        self._name = f"Adjoint({base.name})"
        super().__init__(base, do_queue=do_queue, id=id)

    def __repr__(self):
        if self.arithmetic_depth == 1:
            return super().__repr__()
        return f"Adjoint({self.base})"

    def label(self, decimals=None, base_label=None, cache=None):
        base_label = self.base.label(decimals, base_label, cache=cache)
        return f"({base_label})†" if self.base.arithmetic_depth > 0 else f"{base_label}†"

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_matrix(self):
        return self.base.has_matrix if self.base.batch_size is None else False

    def matrix(self, wire_order=None):
        if getattr(self.base, "batch_size", None) is not None:
            raise qml.operation.MatrixUndefinedError

        if isinstance(self.base, qml.Hamiltonian):
            base_matrix = qml.matrix(self.base, wire_order=wire_order)
        else:
            base_matrix = self.base.matrix(wire_order=wire_order)

        return transpose(conj(base_matrix))

    def decomposition(self):
        try:
            return [self.base.adjoint()]
        except AdjointUndefinedError:
            base_decomp = self.base.decomposition()
            return [Adjoint(op) for op in reversed(base_decomp)]

    # pylint: disable=arguments-differ
    @staticmethod
    def compute_sparse_matrix(*params, base=None):
        base_matrix = base.compute_sparse_matrix(*params, **base.hyperparameters)
        return transpose(conj(base_matrix)).tocsr()

    def eigvals(self):
        # Cannot define ``compute_eigvals`` because Hermitian only defines ``eigvals``
        return conj(self.base.eigvals())

    def diagonalizing_gates(self):
        return self.base.diagonalizing_gates()

    def adjoint(self):
        return self.base.queue()

    def simplify(self):
        try:
            return self.base.adjoint().simplify()
        except AdjointUndefinedError:
            return Adjoint(base=self.base.simplify())
