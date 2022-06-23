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
from pennylane.operation import Operator, Operation, AdjointUndefinedError, Observable
from pennylane.queuing import QueuingContext
from pennylane.math import transpose, conj


# pylint: disable=no-member
class AdjointOperation(Operation):
    """This mixin class is dynamically added to an ``Adjoint`` instance if the provided base class is an ``Operation``.

    .. warning::
        This mixin class should never be initialized independent of ``Adjoint``.

    Overriding the dunder method ``__new__`` in ``Adjoint`` allows us to customize the creation of an instance and dynamically
    add in parent classes.

    .. note:: Once the ``Operation`` class does not contain any unique logic any more, this mixin class can be removed.
    """

    # This inverse behavior only needs to temporarily patch behavior until in-place inversion is removed.

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
class Adjoint(Operator):
    """
    The Adjoint of an operator.

    Args:
        base (~.operation.Operator): The operator that is adjointed.

    .. seealso:: :func:`~.adjoint`, :meth:`~.operation.Operator.adjoint`

    This is a *developer*-facing class, and the :func:`~.adjoint` transform should be used to construct instances
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

    This class mixes in parent classes based on the inheritance tree of the provided ``Operator``.  For example, when
    provided an ``Operation``, the instance will inherit from ``Operation`` and the ``AdjointOperation`` mixin.

    >>> op = Adjoint(qml.RX(1.234, wires=0))
    >>> isinstance(op, qml.operation.Operation)
    True
    >>> isinstance(op, AdjointOperation)
    True
    >>> op.grad_method
    'A'

    If the base class is an ``Observable`` instead, the ``Adjoint`` will be an ``Observable`` as well.

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

        Though all the types will be named "Adjoint", their *identity* and location in memory will be different
        based on ``base``'s inheritance.  We cache the different types in private class variables so that:

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
                    class_bases = (AdjointOperation, Adjoint, Observable, Operation)
                    cls._operation_observable_type = type(
                        "Adjoint", class_bases, dict(cls.__dict__)
                    )
                return object.__new__(cls._operation_observable_type)

            # not an observable
            if cls._operation_type is None:
                class_bases = (AdjointOperation, Adjoint, Operation)
                cls._operation_type = type("Adjoint", class_bases, dict(cls.__dict__))
            return object.__new__(cls._operation_type)

        if isinstance(base, Observable):
            if cls._observable_type is None:
                class_bases = (Adjoint, Observable)
                cls._observable_type = type("Adjoint", class_bases, dict(cls.__dict__))
            return object.__new__(cls._observable_type)

        return object.__new__(Adjoint)

    # pylint: disable=attribute-defined-outside-init
    def __copy__(self):
        # this method needs to be overwritten becuase the base must be copied too.
        copied_op = object.__new__(type(self))
        # copied_op must maintain inheritance structure of self
        # For example, it must keep AdjointOperation if self has it
        # this way preserves inheritance structure

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
        """The operator that is adjointed."""
        return self.hyperparameters["base"]

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

    # pylint: disable=protected-access
    @property
    def _wires(self):
        return self.base._wires

    # pylint: disable=protected-access
    @_wires.setter
    def _wires(self, new_wires):
        # we should have a better way of updating wires than accessing a private attribute.
        self.base._wires = new_wires

    @property
    def num_wires(self):
        return self.base.num_wires

    @property
    def batch_size(self):
        return self.base.batch_size

    @property
    def ndim_params(self):
        return self.base.ndim_params

    @property
    def is_hermitian(self):
        return self.base.is_hermitian

    def queue(self, context=QueuingContext):
        context.safe_update_info(self.base, owner=self)
        context.append(self, owns=self.base)

        return self

    def label(self, decimals=None, base_label=None, cache=None):
        return self.base.label(decimals, base_label, cache=cache) + "†"

    # pylint: disable=arguments-differ
    @staticmethod
    def compute_matrix(*params, base=None):
        base_matrix = base.compute_matrix(*params, **base.hyperparameters)
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

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_matrix(self):
        return self.base.has_matrix

    def adjoint(self):
        return self.base.queue()

    @property
    def _queue_category(self):
        """Used for sorting objects into their respective lists in `QuantumTape` objects.

        This property is a temporary solution that should not exist long-term and should not be
        used outside of ``QuantumTape._process_queue``.

        Returns ``_queue_cateogory`` for base operator.
        """
        return self.base._queue_category  # pylint: disable=protected-access
