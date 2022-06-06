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
This submodule defines the symbolic operation that stands for the power of an operator.
"""
from copy import copy
from scipy.linalg import fractional_matrix_power

from pennylane.operation import (
    DecompositionUndefinedError,
    SparseMatrixUndefinedError,
    PowUndefinedError,
    Operator,
    Operation,
    Observable,
    expand_matrix,
)
from pennylane.queuing import QueuingContext, apply
from pennylane.wires import Wires

from pennylane import math as qmlmath

_superscript = str.maketrans("0123456789.+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⋅⁺⁻")


# pylint: disable=no-member
class PowOperation(Operation):
    """Operation-specific methods and properties for the ``Pow`` class.

    Dynamically mixed in based on the provided base operator.  If the base operator is an
    Operation, this class will be mixed in.

    When we no longer rely on certain functionality through `Operation`, we can get rid of this
    class.
    """

    # until we add gradient support
    grad_method = None

    def inv(self):
        self.hyperparameters["z"] *= -1
        self._name = f"{self.base.name}**{self.z}"
        return self

    @property
    def inverse(self):
        return False

    @inverse.setter
    def inverse(self, boolean):
        if boolean is True:
            raise NotImplementedError("The inverse can not be set for a power operator")

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


class Pow(Operator):
    """Symbolic operator denoting an operator raised to a power.

    Args:
        base (~.operation.Operator): the operator to be raised to a power
        z=1 (float): the exponent

    **Example**

    >>> sqrt_x = Pow(qml.PauliX(0), 0.5)
    >>> sqrt_x.decomposition()
    [SX(wires=[0])]
    >>> qml.matrix(sqrt_x)
    array([[0.5+0.5j, 0.5-0.5j],
                [0.5-0.5j, 0.5+0.5j]])
    >>> qml.matrix(qml.SX(0))
    array([[0.5+0.5j, 0.5-0.5j],
       [0.5-0.5j, 0.5+0.5j]])
    >>> qml.matrix(Pow(qml.T(0), 1.234))
    array([[1.        +0.j        , 0.        +0.j        ],
       [0.        +0.j        , 0.56597465+0.82442265j]])

    """

    _operation_type = None  # type if base inherits from operation and not observable
    _operation_observable_type = None  # type if base inherits from both operation and observable
    _observable_type = None  # type if base inherits from observable and not oepration

    # pylint: disable=unused-argument
    def __new__(cls, base=None, z=1, do_queue=True, id=None):
        """Mixes in parents based on inheritance structure of base.

        Though all the types will be named "Pow", their *identity* and location in memory will be different
        based on ``base``'s inheritance.  We cache the different types in private class variables so that:

        """

        if isinstance(base, Operation):
            if isinstance(base, Observable):
                if cls._operation_observable_type is None:
                    base_classes = (PowOperation, Pow, Observable, Operation)
                    cls._operation_observable_type = type("Pow", base_classes, dict(cls.__dict__))
                return object.__new__(cls._operation_observable_type)

            # not an observable
            if cls._operation_type is None:
                base_classes = (PowOperation, Pow, Operation)
                cls._operation_type = type("Pow", base_classes, dict(cls.__dict__))
            return object.__new__(cls._operation_type)

        if isinstance(base, Observable):
            if cls._observable_type is None:
                base_classes = (Pow, Observable)
                cls._observable_type = type("Pow", base_classes, dict(cls.__dict__))
            return object.__new__(cls._observable_type)

        return object.__new__(Pow)

    # pylint: disable=attribute-defined-outside-init
    def __copy__(self):
        # this method needs to be overwritten becuase the base must be copied too.
        copied_op = object.__new__(type(self))
        # copied_op must maintain inheritance structure of self
        # For example, it must keep AdjointOperation if self has it
        # this way preserves inheritance structure

        for attr, value in vars(self).items():
            if attr != "_hyperparameters":
                setattr(copied_op, attr, value)
        copied_op._hyperparameters = copy(self._hyperparameters)
        copied_op._hyperparameters["base"] = copy(self.base)

        return copied_op

    # pylint: disable=super-init-not-called
    def __init__(self, base=None, z=1, do_queue=True, id=None):

        # incorporate base inverse attribute into the exponent
        if getattr(base, "inverse", False):
            base.inverse = False
            z *= -1

        self.hyperparameters["base"] = base
        self.hyperparameters["z"] = z
        self._id = id
        self.queue_idx = None

        self._name = f"{self.base.name}**{z}"

        if do_queue:
            self.queue()

    @property
    def base(self):
        """The operator that is raised to a power."""
        return self.hyperparameters["base"]

    @property
    def z(self):
        """The exponent."""
        return self.hyperparameters["z"]

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
        # used in a couple places that want to update the wires of an operator
        # we should create a better way to set new wires in the future
        self.base._wires = new_wires

    @property
    def num_wires(self):
        return len(self.wires)

    def queue(self, context=QueuingContext):
        context.safe_update_info(self.base, owner=self)
        context.append(self, owns=self.base)

        return self

    def label(self, decimals=None, base_label=None, cache=None):
        z_string = format(self.z).translate(_superscript)
        return self.base.label(decimals, base_label, cache=cache) + z_string

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_matrix(self):
        return self.base.has_matrix

    @property
    def is_hermitian(self):
        return self.base.is_hermitian

    def matrix(self, wire_order=None):
        base_matrix = self.base.matrix()

        if isinstance(self.z, int):
            mat = qmlmath.linalg.matrix_power(base_matrix, self.z)
        else:
            mat = fractional_matrix_power(base_matrix, self.z)

        if wire_order is None or self.wires == Wires(wire_order):
            return mat

        return expand_matrix(mat, wires=self.wires, wire_order=wire_order)

    # pylint: disable=arguments-differ
    @staticmethod
    def compute_sparse_matrix(*params, base=None, z=0):
        if isinstance(z, int):
            base_matrix = base.compute_sparse_matrix(*params, **base.hyperparameters)
            return base_matrix**z
        raise SparseMatrixUndefinedError

    def decomposition(self):
        try:
            return self.base.pow(self.z)
        except PowUndefinedError as e:
            if isinstance(self.z, int) and self.z > 0:
                if QueuingContext.recording():
                    return [apply(self.base) for _ in range(self.z)]
                return [self.base.__copy__() for _ in range(self.z)]
            # TODO: consider: what if z is an int and less than 0?
            # do we want Pow(base, -1) to be a "more fundamental" op
            raise DecompositionUndefinedError from e

    def diagonalizing_gates(self):
        r"""Sequence of gates that diagonalize the operator in the computational basis.

        Given the eigendecomposition :math:`O = U \Sigma U^{\dagger}` where
        :math:`\Sigma` is a diagonal matrix containing the eigenvalues,
        the sequence of diagonalizing gates implements the unitary :math:`U`.

        The diagonalizing gates of an operator to a power is the same as the diagonalizing
        gates as the original operator. As we can see,

        .. math::

            O^2 = U \Sigma U^{\dagger} U \Sigma U^{\dagger} = U \Sigma^2 U^{\dagger}

        This formula can be extended to inversion and any rational number.

        The diagonalizing gates rotate the state into the eigenbasis
        of the operator.

        A ``DiagGatesUndefinedError`` is raised if no representation by decomposition is defined.

        .. seealso:: :meth:`~.Operator.compute_diagonalizing_gates`.

        Returns:
            list[.Operator] or None: a list of operators
        """
        return self.base.diagonalizing_gates()

    def eigvals(self):
        base_eigvals = self.base.eigvals()
        return [value**self.z for value in base_eigvals]

    def generator(self):
        r"""Generator of an operator that is in single-parameter-form.

        The generator of a power operator is ``z`` times the generator of the
        base matrix.

        .. math::

            U(\phi)^z = e^{i\phi (z G)}

        See also :func:`~.generator`
        """
        return self.z * self.base.generator()

    @property
    def _queue_category(self):
        """Used for sorting objects into their respective lists in `QuantumTape` objects.

        This property is a temporary solution that should not exist long-term and should not be
        used outside of ``QuantumTape._process_queue``.

        Returns ``_queue_cateogory`` for base operator.

        Options are:
            * `"_prep"`
            * `"_ops"`
            * `"_measurements"`
            * `None`
        """
        return self.base._queue_category  # pylint: disable=protected-access
