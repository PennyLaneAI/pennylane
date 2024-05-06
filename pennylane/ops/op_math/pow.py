# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
import copy
from typing import Union

from scipy.linalg import fractional_matrix_power

import pennylane as qml
from pennylane import math as qmlmath
from pennylane.operation import (
    DecompositionUndefinedError,
    Observable,
    Operation,
    PowUndefinedError,
    SparseMatrixUndefinedError,
)
from pennylane.ops.identity import Identity
from pennylane.queuing import QueuingManager, apply

from .symbolicop import ScalarSymbolicOp

_superscript = str.maketrans("0123456789.+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⋅⁺⁻")


def pow(base, z=1, lazy=True, id=None):
    """Raise an Operator to a power.

    Args:
        base (~.operation.Operator): the operator to be raised to a power
        z (float): the exponent (default value is 1)

    Keyword Args:
        lazy=True (bool): In lazy mode, all operations are wrapped in a ``Pow`` class
            and handled later. If ``lazy=False``, operation-specific simplifications are first attempted.
        id (str): custom label given to an operator instance,
            can be useful for some applications where the instance has to be identified

    Returns:
        Operator

    .. note::

        This operator supports a batched base, a batched coefficient and a combination of both:

        >>> op = qml.pow(qml.RX([1, 2, 3], wires=0), z=4)
        >>> qml.matrix(op).shape
        (3, 2, 2)
        >>> op = qml.pow(qml.RX(1, wires=0), z=[1, 2, 3])
        >>> qml.matrix(op).shape
        (3, 2, 2)
        >>> op = qml.pow(qml.RX([1, 2, 3], wires=0), z=[4, 5, 6])
        >>> qml.matrix(op).shape
        (3, 2, 2)

        But it doesn't support batching of operators:

        >>> op = qml.pow([qml.RX(1, wires=0), qml.RX(2, wires=0)], z=4)
        AttributeError: 'list' object has no attribute 'name'

    .. seealso:: :class:`~.Pow`, :meth:`~.Operator.pow`.

    **Example**

    >>> qml.pow(qml.X(0), 0.5)
    X(0)**0.5
    >>> qml.pow(qml.X(0), 0.5, lazy=False)
    SX(wires=[0])
    >>> qml.pow(qml.X(0), 0.1, lazy=False)
    X(0)**0.1
    >>> qml.pow(qml.X(0), 2, lazy=False)
    I(0)

    Lazy behaviour can also be accessed via ``op ** z``.

    """
    if lazy:
        return Pow(base, z, id=id)
    try:
        pow_ops = base.pow(z)
    except PowUndefinedError:
        return Pow(base, z, id=id)

    num_ops = len(pow_ops)
    if num_ops == 0:
        pow_op = qml.Identity(base.wires, id=id)
    elif num_ops == 1:
        pow_op = pow_ops[0]
    else:
        pow_op = qml.prod(*pow_ops)
    QueuingManager.remove(base)

    return pow_op


class Pow(ScalarSymbolicOp):
    """Symbolic operator denoting an operator raised to a power.

    Args:
        base (~.operation.Operator): the operator to be raised to a power
        z=1 (float): the exponent

    **Example**

    >>> sqrt_x = Pow(qml.X(0), 0.5)
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

    def _flatten(self):
        return (self.base, self.z), tuple()

    @classmethod
    def _unflatten(cls, data, _):
        return pow(data[0], z=data[1])

    # pylint: disable=unused-argument
    def __new__(cls, base=None, z=1, id=None):
        """Mixes in parents based on inheritance structure of base.

        Though all the types will be named "Pow", their *identity* and location in memory will be
        different based on ``base``'s inheritance.  We cache the different types in private class
        variables so that:

        >>> Pow(op, z).__class__ is Pow(op, z).__class__
        True
        >>> type(Pow(op, z)) == type(Pow(op, z))
        True
        >>> isinstance(Pow(op, z), type(Pow(op, z)))
        True
        >>> Pow(qml.RX(1.2, wires=0), 0.5).__class__ is Pow._operation_type
        True
        >>> Pow(qml.X(0), 1.2).__class__ is Pow._operation_observable_type
        True

        """

        if isinstance(base, Operation):
            if isinstance(base, Observable):
                return object.__new__(PowOpObs)

            # not an observable
            return object.__new__(PowOperation)

        if isinstance(base, Observable):
            return object.__new__(PowObs)

        return object.__new__(Pow)

    def __init__(self, base=None, z=1, id=None):
        self.hyperparameters["z"] = z
        self._name = f"{base.name}**{z}"

        super().__init__(base, scalar=z, id=id)

        if isinstance(self.z, int) and self.z > 0:
            if (base_pauli_rep := getattr(self.base, "pauli_rep", None)) and (
                self.batch_size is None
            ):
                pr = base_pauli_rep
                for _ in range(self.z - 1):
                    pr = pr @ base_pauli_rep
                self._pauli_rep = pr
            else:
                self._pauli_rep = None
        else:
            self._pauli_rep = None

    def __repr__(self):
        return (
            f"({self.base})**{self.z}"
            if self.base.arithmetic_depth > 0
            else f"{self.base}**{self.z}"
        )

    @property
    def z(self):
        """The exponent."""
        return self.hyperparameters["z"]

    @property
    def ndim_params(self):
        return self.base.ndim_params

    @property
    def data(self):
        """The trainable parameters"""
        return self.base.data

    @data.setter
    def data(self, new_data):
        self.base.data = new_data

    def label(self, decimals=None, base_label=None, cache=None):
        z_string = format(self.z).translate(_superscript)
        base_label = self.base.label(decimals, base_label, cache=cache)
        return (
            f"({base_label}){z_string}" if self.base.arithmetic_depth > 0 else base_label + z_string
        )

    @staticmethod
    def _matrix(scalar, mat):
        if isinstance(scalar, int):
            if qml.math.get_deep_interface(mat) != "tensorflow":
                return qmlmath.linalg.matrix_power(mat, scalar)

            # TensorFlow doesn't have a matrix_power func, and scipy.linalg.fractional_matrix_power
            # is not differentiable. So we use a custom implementation of matrix power for integer
            # exponents below.
            if scalar == 0:
                # Used instead of qml.math.eye for tracing derivatives
                return mat @ qmlmath.linalg.inv(mat)
            if scalar > 0:
                out = mat
            else:
                out = mat = qmlmath.linalg.inv(mat)
                scalar *= -1

            for _ in range(scalar - 1):
                out @= mat
            return out

        return fractional_matrix_power(mat, scalar)

    # pylint: disable=arguments-differ
    @staticmethod
    def compute_sparse_matrix(*params, base=None, z=0):
        if isinstance(z, int):
            base_matrix = base.compute_sparse_matrix(*params, **base.hyperparameters)
            return base_matrix**z
        raise SparseMatrixUndefinedError

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_decomposition(self):
        if isinstance(self.z, int) and self.z > 0:
            return True
        try:
            self.base.pow(self.z)
        except PowUndefinedError:
            return False
        return True

    def decomposition(self):
        try:
            return self.base.pow(self.z)
        except PowUndefinedError as e:
            if isinstance(self.z, int) and self.z > 0:
                if QueuingManager.recording():
                    return [apply(self.base) for _ in range(self.z)]
                return [copy.copy(self.base) for _ in range(self.z)]
            # TODO: consider: what if z is an int and less than 0?
            # do we want Pow(base, -1) to be a "more fundamental" op
            raise DecompositionUndefinedError from e

    @property
    def has_diagonalizing_gates(self):
        return self.base.has_diagonalizing_gates

    def diagonalizing_gates(self):
        r"""Sequence of gates that diagonalize the operator in the computational basis.

        Given the eigendecomposition :math:`O = U \Sigma U^{\dagger}` where
        :math:`\Sigma` is a diagonal matrix containing the eigenvalues,
        the sequence of diagonalizing gates implements the unitary :math:`U^{\dagger}`.

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

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_generator(self):
        return self.base.has_generator

    def generator(self):
        r"""Generator of an operator that is in single-parameter-form.

        The generator of a power operator is ``z`` times the generator of the
        base matrix.

        .. math::

            U(\phi)^z = e^{i\phi (z G)}

        See also :func:`~.generator`
        """
        return self.z * self.base.generator()

    def pow(self, z):
        return [Pow(base=self.base, z=self.z * z)]

    def adjoint(self):
        return Pow(base=qml.adjoint(self.base), z=self.z)

    def simplify(self) -> Union["Pow", Identity]:
        # try using pauli_rep:
        if pr := self.pauli_rep:
            pr.simplify()
            return pr.operation(wire_order=self.wires)

        base = self.base.simplify()
        try:
            ops = base.pow(z=self.z)
            if not ops:
                return qml.Identity(self.wires)
            op = qml.prod(*ops) if len(ops) > 1 else ops[0]
            return op.simplify()
        except PowUndefinedError:
            return Pow(base=base, z=self.z)


# pylint: disable=no-member
class PowOperation(Pow, Operation):
    """Operation-specific methods and properties for the ``Pow`` class.

    Dynamically mixed in based on the provided base operator.  If the base operator is an
    Operation, this class will be mixed in.

    When we no longer rely on certain functionality through `Operation`, we can get rid of this
    class.
    """

    def __new__(cls, *_, **__):
        return object.__new__(cls)

    # until we add gradient support
    grad_method = None

    @property
    def name(self):
        return self._name

    @property
    def control_wires(self):
        return self.base.control_wires


class PowObs(Pow, Observable):
    """A child class of ``Pow`` that also inherits from ``Observable``."""

    def __new__(cls, *_, **__):
        return object.__new__(cls)


# pylint: disable=too-many-ancestors
class PowOpObs(PowOperation, Observable):
    """A child class of ``Pow`` that inherits from both
    ``Observable`` and ``Operation``.
    """

    def __new__(cls, *_, **__):
        return object.__new__(cls)
