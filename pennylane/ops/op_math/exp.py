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
This submodule defines the symbolic operation that stands for an exponential of an operator.
"""
from functools import lru_cache
from warnings import warn

import numpy as np
from scipy.sparse.linalg import expm as sparse_expm

import pennylane as qml
from pennylane import math, queuing
from pennylane.decomposition import (
    add_decomps,
    register_condition,
    register_resources,
    resource_rep,
)
from pennylane.exceptions import (
    DecompositionUndefinedError,
    GeneratorUndefinedError,
    OperatorPropertyUndefined,
    PennyLaneDeprecationWarning,
    TermsUndefinedError,
)
from pennylane.operation import Operation, Operator
from pennylane.wires import Wires

from .linear_combination import LinearCombination
from .sprod import SProd
from .sum import Sum
from .symbolicop import ScalarSymbolicOp


@lru_cache
def _get_has_generator_types(num_wires):
    # Store operator classes with generators
    has_generator_types = []
    any_wires_types = []
    for op_name in qml.ops.qubit.__all__:
        op_class = getattr(qml.ops.qubit, op_name)
        if op_class not in {qml.PauliRot, qml.PCPhase} and op_class.has_generator:
            if op_class.num_wires == num_wires:
                has_generator_types.append(op_class)
            elif op_class.num_wires is None:
                any_wires_types.append(op_class)

    # prioritize types with that exact number of wires over types with any number of wires
    # ie choose RZ before MultiRZ
    return has_generator_types + any_wires_types


def _find_equal_generator(base, coeff):
    for op_class in _get_has_generator_types(len(base.wires)):
        g, c = qml.generator(op_class)(coeff, base.wires)
        # Some generators are not wire-ordered (e.g. OrbitalRotation)
        mapped_wires_g = qml.map_wires(g, dict(zip(g.wires, base.wires)))

        if qml.equal(mapped_wires_g, base):
            # Cancel the coefficients added by the generator
            coeff = math.real(-1j / c * coeff)
            return op_class(coeff, g.wires)

        # could have absorbed the coefficient.
        simplified_g = qml.simplify(qml.s_prod(c, mapped_wires_g))

        if qml.equal(simplified_g, base):
            coeff = math.real(-1j * coeff)
            return op_class(coeff, g.wires)

    return None


def exp(op, coeff=1, num_steps=None, id=None):
    """Take the exponential of an Operator times a coefficient.

    Args:
        base (~.operation.Operator): The Operator to be exponentiated
        coeff (float): a scalar coefficient of the operator
        num_steps (int): The number of steps used in the decomposition of the exponential operator,
            also known as the Trotter number. If this value is `None` and the Suzuki-Trotter
            decomposition is needed, an error will be raised.
        id (str): id for the Exp operator. Default is None.

    Returns:
       :class:`Exp`: An :class:`~.operation.Operator` representing an operator exponential.

    .. warning::

        Providing ``num_steps`` to ``qml.exp`` is deprecated and will be removed in a future release.
        Instead, use :class:`~.TrotterProduct` for approximate methods, providing the ``n`` parameter to perform the
        Suzuki-Trotter product approximation of a Hamiltonian with the specified number of Trotter steps.

    .. note::

        This operator supports a batched base, a batched coefficient and a combination of both:

        >>> op = qml.exp(qml.RX([1, 2, 3], wires=0), coeff=4)
        >>> qml.matrix(op).shape
        (3, 2, 2)
        >>> op = qml.exp(qml.RX(1, wires=0), coeff=[1, 2, 3])
        >>> qml.matrix(op).shape
        (3, 2, 2)
        >>> op = qml.exp(qml.RX([1, 2, 3], wires=0), coeff=[4, 5, 6])
        >>> qml.matrix(op).shape
        (3, 2, 2)

        But it doesn't support batching of operators:

        >>> qml.exp([qml.RX(1, wires=0), qml.RX(2, wires=0)], coeff=4)
        Traceback (most recent call last):
            ...
        TypeError: base is expected to be of type Operator, but received <class 'list'>

    **Example**

    This symbolic operator can be used to make general rotation operators:

    >>> x = np.array(1.23)
    >>> op = qml.exp(qml.X(0), -0.5j * x)
    >>> qml.math.allclose(op.matrix(), qml.RX(x, wires=0).matrix())
    True

    This can even be used for more complicated generators:

    >>> t = qml.X(0) @ qml.X(1) + qml.Y(0) @ qml.Y(1)
    >>> isingxy = qml.exp(t, 0.25j * x)
    >>> qml.math.allclose(isingxy.matrix(), qml.IsingXY(x, wires=(0,1)).matrix())
    True

    If the coefficient is purely imaginary and the base operator is Hermitian, then
    the gate can be used in a circuit, though it may not be supported by the device and
    may not be differentiable.

    >>> @qml.qnode(qml.device('default.qubit', wires=1))
    ... def circuit(x):
    ...     qml.exp(qml.X(0), -0.5j * x)
    ...     return qml.expval(qml.Z(0))
    >>> print(qml.draw(circuit)(1.23))
    0: ──Exp(0.00-0.61j X)─┤  <Z>

    If the base operator is Hermitian and the coefficient is real, then the ``Exp`` operator
    can be measured as an observable:

    >>> obs = qml.exp(qml.Z(0), 3)
    >>> @qml.qnode(qml.device('default.qubit', wires=1))
    ... def circuit():
    ...     return qml.expval(obs)
    >>> print(circuit())
    20.085...


    """
    return Exp(op, coeff, num_steps=num_steps, id=id)


class Exp(ScalarSymbolicOp, Operation):
    """A symbolic operator representing the exponential of a operator.

    Args:
        base (~.operation.Operator): The Operator to be exponentiated
        coeff=1 (Number): A scalar coefficient of the operator.
        num_steps (int): The number of steps used in the decomposition of the exponential operator,
            also known as the Trotter number. If this value is `None` and the Suzuki-Trotter
            decomposition is needed, an error will be raised.
        id (str): id for the Exp operator. Default is None.

    .. warning::

        Providing ``num_steps`` to ``Exp`` is deprecated and will be removed in a future release.
        Instead, use :class:`~.TrotterProduct` for approximate methods, providing the ``n`` parameter to perform the
        Suzuki-Trotter product approximation of a Hamiltonian with the specified number of Trotter steps.

    **Example**

    This symbolic operator can be used to make general rotation operators:

    >>> x = np.array(1.23)
    >>> op = Exp( qml.X(0), -0.5j * x)
    >>> qml.math.allclose(op.matrix(), qml.RX(x, wires=0).matrix())
    True

    This can even be used for more complicated generators:

    >>> t = qml.X(0) @ qml.X(1) + qml.Y(0) @ qml.Y(1)
    >>> isingxy = Exp(t, 0.25j * x)
    >>> qml.math.allclose(isingxy.matrix(), qml.IsingXY(x, wires=(0,1)).matrix())
    True

    If the coefficient is purely imaginary and the base operator is Hermitian, then
    the gate can be used in a circuit, though it may not be supported by the device and
    may not be differentiable.

    >>> @qml.qnode(qml.device('default.qubit', wires=1))
    ... def circuit(x):
    ...     Exp(qml.X(0), -0.5j * x)
    ...     return qml.expval(qml.Z(0))
    >>> print(qml.draw(circuit)(1.23))
    0: ──Exp(0.00-0.61j X)─┤  <Z>

    If the base operator is Hermitian and the coefficient is real, then the ``Exp`` operator
    can be measured as an observable:

    >>> obs = Exp(qml.Z(0), 3)
    >>> @qml.qnode(qml.device('default.qubit', wires=1))
    ... def circuit():
    ...     return qml.expval(obs)
    >>> print(circuit())
    20.085...

    """

    control_wires = Wires([])
    _name = "Exp"

    resource_keys = {"base", "num_steps"}

    def _flatten(self):
        return (self.base, self.data[0]), (self.num_steps,)

    @classmethod
    def _unflatten(cls, data, metadata):
        return cls(data[0], data[1], num_steps=metadata[0])

    def __init__(self, base, coeff=1, num_steps=None, id=None):
        if num_steps is not None:
            warn(
                "Providing 'num_steps' to 'qml.evolve' and 'qml.exp' is deprecated and will be "
                "removed in a future release. Instead, use 'qml.TrotterProduct' for approximate "
                "methods, providing the 'n' parameter to perform the Suzuki-Trotter product "
                "approximation of a Hamiltonian with the specified number of Trotter steps.",
                PennyLaneDeprecationWarning,
            )

        if not isinstance(base, Operator):
            raise TypeError(f"base is expected to be of type Operator, but received {type(base)}")
        super().__init__(base, scalar=coeff, id=id)
        self.grad_recipe = [None]
        self.num_steps = num_steps
        self.hyperparameters["num_steps"] = num_steps

    def __repr__(self):
        return (
            f"Exp({self.coeff} {self.base})"
            if self.base.arithmetic_depth > 0
            else f"Exp({self.coeff} {self.base.name})"
        )

    @property
    def hash(self):
        return hash((str(self.name), self.base.hash, str(self.coeff)))

    @property
    def coeff(self):
        """The numerical coefficient of the operator in the exponent."""
        return self.scalar

    @property
    def num_params(self):
        return self.base.num_params + 1

    @property
    def is_hermitian(self):
        return self.base.is_hermitian and math.allequal(math.imag(self.coeff), 0)

    @property
    def _queue_category(self):
        return "_ops"

    @property
    def resource_params(self) -> dict:
        return {
            "base": self.base,
            "num_steps": self.num_steps,
        }

    # pylint: disable=invalid-overridden-method, arguments-renamed
    @property
    @queuing.QueuingManager.stop_recording()
    def has_decomposition(self):
        base = self.base.simplify()
        coeff = self.coeff
        if isinstance(base, SProd):
            coeff *= base.scalar
            base = base.base

        if self.num_steps is not None and isinstance(base, Sum) and base.is_hermitian:
            return True

        # pylint: disable=unidiomatic-typecheck
        if type(self) is Exp and not math.is_abstract(coeff) and math.real(coeff):
            # if type is Evolution, we assume that is is indeed time evolution
            return False

        if qml.pauli.is_pauli_word(base):
            return True

        op = _find_equal_generator(base, coeff)
        return op is not None

    def decomposition(self):
        r"""Representation of the operator as a product of other operators. Decomposes into
        :class:`~.PauliRot` if the coefficient is imaginary and the base is a Pauli Word.

        .. math:: O = O_1 O_2 \dots O_n

        A ``DecompositionUndefinedError`` is raised if the coefficient is not imaginary or the base
        is not a Pauli Word.

        Returns:
            list[PauliRot]: decomposition of the operator
        """
        with queuing.QueuingManager.stop_recording():
            base = self.base.simplify()  # for things like products of scalar products

            # preferably, this should be added to LinearCombination.simplify
            if isinstance(base, LinearCombination) and len(base) == 1:
                _c, _o = base.terms()
                base = SProd(_c[0], _o[0])
            d = self._recursive_decomposition(base, self.coeff)

        if queuing.QueuingManager.recording():
            for op in d:
                queuing.apply(op)

        return d

    def _recursive_decomposition(self, base: Operator, coeff: complex):
        """Decompose the exponential of ``base`` multiplied by ``coeff``.

        Args:
            base (Operator): exponentiated operator
            coeff (complex): coefficient multiplying the exponentiated operator

        Returns:
            List[Operator]: decomposition
        """
        if isinstance(base, SProd):
            return self._recursive_decomposition(base.base, base.scalar * coeff)

        if self.num_steps is not None and isinstance(base, Sum):
            # Apply trotter decomposition
            coeffs, ops = base.terms()
            coeffs = [c * coeff for c in coeffs]
            return self._trotter_decomposition(ops, coeffs)

        # pylint: disable=unidiomatic-typecheck
        if type(self) is Exp and not math.is_abstract(coeff) and math.real(coeff):

            error_msg = f"The decomposition of the {self} operator is not defined."

            if not self.num_steps:  # if num_steps was not set
                error_msg += (
                    " Please set a value to ``num_steps`` when instantiating the ``Exp`` operator "
                    "if a Suzuki-Trotter decomposition is required."
                )

            if self.base.is_hermitian:
                error_msg += (
                    " Decomposition is not defined for real coefficients of hermitian operators."
                )

            raise DecompositionUndefinedError(error_msg)

        return self._smart_decomposition(coeff, base)

    def _smart_decomposition(self, coeff, base):
        """Decompose to an operator with a generator or a PauliRot if possible."""

        op = _find_equal_generator(base, coeff)
        if op is not None:
            return [op]

        if qml.pauli.is_pauli_word(base):
            # Check if the exponential can be decomposed into a PauliRot gate
            return self._pauli_rot_decomposition(base, coeff)

        error_msg = f"The decomposition of the {self} operator is not defined."

        if not self.num_steps:  # if num_steps was not set
            error_msg += (
                " Please set a value to ``num_steps`` when instantiating the ``Exp`` operator "
                "if a Suzuki-Trotter decomposition is required. "
            )

        raise DecompositionUndefinedError(error_msg)

    @staticmethod
    def _pauli_rot_decomposition(base: Operator, coeff: complex):
        """Decomposes the exponential of a Pauli word into a PauliRot.

        Args:
            base (Operator): exponentiated operator
            coeff (complex): coefficient multiplying the exponentiated operator

        Returns:
            List[Operator]: list containing the PauliRot operator
        """
        # Cancel the coefficients added by PauliRot and Ising gates
        coeff = (
            math.real(2j * coeff)  # jax has no real_if_close
            if math.get_interface(coeff) == "jax"
            else math.real_if_close(2j * coeff)  # only cast to real if close
        )
        pauli_word = qml.pauli.pauli_word_to_string(base)
        return [qml.PauliRot(theta=coeff, pauli_word=pauli_word, wires=base.wires)]

    def _trotter_decomposition(self, ops: list[Operator], coeffs: list[complex]):
        """Uses the Suzuki-Trotter approximation to decompose the exponential of the linear
        combination of ``coeffs`` and ``ops``.

        Args:
            ops (List[Operator]): list of operators of the linear combination
            coeffs (List[complex]): list of coefficients of the linear combination

        Raises:
            ValueError: if the Trotter number (``num_steps``) is not defined
            DecompositionUndefinedError: if the linear combination contains operators that are not
                Pauli words

        Returns:
            List[Operator]: a list of operators containing the decomposition
        """
        op_list = []
        for c, op in zip(coeffs, ops):
            c /= self.num_steps  # divide by trotter number
            op_list.extend(self._recursive_decomposition(op, c))

        return op_list * self.num_steps  # apply operators ``num_steps`` times

    def matrix(self, wire_order=None):
        coeff_interface = math.get_interface(self.scalar)
        if coeff_interface == "autograd" and math.requires_grad(self.scalar):
            # math.expm is not differentiable with autograd
            # So we try to do a differentiable construction if possible
            #
            # This won't catch situations when the base matrix is autograd,
            # but at least this provides as much trainablility as possible
            try:
                eigvals = self.eigvals()
                eigvals_mat = (
                    math.stack([math.diag(e) for e in eigvals])
                    if math.ndim(self.scalar) > 0
                    else math.diag(eigvals)
                )
                if len(self.diagonalizing_gates()) == 0:
                    return math.expand_matrix(eigvals_mat, wires=self.wires, wire_order=wire_order)
                diagonalizing_mat = qml.matrix(self.diagonalizing_gates, wire_order=self.wires)()
                mat = diagonalizing_mat.conj().T @ eigvals_mat @ diagonalizing_mat
                return math.expand_matrix(mat, wires=self.wires, wire_order=wire_order)
            except OperatorPropertyUndefined:
                warn(
                    f"The autograd matrix for {self} is not differentiable. "
                    "Use a different interface if you need backpropagation.",
                    UserWarning,
                )
        return super().matrix(wire_order=wire_order)

    @staticmethod
    def _matrix(scalar, mat):
        return math.expm(scalar * mat)

    def sparse_matrix(self, wire_order=None, format="csr"):
        if wire_order is not None:
            raise NotImplementedError("Wire order is not implemented for sparse_matrix")

        return sparse_expm(self.coeff * self.base.sparse_matrix().tocsc()).asformat(format)

    # pylint: disable=arguments-renamed,invalid-overridden-method
    @property
    def has_diagonalizing_gates(self):
        return self.base.has_diagonalizing_gates

    def diagonalizing_gates(self):
        return self.base.diagonalizing_gates()

    def eigvals(self):
        r"""Eigenvalues of the operator in the computational basis.

        .. math::

            c \mathbf{M} \mathbf{v} = c \lambda \mathbf{v}
            \quad \Longrightarrow \quad
            e^{c \mathbf{M}} \mathbf{v} = e^{c \lambda} \mathbf{v}

        >>> obs = Exp(qml.X(0), 3)
        >>> qml.eigvals(obs)
        array([20.08...,  0.049...])
        >>> np.exp(3 * qml.eigvals(qml.X(0)))
        array([20.08...,  0.049...])

        """
        base_eigvals = math.convert_like(self.base.eigvals(), self.coeff)
        base_eigvals = math.cast_like(base_eigvals, self.coeff)
        if math.ndim(self.scalar) > 0:
            # exp coeff is broadcasted
            return math.stack([math.exp(c * base_eigvals) for c in self.coeff])
        return math.exp(self.coeff * base_eigvals)

    def label(self, decimals=None, base_label=None, cache=None):
        coeff = (
            self.coeff if decimals is None else format(math.toarray(self.coeff), f".{decimals}f")
        )
        return base_label or f"Exp({coeff} {self.base.label(decimals=decimals, cache=cache)})"

    def pow(self, z):
        return Exp(self.base, self.coeff * z)

    def simplify(self):
        new_base = self.base.simplify()
        if isinstance(new_base, qml.ops.op_math.SProd):
            return Exp(new_base.base, self.coeff * new_base.scalar, self.num_steps)
        return Exp(new_base, self.coeff, self.num_steps)

    # pylint: disable=arguments-renamed, invalid-overridden-method
    @property
    def has_generator(self):
        if math.is_abstract(self.coeff):
            return self.base.is_hermitian
        return self.base.is_hermitian and not np.real(self.coeff)

    def generator(self):
        r"""Generator of an operator that is in single-parameter-form.

        For example, for operator

        .. math::

            U(\phi) = e^{i\phi (0.5 Y + Z\otimes X)}

        we get the generator

        >>> U = qml.ops.op_math.Evolution(0.5 * qml.Y(0) + qml.Z(0) @ qml.X(1), 1)
        >>> print(U)
        Evolution(-1j 0.5 * Y(0) + Z(0) @ X(1))
        >>> U.generator()
        -1 * (0.5 * Y(0) + Z(0) @ X(1))

        """
        if self.has_generator:
            return self.base

        raise GeneratorUndefinedError(
            f"Exponential with coefficient {self.coeff} and base operator {self.base} does not appear to have a "
            f"generator. Consider using op.simplify() to simplify before finding the generator, or define the operator "
            f"in the form exp(-ixG) through the Evolution class."
        )


def _pauli_rot_decomp_condition(base, num_steps):
    if num_steps:
        # If num_steps is explicitly provided, always use the Trotter decomposition
        return False
    with queuing.QueuingManager.stop_recording():
        base = base.simplify()
    # The PauliRot decomposition is only applicable when the base is a Pauli word
    return qml.pauli.is_pauli_word(base)


def _pauli_rot_decomp_resource(base, **_):
    with queuing.QueuingManager.stop_recording():
        base = base.simplify()
    return {resource_rep(qml.PauliRot, pauli_word=qml.pauli.pauli_word_to_string(base)): 1}


def _trotter_decomp_condition(base, num_steps):
    if not num_steps:
        return False
    with queuing.QueuingManager.stop_recording():
        base = base.simplify()
    if qml.pauli.is_pauli_word(base):
        return True
    try:
        _, ops = base.terms()
        return all(qml.pauli.is_pauli_word(ob) for ob in ops)
    except TermsUndefinedError:
        return False


def _trotter_decomp_resource(base, num_steps):

    with queuing.QueuingManager.stop_recording():
        base = base.simplify()

    try:
        _, ops = base.terms()
    except TermsUndefinedError:
        ops = [base]  # The condition should've already verified that this is a valid pauli word.

    gate_count = {}
    for op in ops:
        pauli_word = qml.pauli.pauli_word_to_string(op)
        if not all(p == "I" for p in pauli_word):
            op_rep = resource_rep(qml.PauliRot, pauli_word=pauli_word)
            gate_count[op_rep] = gate_count.get(op_rep, 0) + num_steps
    return gate_count


@register_condition(_pauli_rot_decomp_condition)
@register_resources(_pauli_rot_decomp_resource)
def pauli_rot_decomp(*params, wires, base, **_):  # pylint: disable=unused-argument
    """Decompose the operator into a single PauliRot operator."""
    with queuing.QueuingManager.stop_recording():
        base = base.simplify()
    coeff = params[0]
    if isinstance(base, qml.ops.SProd):
        coeff, base = params[0] * base.scalar, base.base
    coeff = 2j * coeff  # The 2j cancels the coefficient added by PauliRot
    pauli_word = qml.pauli.pauli_word_to_string(base)
    if not all(p == "I" for p in pauli_word):
        qml.PauliRot(coeff, pauli_word, base.wires)


@register_condition(_trotter_decomp_condition)
@register_resources(_trotter_decomp_resource)
def trotter_decomp(*params, wires, base, num_steps):  # pylint: disable=unused-argument
    """Uses the Suzuki-Trotter approximation to decompose the operator exponential."""

    with queuing.QueuingManager.stop_recording():
        simplified_base = base.simplify()

    try:
        coeffs, ops = simplified_base.terms()
    except TermsUndefinedError:
        # The condition should've already verified that this is a valid pauli word.
        coeffs, ops = [1.0], [simplified_base]

    new_coeffs, pauli_words, new_wires = [], [], []
    for c, op in zip(coeffs, ops):
        # The 2j cancels the coefficient added by PauliRot
        c = c * params[0] / num_steps * 2j
        pauli_word = qml.pauli.pauli_word_to_string(op)
        if pauli_word != "I" * len(op.wires):
            new_coeffs.append(c)
            pauli_words.append(pauli_word)
            new_wires.append(op.wires)

    for _ in range(num_steps):
        for c, pauli_word, new_wire in zip(new_coeffs, pauli_words, new_wires):
            qml.PauliRot(c, pauli_word, new_wire)


add_decomps(Exp, trotter_decomp, pauli_rot_decomp)
