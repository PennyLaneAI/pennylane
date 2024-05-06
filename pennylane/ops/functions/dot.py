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
This file contains the definition of the dot function, which computes the dot product between
a vector and a list of operators.
"""
# pylint: disable=too-many-branches
from collections import defaultdict
from typing import Sequence, Union, Callable

import pennylane as qml
from pennylane.operation import Operator, convert_to_opmath
from pennylane.pulse import ParametrizedHamiltonian
from pennylane.pauli import PauliWord, PauliSentence


def dot(
    coeffs: Sequence[Union[float, Callable]],
    ops: Sequence[Union[Operator, PauliWord, PauliSentence]],
    pauli=False,
    grouping_type=None,
    method="rlf",
) -> Union[Operator, ParametrizedHamiltonian, PauliSentence]:
    r"""Returns the dot product between the ``coeffs`` vector and the ``ops`` list of operators.

    This function returns the following linear combination: :math:`\sum_{k} c_k O_k`, where
    :math:`c_k` and :math:`O_k` are the elements inside the ``coeffs`` and ``ops`` arguments, respectively.

    Args:
        coeffs (Sequence[float, Callable]): sequence containing the coefficients of the linear combination
        ops (Sequence[Operator, PauliWord, PauliSentence]): sequence containing the operators of the linear combination.
           Can also be ``PauliWord`` or ``PauliSentence`` instances.
        pauli (bool, optional): If ``True``, a :class:`~.PauliSentence`
            operator is used to represent the linear combination. If False, a :class:`Sum` operator
            is returned. Defaults to ``False``. Note that when ``ops`` consists solely of ``PauliWord``
            and ``PauliSentence`` instances, the function still returns a PennyLane operator when ``pauli=False``.
        grouping_type (str): The type of binary relation between Pauli words used to compute
            the grouping. Can be ``'qwc'``, ``'commuting'``, or ``'anticommuting'``. Note that if
            ``pauli=True``, the grouping will be ignored.
        method (str): The graph coloring heuristic to use in solving minimum clique cover for
            grouping, which can be ``'lf'`` (Largest First) or ``'rlf'`` (Recursive Largest
            First). This keyword argument is ignored if ``grouping_type`` is ``None``.

    Raises:
        ValueError: if the number of coefficients and operators does not match or if they are empty

    Returns:
        Operator or ParametrizedHamiltonian: operator describing the linear combination

    .. note::

        If grouping is requested, the computed groupings are stored as a list of list of indices
        in ``Sum.grouping_indices``. The indices refer to the operators and coefficients returned
        by ``Sum.terms()``, not ``Sum.operands``, as these are not guaranteed to be equivalent.

    **Example**

    >>> coeffs = np.array([1.1, 2.2])
    >>> ops = [qml.X(0), qml.Y(0)]
    >>> qml.dot(coeffs, ops)
    1.1 * X(0) + 2.2 * Y(0)
    >>> qml.dot(coeffs, ops, pauli=True)
    1.1 * X(0)
    + 2.2 * Y(0)

    Note that additions of the same operator are not executed by default.

    >>> qml.dot([1., 1.], [qml.X(0), qml.X(0)])
    X(0) + X(0)

    You can obtain a cleaner version by simplifying the resulting expression.

    >>> qml.dot([1., 1.], [qml.X(0), qml.X(0)]).simplify()
    2.0 * X(0)

    ``pauli=True`` can be used to construct a more efficient, simplified version of the operator.
    Note that it returns a :class:`~.PauliSentence`, which is not an :class:`~.Operator`. This
    specialized representation can be converted to an operator:

    >>> qml.dot([1, 2], [qml.X(0), qml.X(0)], pauli=True).operation()
    3.0 * X(0)

    Using ``pauli=True`` and then converting the result to an :class:`~.Operator` is much faster
    than using ``pauli=False``, but it only works for pauli words
    (see :func:`~.is_pauli_word`).

    If any of the parameters listed in ``coeffs`` are callables, the resulting dot product will be a
    :class:`~.ParametrizedHamiltonian`:

    >>> coeffs = [lambda p, t: p * jnp.sin(t) for _ in range(2)]
    >>> ops = [qml.X(0), qml.Y(0)]
    >>> qml.dot(coeffs, ops)
    (
        <lambda>(params_0, t) * X(0)
      + <lambda>(params_1, t) * Y(0)
    )

    .. details::
        :title: Grouping

        Grouping information can be collected during construction using the ``grouping_type`` and ``method``
        keyword arguments. For example:

        .. code-block:: python

            import pennylane as qml

            a = qml.X(0)
            b = qml.prod(qml.X(0), qml.X(1))
            c = qml.Z(0)
            obs = [a, b, c]
            coeffs = [1.0, 2.0, 3.0]

            op = qml.dot(coeffs, obs, grouping_type="qwc")

        >>> op.grouping_indices
        ((2,), (0, 1))

        ``grouping_type`` can be ``"qwc"`` (qubit-wise commuting), ``"commuting"``, or ``"anticommuting"``, and
        ``method`` can be ``"rlf"`` or ``"lf"``. To see more details about how these affect grouping, see
        :ref:`Pauli Graph Colouring<graph_colouring>` and :func:`~pennylane.pauli.group_observables`.
    """

    for t in (Operator, PauliWord, PauliSentence):
        if isinstance(ops, t):
            raise ValueError(
                f"ops must be an Iterable of {t.__name__}'s, not a {t.__name__} itself."
            )

    if len(coeffs) != len(ops):
        raise ValueError("Number of coefficients and operators does not match.")
    if len(coeffs) == 0 and len(ops) == 0:
        raise ValueError("Cannot compute the dot product of an empty sequence.")

    if any(callable(c) for c in coeffs):
        return ParametrizedHamiltonian(coeffs, ops)

    # User-specified Pauli route
    if pauli:
        if all(isinstance(pauli, (PauliWord, PauliSentence)) for pauli in ops):
            # Use pauli arithmetic when ops are just PauliWord and PauliSentence instances
            return _dot_pure_paulis(coeffs, ops)

        # Else, transform all ops to pauli sentences
        return _dot_with_ops_and_paulis(coeffs, ops)

    # Convert possible PauliWord and PauliSentence instances to operation
    ops = [op.operation() if isinstance(op, (PauliWord, PauliSentence)) else op for op in ops]

    # When casting a Hamiltonian to a Sum, we also cast its inner Tensors to Prods
    ops = (convert_to_opmath(op) for op in ops)

    operands = [op if coeff == 1 else qml.s_prod(coeff, op) for coeff, op in zip(coeffs, ops)]
    return (
        operands[0]
        if len(operands) == 1
        else qml.sum(*operands, grouping_type=grouping_type, method=method)
    )


def _dot_with_ops_and_paulis(coeffs: Sequence[float], ops: Sequence[Operator]):
    """Compute dot when operators are a mix of pennylane operators, PauliWord and PauliSentence by turning them all into a PauliSentence instance.
    Returns a PauliSentence instance"""
    pauli_words = defaultdict(lambda: 0)
    for coeff, op in zip(coeffs, ops):
        sentence = qml.pauli.pauli_sentence(op)
        for pw in sentence:
            pauli_words[pw] += sentence[pw] * coeff

    return qml.pauli.PauliSentence(pauli_words)


def _dot_pure_paulis(coeffs: Sequence[float], ops: Sequence[Union[PauliWord, PauliSentence]]):
    """Faster computation of dot when all ops are PauliSentences or PauliWords"""
    return sum((c * op for c, op in zip(coeffs[1:], ops[1:])), start=coeffs[0] * ops[0])
