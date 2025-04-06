# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A function to compute the Lie closure of a set of operators"""
# pylint: disable=too-many-arguments
import warnings
from typing import Iterable, Union

import numpy as np

import pennylane as qml
from pennylane.operation import Operator
from pennylane.typing import TensorLike

from ..pauli_arithmetic import PauliSentence, PauliWord


def lie_closure(
    generators: Iterable[Union[PauliWord, PauliSentence, Operator, TensorLike]],
    *,  # force non-positional kwargs of the following
    max_iterations: int = 10000,
    verbose: bool = False,
    pauli: bool = False,
    matrix: bool = False,
    tol: float = None,
) -> Iterable[Union[PauliWord, PauliSentence, Operator, np.ndarray]]:
    r"""Compute the dynamical Lie algebra from a set of generators.

    .. warning:: :func:`~lie_closure` has moved to the :mod:`pennylane.liealg` module and can be called from there via ``qml.liealg.lie_closure`` or from the top level via ``qml.lie_closure``.

    The Lie closure, pronounced "Lee" closure, is a way to compute the so-called dynamical Lie algebra (DLA) of a set of generators :math:`\mathcal{G} = \{G_1, .. , G_N\}`.
    For such generators, one computes all nested commutators :math:`[G_i, [G_j, .., [G_k, G_\ell]]]` until no new operators are generated from commutation.
    All these operators together form the DLA, see e.g. section IIB of `arXiv:2308.01432 <https://arxiv.org/abs/2308.01432>`__.

    Args:
        generators (Iterable[Union[PauliWord, PauliSentence, Operator, TensorLike]]): generating set for which to compute the
            Lie closure.
        max_iterations (int): maximum depth of nested commutators to consider. Default is ``10000``.
        verbose (bool): whether to print out progress updates during Lie closure
            calculation. Default is ``False``.
        pauli (bool): Indicates whether it is assumed that :class:`~.PauliSentence` or :class:`~.PauliWord` instances are input and returned.
            This can help with performance to avoid unnecessary conversions to :class:`~pennylane.operation.Operator`
            and vice versa. Default is ``False``.
        matrix (bool): Whether or not matrix representations should be used and returned in the Lie closure computation. This can help
            speed up the computation when using sums of Paulis with many terms. Default is ``False``.
        tol (float): Numerical tolerance for the linear independence check used in :class:`~.PauliVSpace`.

    Returns:
        Union[list[:class:`~.PauliSentence`], list[:class:`~.Operator`], np.ndarray]: a basis of either :class:`~.PauliSentence`, :class:`~.Operator`, or ``np.ndarray`` instances that is closed under
        commutators (Lie closure).

    .. seealso:: :func:`~structure_constants`, :func:`~center`, :class:`~pennylane.pauli.PauliVSpace`, :doc:`Introduction to Dynamical Lie Algebras for quantum practitioners <demos/tutorial_liealgebra>`

    **Example**

    >>> ops = [X(0) @ X(1), Z(0), Z(1)]
    >>> dla = qml.lie_closure(ops)

    Let us walk through what happens in this simple example of computing the Lie closure of these generators (the transverse field Ising model on two qubits).
    A first round of commutators between all elements yields:

    >>> qml.commutator(X(0) @ X(1), Z(0))
    -2j * (Y(0) @ X(1))
    >>> qml.commutator(X(0) @ X(1), Z(1))
    -2j * (X(0) @ Y(1))

    A next round of commutators between all elements further yields the new operator ``Y(0) @ Y(1)``.

    >>> qml.commutator(X(0) @ Y(1), Z(0))
    -2j * (Y(0) @ Y(1))

    After that, no new operators emerge from taking nested commutators and we have the resulting DLA.
    This can be done in short via ``lie_closure`` as follows.

    >>> ops = [X(0) @ X(1), Z(0), Z(1)]
    >>> dla = qml.lie_closure(ops)
    >>> print(dla)
    [X(1) @ X(0),
     Z(0),
     Z(1),
     -1.0 * (Y(0) @ X(1)),
     -1.0 * (X(0) @ Y(1)),
     -1.0 * (Y(0) @ Y(1))]

    Note that we normalize by removing the factors of :math:`2i`, though minus signs are left intact.

    .. details::
        :title: Usage Details

        Note that by default, ``lie_closure`` returns PennyLane operators. Internally we use the more
        efficient representation in terms of :class:`~pennylane.pauli.PauliSentence` by making use of the ``op.pauli_rep``
        attribute of operators composed of Pauli operators. If desired, this format can be returned by using
        the keyword ``pauli=True``. In that case, the input is also assumed to be a :class:`~pennylane.pauli.PauliSentence` instance.

        >>> ops = [
        ...     PauliSentence({PauliWord({0: "X", 1: "X"}): 1.}),
        ...     PauliSentence({PauliWord({0: "Z"}): 1.}),
        ...     PauliSentence({PauliWord({1: "Z"}): 1.}),
        ... ]
        >>> dla = qml.lie_closure(ops, pauli=True)
        >>> print(dla)
        [1.0 * X(0) @ X(1),
         1.0 * Z(0),
         1.0 * Z(1),
         -1.0 * Y(0) @ X(1),
         -1.0 * X(0) @ Y(1),
         -1.0 * Y(0) @ Y(1)]
        >>> type(dla[0])
        pennylane.pauli.pauli_arithmetic.PauliSentence

        In the case of sums of Pauli operators with many terms, it is often faster to use the matrix representation of the operators rather than
        the semi-analytic :class:`~pennylane.pauli.PauliSentence` or :class:`~Operator` representation.
        We can force this by using the ``matrix`` keyword. The resulting ``dla`` is a ``np.ndarray`` of dimension ``(dim_g, 2**n, 2**n)``, where ``dim_g`` is the
        dimension of the DLA and ``n`` the number of qubits.

        >>> dla = qml.lie_closure(ops, matrix=True)
        >>> dla.shape
        (6, 4, 4)

        You can retrieve a semi-analytic representation again by using :func:`~pauli_decompose`.

        >>> dla_ops = [qml.pauli_decompose(op) for op in dla]

        Note that the results are only equivalent up to minus signs. This is okay because the sets of basis operators describe the same Lie algebra.

    """
    warnings.warn(
        "Calling lie_closure via qml.pauli.lie_closure is deprecated. lie_closure has moved to pennylane.liealg. "
        "Please call lie_closure from top level as qml.lie_closure or from the liealg module via qml.liealg.lie_closure.",
        qml.PennyLaneDeprecationWarning,
    )

    return qml.lie_closure(
        generators,
        max_iterations=max_iterations,
        verbose=verbose,
        pauli=pauli,
        matrix=matrix,
        tol=tol,
    )
