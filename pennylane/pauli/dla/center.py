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
"""A function to compute the center of a Lie algebra"""
from typing import Union

import numpy as np
from scipy.linalg import null_space

from pennylane.operation import Operator
from pennylane.pauli import PauliSentence, PauliWord
from pennylane.pauli.dla import structure_constants


def center(
    g: list[Union[Operator, PauliWord, PauliSentence]], pauli: bool = False
) -> list[Union[Operator, PauliSentence]]:
    r"""
    A function to compute the center of a Lie algebra.

    Given a Lie algebra :math:`\mathfrak{g} = \{h_1,.., h_d\}`, the center :math:`\mathfrak{\xi}(\mathfrak{g})`
    is given by all elements in :math:`\mathfrak{g}` that commute with `all` other elements in :math:`\mathfrak{g}`,

    .. math:: \mathfrak{\xi}(\mathfrak{g}) := \{h \in \mathfrak{g} | [h, h_i]=0 \ \forall h_i \in \mathfrak{g} \}

    Args:
        g (List[Union[Operator, PauliSentence, PauliWord]]): List of operators that spans
            the algebra for which to find the center.
        pauli (bool): Indicates whether it is assumed that :class:`~.PauliSentence` or
            :class:`~.PauliWord` instances are input and returned. This can help with performance
            to avoid unnecessary conversions to :class:`~pennylane.operation.Operator`
            and vice versa. Default is ``False``.

    Returns:
        List[Union[Operator, PauliSentence]]: Center of ``g``

    .. seealso:: :func:`~lie_closure`, :func:`~structure_constants`, :class:`~pennylane.pauli.PauliVSpace`, `Demo: Introduction to Dynamical Lie Algebras for quantum practitioners <https://pennylane.ai/qml/demos/tutorial_liealgebra/>`__

    **Example**

    We can compute the center of a DLA ``g``. For that, we compute the DLA via :func:`~lie_closure`.

    >>> generators = [qml.X(0), qml.X(0) @ qml.X(1), qml.Y(1)]
    >>> g = qml.lie_closure(generators)

    The ``center`` is then the collection of operators that commute with `all` other operators in the DLA.
    In this case, just ``X(0)``.

    >>> qml.center(g)
    [-0.9999999999999999 * X(0)]

    """
    if len(g) < 2:
        # A length-zero list has zero center, a length-one list has full center
        return g

    adjoint_repr = structure_constants(g, pauli)
    # Compute kernel for adjoint rep of each DLA element
    kernels = [null_space(ad_x) for ad_x in adjoint_repr]
    # If any kernels vanishes, their overlap vanishes as well
    if any(k.shape[1] == 0 for k in kernels):
        return []
    # Compute the complements of the kernels
    supports = [null_space(k.T) for k in kernels]
    # Combine all complements
    combined_support = np.hstack(supports)
    if combined_support.shape[1] == 0:
        return g
    # Compute the complement of the combined complements: It is the intersection of the kernels
    center_coefficients = null_space(combined_support.T)

    # Construct operators from numerical output and convert to desired format
    res = [sum(c * x for c, x in zip(c_coeffs, g)) for c_coeffs in center_coefficients.T]

    have_paulis = isinstance(g[0], (PauliWord, PauliSentence))
    if pauli or have_paulis:
        _ = [el.simplify() for el in res]
        if not pauli:
            res = [el.operation() for el in res]
    else:
        res = [el.simplify() for el in res]

    return res
