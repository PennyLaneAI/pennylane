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
import warnings
from typing import Union

import pennylane as qml
from pennylane.operation import Operator
from pennylane.pauli import PauliSentence, PauliWord


def center(
    g: list[Union[Operator, PauliWord, PauliSentence]], pauli: bool = False
) -> list[Union[Operator, PauliSentence]]:
    r"""
    A function to compute the center of a Lie algebra.

    .. warning:: :func:`~center` has moved to the :mod:`pennylane.liealg` module and can be called from there via ``qml.liealg.center`` or from the top level via ``qml.center``.

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

    .. seealso:: :func:`~lie_closure`, :func:`~structure_constants`, :class:`~pennylane.pauli.PauliVSpace`, :doc:`Introduction to Dynamical Lie Algebras for quantum practitioners <demos/tutorial_liealgebra>`

    **Example**

    We can compute the center of a DLA ``g``. For that, we compute the DLA via :func:`~lie_closure`.

    >>> generators = [qml.X(0), qml.X(0) @ qml.X(1), qml.Y(1)]
    >>> g = qml.lie_closure(generators)

    The ``center`` is then the collection of operators that commute with `all` other operators in the DLA.
    In this case, just ``X(0)``.

    >>> qml.center(g)
    [X(0)]

    .. details::
        :title: Derivation
        :href: derivation

        The center :math:`\mathfrak{\xi}(\mathfrak{g})` of an algebra :math:`\mathfrak{g}`
        can be computed in the following steps. First, compute the
        :func:`~.pennylane.structure_constants`, or adjoint representation, of the algebra
        with respect to some basis :math:`\mathbb{B}` of :math:`\mathfrak{g}`.
        The center of :math:`\mathfrak{g}` is then given by

        .. math::

            \mathfrak{\xi}(\mathfrak{g}) = \operatorname{span}\left\{\bigcap_{x\in\mathbb{B}}
            \operatorname{ker}(\operatorname{ad}_x)\right\},

        i.e., the intersection of the kernels, or null spaces, of all basis elements in the
        adjoint representation.

        The kernel can be computed with ``scipy.linalg.null_space``, and vector space
        intersections are computed recursively from pairwise intersections. The intersection
        between two vectors spaces :math:`V_1` and :math:`V_2` given by (orthonormal) bases
        :math:`\mathbb{B}_i` can be computed from the kernel of the matrix that has all
        basis vectors from :math:`\mathbb{B}_1` and :math:`-\mathbb{B}_2` as columns, i.e.,
        :math:`\operatorname{ker}(\left[\ \mathbb{B}_1 \ | -\mathbb{B}_2\ \right])`. For an
        (orthonormal) basis of this kernel, consisting of two stacked column vectors
        :math:`u^{(i)}_1` and :math:`u^{(i)}_2` for each basis, a basis of the
        intersection space :math:`V_1 \cap V_2` is given by :math:`\{\mathbb{B}_1 u_1^{(i)}\}_i`
        (or equivalently by :math:`\{\mathbb{B}_2 u_2^{(i)}\}_i`).
        Also see `this post <https://math.stackexchange.com/questions/25371/how-to-find-a-basis-for-the-intersection-of-two-vector-spaces-in-mathbbrn>`_
        for details.

        If the input consists of :class:`~.pennylane.PauliWord` instances only, we can
        instead compute pairwise commutators and know that the center consists solely of
        basis elements that commute with all other basis elements. This can be seen in the
        following way.

        Assume that the center elements identified based on the basis have been removed
        already and we are left with a basis :math:`\mathbb{B}=\{p_i\}_i` of Pauli
        words such that :math:`\forall i\ \exists j:\ [p_i, p_j] \neq 0`. Assume that there is
        another center element :math:`x\neq 0`, which was missed before because it is a linear
        combination of Pauli words:

        .. math::

            \forall j: \ [x, p_j] = [\sum_i x_i p_i, p_j] = 0.

        As products of Paulis are unique when fixing one of the factors (:math:`p_j` is fixed
        above), we then know that

        .. math::

            &\forall j: \ 0 = \sum_i x_i [p_i, p_j] = 2 \sum_i x_i \chi_{i,j} p_ip_j\\
            \Rightarrow &\forall i,j \text{ s.t. } \chi_{i,j}\neq 0: x_i = 0,

        where :math:`\chi_{i,j}` denotes an indicator that is :math:`0` if the commutator
        :math:`[p_i, p_j]` vanishes and :math:`1` otherwise.
        However, we know that for each :math:`i` there is at least one :math:`j` such that
        :math:`\chi_{i,j}\neq 0`. This means that :math:`x_i = 0` is guaranteed for all
        :math:`i` by at least one :math:`j`. Therefore :math:`x=0`, which is a contradiction
        to our initial assumption that :math:`x\neq 0`.
    """
    warnings.warn(
        "Calling center via qml.pauli.center is deprecated. center has moved to pennylane.liealg. "
        "Please call center from top level as qml.center or from the liealg module via qml.liealg.center.",
        qml.PennyLaneDeprecationWarning,
    )
    return qml.center(g, pauli)
