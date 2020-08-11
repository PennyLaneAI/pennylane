# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
This file contains built-in functions for constructing QAOA mixer Hamiltonians.
"""
import itertools
import numpy as np
import networkx as nx
import pennylane as qml
from pennylane.wires import Wires
from functools import reduce

from collections.abc import Iterable


def x_mixer(wires):
    r"""Creates a basic Pauli-X mixer Hamiltonian.

    This Hamiltonian is defined as:

    .. math:: H_M \ = \ \displaystyle\sum_{i} X_{i},

    where :math:`i` ranges over all wires, and :math:`X_i`
    denotes the Pauli-X operator on the :math:`i`-th wire.

    This is mixer is used in *A Quantum Approximate Optimization Algorithm*
    by Edward Farhi, Jeffrey Goldstone, Sam Gutmann [`arXiv:1411.4028 <https://arxiv.org/abs/1411.4028>`__].


    Args:
        wires (Iterable or Wires): The wires on which the Hamiltonian is applied

    Returns:
        Hamiltonian: Mixer Hamiltonian

    .. UsageDetails::

        The mixer Hamiltonian can be called as follows:

        .. code-block:: python

            from pennylane import qaoa

            wires = range(3)
            mixer_h = qaoa.x_mixer(wires)

        >>> print(mixer_h)
        (1.0) [X0] + (1.0) [X1] + (1.0) [X2]
    """

    wires = Wires(wires)

    coeffs = [1 for w in wires]
    obs = [qml.PauliX(w) for w in wires]

    return qml.Hamiltonian(coeffs, obs)


def xy_mixer(graph):
    r"""Creates a generalized SWAP/XY mixer Hamiltonian.

    This mixer Hamiltonian is defined as:

    .. math:: H_M \ = \ \frac{1}{2} \displaystyle\sum_{(i, j) \in E(G)} X_i X_j \ + \ Y_i Y_j,

    for some graph :math:`G`. :math:`X_i` and :math:`Y_i` denote the Pauli-X and Pauli-Y operators on the :math:`i`-th
    wire respectively.

    This mixer was introduced in *From the Quantum Approximate Optimization Algorithm
    to a Quantum Alternating Operator Ansatz* by Stuart Hadfield, Zhihui Wang, Bryan O'Gorman,
    Eleanor G. Rieffel, Davide Venturelli, and Rupak Biswas [`arXiv:1709.03489 <https://arxiv.org/abs/1709.03489>`__].

    Args:
        graph (nx.Graph): A graph defining the pairs of wires on which each term of the Hamiltonian acts.

    Returns:
         Hamiltonian: Mixer Hamiltonian

    .. UsageDetails::

        The mixer Hamiltonian can be called as follows:

        .. code-block:: python

            from pennylane import qaoa
            from networkx import Graph

            graph = Graph([(0, 1), (1, 2)])
            mixer_h = qaoa.xy_mixer(graph)

        >>> print(mixer_h)
        (0.5) [X0 X1] + (0.5) [Y0 Y1] + (0.5) [X1 X2] + (0.5) [Y1 Y2]
        """

    if not isinstance(graph, nx.Graph):
        raise ValueError(
            "Input graph must be a nx.Graph object, got {}".format(type(graph).__name__)
        )

    edges = graph.edges
    coeffs = 2 * [0.5 for e in edges]

    obs = []
    for node1, node2 in edges:
        obs.append(qml.PauliX(node1) @ qml.PauliX(node2))
        obs.append(qml.PauliY(node1) @ qml.PauliY(node2))

    return qml.Hamiltonian(coeffs, obs)


def _creation(wire):
    r"""Creates the spin-creation operator acting on a single wire.

    .. warning::

        This method ** does not** return a valid mixer Hamiltonian

    The creation operator is defined as:

    .. math:: S^{+} \ = \ |1\rangle \langle 0 | \ = \ \frac{1}{2} (X \ - \ iY)

    where :math:`X` and :math:`Y` are the Pauli-X and Pauli-Y. Note that this operator
    takes :math:`|0\rangle` to :math:`|1\rangle` and :math:`|1\rangle` to :math:`0`.

    Args:
        wire (Iterable or Wires): The wire on which the creation operator is acting

    Returns:
         (coeffs, terms)
    """

    return ([0.5, -0.5j], [qml.PauliX(wire), qml.PauliY(wire)])


def _annihilation(wire):
    r"""Creates the spin-annihilation operator acting on a single wire.

    .. warning::

        This method **does not** return a valid mixer Hamiltonian.

    The annihilation operator is defined as:

    .. math:: S^{-} \ = \ |0\rangle \langle 1 | \ = \ \frac{1}{2} (X \ + \ iY)

    where :math:`X` and :math:`Y` are the Pauli-X and Pauli-Y. Note that this operator
    takes :math:`|1\rangle` to :math:`|0\rangle` and :math:`|0\rangle` to :math:`0`.

    Args:
        wire (Iterable or Wires): The wire on which the annihilation operator is acting

    Returns:
        (coeffs, terms)
    """

    return ([0.5, 0.5j], [qml.PauliX(wire), qml.PauliY(wire)])


def _creation_annihilation_tensor(word, wires):
    r"""Creates an arbitrary tensor product of creation and annihilation operators.

    .. warning::

        This method **does not** return a valid mixer Hamiltonian.

    Given a string of the characters ``"+"``, ``"-"``, and ``"0"``, corresponding
    to the creation, annihilation, and identity operators respectively, this method takes
    a tensor product of each of these operations.

    Args:
        word (str): The string encoding creation, annihilation, and identity operators
        wires (Iterable or Wires): The wires on which the operators are defined

    Returns:
        (coeffs, terms)
    """

    coeff_list = None
    term_list = None

    for i, w in enumerate(word):
        if coeff_list is None:
            if w == "+":
                coeff_list, term_list = _creation(wires[i])
            elif w == "-":
                coeff_list, term_list = _annihilation(wires[i])
            elif w != "0":
                raise ValueError("Encountered invalid character {} in word".format(w))
        else:
            if w == "+":
                coeff, term = _creation(wires[i])
                coeff_list = itertools.product(coeff_list, coeff)
                term_list = itertools.product(term_list, term)
            elif w == "-":
                coeff, term = _annihilation(wires[i])
                coeff_list = itertools.product(coeff_list, coeff)
                term_list = itertools.product(term_list, term)
            elif w != "0":
                raise ValueError("Encountered invalid character {} in word".format(w))

    coeffs = [
        reduce(lambda x, y: x * y, term) if isinstance(term, Iterable) else term
        for term in list(coeff_list)
    ]
    terms = [
        qml.operation.Tensor(*term) if isinstance(term, Iterable) else term
        for term in list(term_list)
    ]

    return (coeffs, terms)


def permutation_mixer(words, coeffs, wires):
    r"""Takes a series of "creation-annihilation words", and converts them
    to a mixer Hamiltonian.

    In general, this Hamiltonian is defined as:

    .. math:: H_M \ = \ \displaystyle\sum_{j} c_{j} \displaystyle\prod_{i} \big( S_{i}^{-} \big)^{a_{ij}} \big( S_{i}^{+} \big)^{b_{ij}}

    where :math:`i` sums over all wires on which the Hamiltonian is defined. :math:`S^{-}_{j}` and :math:`S^{+}_{j}` are
    defined as the spin annihilation and creation operators applied on wire :math:`j`. The values :math:`a_{ij}`
    and :math:`b_{ij}` exclusively take on the values :math:`0` or :math:`1`.

    .. warning::

        An arbitrary linear combination of products of annihilation and creation operators is not necessarily
        Hermitian. If the user-supplied terms do not form a Hermitian operator, an error will be thrown.

    Args:
        words (list[str]): A list of "creation-annihilation words". See Usage Details for more information.
        coeffs (list): A list of coefficients on each creation-annihilation term.
        wires (Iterable or Wires): The wires on which the mixer Hamiltonian acts.

    Returns:
        Hamiltonian: Mixer Hamiltonian

    Raises:
        ValueError: if the supplied operator is not Hermitian.

    .. UsageDetails::

        **Creation-Annihilation Words**

        Creation-annihilation words are defined as strings of the characters: "+", "-", and "0".
        Each of these characters at the :math:`j`-th position within the word corresponds to a creation, annihilation,
        or identity operator acting on the :math:`j`-th wire, respectively.

        For example, the word: ``"+0-"` acting on wires ``[0, 1, 2]`` corresponds to the following operator:

        .. math:: S \ = \ S_{0}^{+} S_{2}^{-}

        **Defining the Hamiltonian**

        The mixer Hamiltonian can be called as follows:

        .. code-block:: python3

            from pennylane import qaoa

            mixer_h = qaoa.plus_minus_mixer(["+-", "-+"], [1, 1], wires=[0, 1])

        >>> print(mixer_h)
        (0.5) [X0 X1] + (0.5) [Y0 Y1]
    """

    H_coeffs, H_terms = _creation_annihilation_tensor(words[0], wires)
    H_coeffs = [i * coeffs[0] for i in H_coeffs]

    identifier = [(t.name, t.parameters, t.wires) for t in H_terms]

    for i, w in enumerate(words[1:]):
        co, ops = _creation_annihilation_tensor(w, wires)
        co = [c * coeffs[i] for c in co]

        for j, op in enumerate(ops):
            id = (op.name, op.parameters, op.wires)
            if id in identifier:
                H_coeffs[identifier.index(id)] += co[j]
            else:
                H_coeffs.append(co[j])
                H_terms.append(op)
                identifier.append(id)

    new_coeffs = []
    new_terms = []

    for i, num in enumerate(H_coeffs):
        if not np.allclose([num], [0]):
            new_coeffs.append(num)
            new_terms.append(H_terms[i])

    return qml.Hamiltonian(new_coeffs, new_terms)
