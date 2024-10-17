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
"""Functionality for Cartan decomposition"""
from functools import partial, singledispatch
from typing import Union

import numpy as np

import pennylane as qml
from pennylane import QubitUnitary, Y
from pennylane.operation import Operator
from pennylane.pauli import PauliSentence

from .dense_util import apply_basis_change, check_cartan_decomp
from .involutions import AI, AII, AIII, BDI, CI, CII, DIII, int_log2


def cartan_decomposition(g, involution):
    r"""Cartan Decomposition :math:`\mathfrak{g} = \mathfrak{k} \osum \mathfrak{m}`.

    Given a Lie algebra :math:`\mathfrak{g}`, the Cartan decomposition is a decomposition
    :math:`\mathfrak{g} = \mathfrak{k} \osum \mathfrak{m}` into orthogonal complements.
    This is realized by an involution :math:`\Theta(g)` that maps each operator :math:`g \in \mathfrak{g}`
    back to itself after two consecutive applications, i.e., :math:`\Theta(\Theta(g)) = g \forall g \in \mathfrak{g}`.

    The ``involution`` argument can be any function that maps the operators in the provided ``g`` to a boolean output.
    ``True`` for operators that go into :math:`\mathfrak{k}` and ``False`` for operators in :math:`\mathfrak{m}`.

    The resulting subspaces fulfill the Cartan commutation relations

    .. math:: [\mathfrak{k}, \mathfrak{k}] \subseteq \mathfrak{k} \text{ ; } [\mathfrak{k}, \mathfrak{m}] \subseteq \mathfrak{m} \text{ ; } [\mathfrak{m}, \mathfrak{m}] \subseteq \mathfrak{k}

    Args:
        g (List[Union[PauliSentence, Operator]]): the (dynamical) Lie algebra to decompose
        involution (callable): Involution function :math:`\Theta(\cdot)` to act on the input operator, should return ``0/1`` or ``True/False``.
            E.g., :func:`~even_odd_involution` or :func:`~concurrence_involution`.

    Returns:
        k (List[Union[PauliSentence, Operator]]): the even parity subspace :math:`\Theta(\mathfrak{k}) = \mathfrak{k}`
        m (List[Union[PauliSentence, Operator]]): the odd parity subspace :math:`\Theta(\mathfrak{m}) = \mathfrak{m}`

    .. seealso:: :func:`~even_odd_involution`, :func:`~concurrence_involution`
    """
    # simple implementation assuming all elements in g are already either in k and m
    # TODO: Figure out more general way to do this when the above is not the case
    m = []
    k = []

    for op in g:
        if involution(op):  # odd parity theta(k) = k
            k.append(op)
        else:  # even parity theta(m) = -m
            m.append(op)

    return k, m


# dispatch to different input types
def even_odd_involution(op: Union[PauliSentence, np.ndarray, Operator]):
    r"""The Even-Odd involution

    This is defined in `quant-ph/0701193 <https://arxiv.org/pdf/quant-ph/0701193>`__, and for Pauli words and sentences comes down to counting Pauli-Y operators.

    Args:
        op ( Union[PauliSentence, np.ndarray, Operator]): Input operator

    Returns:
        bool: Boolean output ``True`` or ``False`` for odd (:math:`\mathfrak{k}`) and even parity subspace (:math:`\mathfrak{m}`), respectively

    .. seealso:: :func:`~cartan_decomposition`
    """
    return _even_odd_involution(op)


@singledispatch
def _even_odd_involution(op):  # pylint:disable=unused-argument
    return NotImplementedError(f"Involution not defined for operator {op} of type {type(op)}")


@_even_odd_involution.register(PauliSentence)
def _even_odd_involution_ps(op: PauliSentence):
    # Generalization to sums of Paulis: check each term and assert they all have the same parity
    parity = []
    for pw in op.keys():
        parity.append(len(pw) % 2)

    # only makes sense if parity is the same for all terms, e.g. Heisenberg model
    assert all(
        parity[0] == p for p in parity
    ), f"The Even-Odd involution is not well-defined for operator {op} as individual terms have different parity"
    return parity[0]


@_even_odd_involution.register(np.ndarray)
def _even_odd_involution_matrix(op: np.ndarray):
    """see Table CI in https://arxiv.org/abs/2406.04418"""
    n = int(np.round(np.log2(op.shape[-1])))
    YYY = qml.prod(*[Y(i) for i in range(n)])
    YYY = qml.matrix(YYY, range(n))

    transformed = YYY @ op.conj() @ YYY
    return not np.allclose(transformed, op)


@_even_odd_involution.register(Operator)
def _even_odd_involution_op(op: Operator):
    """use pauli representation"""
    return _even_odd_involution_ps(op.pauli_rep)


# dispatch to different input types
def concurrence_involution(op: Union[PauliSentence, np.ndarray, Operator]):
    r"""The Concurrence Canonical Decomposition :math:`\Theta(g) = -g^T` as a Cartan involution function

    This is defined in `quant-ph/0701193 <https://arxiv.org/pdf/quant-ph/0701193>`__, and for Pauli words and sentences comes down to counting Pauli-Y operators.

    Args:
        op ( Union[PauliSentence, np.ndarray, Operator]): Input operator

    Returns:
        bool: Boolean output ``True`` or ``False`` for odd (:math:`\mathfrak{k}`) and even parity subspace (:math:`\mathfrak{m}`), respectively

    .. seealso:: :func:`~cartan_decomposition`

    """
    return _concurrence_involution(op)


@singledispatch
def _concurrence_involution(op):
    return NotImplementedError(f"Involution not defined for operator {op} of type {type(op)}")


@_concurrence_involution.register(PauliSentence)
def _concurrence_involution_pauli(op: PauliSentence):
    # Generalization to sums of Paulis: check each term and assert they all have the same parity
    parity = []
    for pw in op.keys():
        result = sum(1 if el == "Y" else 0 for el in pw.values())
        parity.append(result % 2)

    # only makes sense if parity is the same for all terms, e.g. Heisenberg model
    assert all(
        parity[0] == p for p in parity
    ), f"The concurrence canonical decomposition is not well-defined for operator {op} as individual terms have different parity"
    return bool(parity[0])


@_concurrence_involution.register(Operator)
def _concurrence_involution_operation(op: Operator):
    op = op.matrix()
    return np.allclose(op, -op.T)


@_concurrence_involution.register(np.ndarray)
def _concurrence_involution_matrix(op: np.ndarray):
    return np.allclose(op, -op.T)


IDENTITY = object()


def pauli_y_eigenbasis(wire, num_wires):
    V = np.array([[1, 1], [1j, -1j]]) / np.sqrt(2)
    return QubitUnitary(V, wire).matrix(wire_order=range(num_wires))


def _not_implemented_yet(wire, num_wires, pair):
    raise NotImplementedError(
        f"The pair {pair} is a valid pair of involutions conceptually, but the basis change between them has not been implemented yet."
    )


_basis_change_constructors = {
    ("AI", "BDI"): IDENTITY,
    ("AI", "DIII"): IDENTITY,
    ("AII", "CI"): IDENTITY,
    ("AII", "CII"): IDENTITY,
    ("AIII", "ClassB"): IDENTITY,
    ("BDI", "ClassB"): IDENTITY,
    ("CI", "AI"): pauli_y_eigenbasis,
    ("CI", "AII"): pauli_y_eigenbasis,
    ("CI", "AIII"): IDENTITY,
    ("CII", "ClassB"): IDENTITY,
    ("DIII", "AI"): pauli_y_eigenbasis,
    ("DIII", "AII"): pauli_y_eigenbasis,
    ("DIII", "AIII"): pauli_y_eigenbasis,
    ("ClassB", "AI"): IDENTITY,
    ("ClassB", "AII"): IDENTITY,
    ("ClassB", "AIII"): IDENTITY,
    ("ClassB", "BDI"): IDENTITY,
    ("ClassB", "DIII"): IDENTITY,
    ("ClassB", "CI"): IDENTITY,
    ("ClassB", "CII"): IDENTITY,
}


def _check_classb_sequence(before, after):
    if before == "AIII" and after.startswith("A"):
        return
    if before == "BDI" and after in ("BDI", "DIII"):
        return
    if before == "CII" and after.startswith("C"):
        return
    raise ValueError(
        f"The 3-sequence ({before}, ClassB, {after}) of involutions is not a valid sequence."
    )


def recursive_cartan_decomposition(g, chain, validate=True, verbose=True):
    """Apply a recursive Cartan decomposition specified by a chain of decomposition types.
    The decompositions will use canonical involutions and hardcoded basis transformations
    between them in order to obtain a valid recursion."""

    # Prerun the validation by obtaining the required basis changes and raising an error if
    # an invalid pair is found.
    basis_changes = []
    names = [getattr(phi, "func", phi).__name__ for phi in chain]

    # Assume some standard behaviour regarding the wires on which we need to perform basis changes
    wire = 0
    num_wires = int_log2(np.shape(g)[-1])
    for i, name in enumerate(names[:-1]):
        invol_pair = (name, names[i + 1])
        if invol_pair not in _basis_change_constructors:
            raise ValueError(
                f"The specified chain contains the pair {'-->'.join(invol_pair)}, "
                "which is not a valid pair."
            )
        # Run specific check for sequence of three involutions where ClassB is the middle one
        if name == "ClassB" and i > 0:
            _check_classb_sequence(names[i - 1], names[i + 1])
        bc_constructor = _basis_change_constructors[invol_pair]
        if bc_constructor is IDENTITY:
            bc = bc_constructor
        else:
            bc = bc_constructor(wire, num_wires)
            # Next assumption: The wire is only incremented if a basis change is applied.
            wire += 1
        basis_changes.append(bc)

    basis_changes.append(IDENTITY)  # Do not perform any basis change after last involution.

    decompositions = {}
    for i, (phi, bc) in enumerate(zip(chain, basis_changes)):
        k, m = cartan_decomposition(g, phi)
        if validate:
            check_cartan_decomp(k, m, verbose=verbose)
        name = getattr(phi, "func", phi).__name__
        if verbose:
            print(f"Iteration {i}: {len(g):>4} -{name:-^10}> {len(k):>4},{len(m):>4}")
        decompositions[i] = (k, m)
        if bc is not IDENTITY:
            k = apply_basis_change(bc, k)
            m = apply_basis_change(bc, m)
        g = k

    return decompositions
