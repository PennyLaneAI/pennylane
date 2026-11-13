# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Detect Pauli operators and convert them to Pauli algebra."""

from functools import reduce, singledispatch

from pennylane.core.pauli import PauliSentence, PauliWord
from pennylane.ops import Identity, LinearCombination, PauliX, PauliY, PauliZ, Prod, SProd, Sum


def is_pauli_word(observable):
    """
    Checks if an observable instance consists only of Pauli and Identity Operators.

    A Pauli word can be either:

    * A single pauli operator (see :class:`~.PauliX` for an example).

    * A :class:`.Prod` instance containing Pauli operators.

    * A :class:`.SProd` instance containing a valid Pauli word.

    * A :class:`.Sum` instance with only one term.

    .. Warning::

        This function will only confirm that all operators are Pauli or Identity operators,
        and not whether the observable is mathematically a Pauli word.
        If an Observable consists of multiple Pauli operators targeting the same wire, the
        function will return ``True`` regardless of any complex coefficients.


    Args:
        observable (~.Operator): the operator to be examined

    Returns:
        bool: true if the input observable is a Pauli word, false otherwise.

    **Example**

    >>> is_pauli_word(qp.Identity(0))
    True
    >>> is_pauli_word(qp.X(0) @ qp.Z(2))
    True
    >>> is_pauli_word(qp.Z(0) @ qp.Hadamard(1))
    False
    >>> is_pauli_word(4 * qp.X(0) @ qp.Z(0))
    True
    """
    return _is_pauli_word(observable) or (len(observable.pauli_rep or []) == 1)


@singledispatch
def _is_pauli_word(observable):  # pylint: disable=unused-argument
    """
    Private implementation of is_pauli_word, to prevent all of the
    registered functions from appearing in the Sphinx docs.
    """
    return False


@_is_pauli_word.register(PauliX)
@_is_pauli_word.register(PauliY)
@_is_pauli_word.register(PauliZ)
@_is_pauli_word.register(Identity)
def _is_pw_pauli(
    observable: PauliX | PauliY | PauliZ | Identity,
):
    return True


@_is_pauli_word.register(Sum)
def _is_pw_ham(observable: Sum):
    ops = observable.terms()[1]
    return False if len(ops) != 1 else is_pauli_word(ops[0])


@_is_pauli_word.register
def _is_pw_prod(observable: Prod):
    return all(is_pauli_word(op) for op in observable)


@_is_pauli_word.register
def _is_pw_sprod(observable: SProd):
    return is_pauli_word(observable.base)


def pauli_sentence(op):
    """Return the PauliSentence representation of an arithmetic operator or Hamiltonian.

    Args:
        op (~.Operator): The operator or Hamiltonian that needs to be converted.

    Raises:
        ValueError: Op must be a linear combination of Pauli operators

    Returns:
        .PauliSentence: the PauliSentence representation of an arithmetic operator or Hamiltonian
    """

    if isinstance(op, PauliWord):
        return PauliSentence({op: 1.0})

    if isinstance(op, PauliSentence):
        return op

    if (ps := op.pauli_rep) is not None:
        return ps

    return _pauli_sentence(op)


def is_pauli_sentence(op):
    """Returns True of the operator is a PauliSentence and False otherwise."""
    if op.pauli_rep is not None:
        return True
    return False


@singledispatch
def _pauli_sentence(op):
    """Private function to dispatch"""
    raise ValueError(f"Op must be a linear combination of Pauli operators only, got: {op}")


@_pauli_sentence.register
def _(op: PauliX):
    return PauliSentence({PauliWord({op.wires[0]: "X"}): 1.0})


@_pauli_sentence.register
def _(op: PauliY):
    return PauliSentence({PauliWord({op.wires[0]: "Y"}): 1.0})


@_pauli_sentence.register
def _(op: PauliZ):
    return PauliSentence({PauliWord({op.wires[0]: "Z"}): 1.0})


@_pauli_sentence.register
def _(op: Identity):
    return PauliSentence({PauliWord({}): 1.0})


@_pauli_sentence.register
def _(op: Prod):
    factors = (_pauli_sentence(factor) for factor in op)
    return reduce(lambda a, b: a @ b, factors)


@_pauli_sentence.register
def _(op: SProd):
    ps = _pauli_sentence(op.base)
    for pw, coeff in ps.items():
        ps[pw] = coeff * op.scalar
    return ps


@_pauli_sentence.register(LinearCombination)
def _(op: LinearCombination):
    if not all(is_pauli_word(o) for o in op.ops):
        raise ValueError(f"Op must be a linear combination of Pauli operators only, got: {op}")

    return op._build_pauli_rep()  # pylint: disable=protected-access


@_pauli_sentence.register
def _(op: Sum):
    ps = PauliSentence()
    for term in op:
        ps += _pauli_sentence(term)
    return ps
