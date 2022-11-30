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
Utility functions to convert between ``~.PauliSentence`` and other PennyLane operators.
"""
from functools import reduce, singledispatch

from pennylane.operation import Tensor
from pennylane.ops import Hamiltonian, Identity, PauliX, PauliY, PauliZ, Prod, SProd, Sum

from .pauli_arithmetic import PauliWord, PauliSentence, X, Y, Z
from .utils import is_pauli_word


@singledispatch
def pauli_sentence(op):
    """Return the PauliSentence representation of an arithmetic operator or Hamiltonian.

    Args:
        op (~.Operator): The operator or Hamiltonian that needs to be converted.

    Raises:
        ValueError: Op must be a linear combination of Pauli operators

    Returns ~.PauliSentence
    """
    raise ValueError(f"Op must be a linear combination of Pauli operators only, got: {op}")


@pauli_sentence.register
def _(op: PauliX):
    return PauliSentence({PauliWord({op.wires[0]: X}): 1.0})


@pauli_sentence.register
def _(op: PauliY):
    return PauliSentence({PauliWord({op.wires[0]: Y}): 1.0})


@pauli_sentence.register
def _(op: PauliZ):
    return PauliSentence({PauliWord({op.wires[0]: Z}): 1.0})


@pauli_sentence.register
def _(op: Identity):  # pylint:disable=unused-argument
    return PauliSentence({PauliWord({}): 1.0})


@pauli_sentence.register
def _(op: Tensor):
    if not is_pauli_word(op):
        raise ValueError(f"Op must be a linear combination of Pauli operators only, got: {op}")

    factors = (pauli_sentence(factor) for factor in op.obs)
    return reduce(lambda a, b: a * b, factors)


@pauli_sentence.register
def _(op: Prod):
    factors = (pauli_sentence(factor) for factor in op)
    return reduce(lambda a, b: a * b, factors)


@pauli_sentence.register
def _(op: SProd):
    ps = pauli_sentence(op.base)
    for pw, coeff in ps.items():
        ps[pw] = coeff * op.scalar
    return ps


@pauli_sentence.register
def _(op: Hamiltonian):
    if not is_pauli_word(op):
        raise ValueError(f"Op must be a linear combination of Pauli operators only, got: {op}")

    summands = []
    for coeff, term in zip(*op.terms()):
        ps = pauli_sentence(term)
        for pw, sub_coeff in ps.items():
            ps[pw] = coeff * sub_coeff
        summands.append(ps)

    return reduce(lambda a, b: a + b, summands)


@pauli_sentence.register
def _(op: Sum):
    summands = (pauli_sentence(summand) for summand in op)
    return reduce(lambda a, b: a + b, summands)
