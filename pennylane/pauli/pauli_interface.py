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
Utility functions to interact with and extract information from Pauli words and Pauli sentences.
"""
from typing import Union
from functools import singledispatch

from pennylane.ops import (
    Hamiltonian,
    LinearCombination,
    Identity,
    PauliX,
    PauliY,
    PauliZ,
    Prod,
    SProd,
)
from pennylane.operation import Tensor

from .utils import is_pauli_word
from .conversion import pauli_sentence


def pauli_word_prefactor(observable):
    """If the operator provided is a valid Pauli word (i.e a single term which may be a tensor product
    of pauli operators), then this function extracts the prefactor.

    Args:
        observable (~.Operator): the operator to be examined

    Returns:
        Union[int, float, complex]: The scaling/phase coefficient of the Pauli word.

    Raises:
        ValueError: If an operator is provided that is not a valid Pauli word.

    **Example**

    >>> pauli_word_prefactor(qml.Identity(0))
    1
    >>> pauli_word_prefactor(qml.X(0) @ qml.Y(1))
    1
    >>> pauli_word_prefactor(qml.X(0) @ qml.Y(0))
    1j
    """
    return _pauli_word_prefactor(observable)


@singledispatch
def _pauli_word_prefactor(observable):
    """Private wrapper function for pauli_word_prefactor."""
    raise ValueError(f"Expected a valid Pauli word, got {observable}")


@_pauli_word_prefactor.register(PauliX)
@_pauli_word_prefactor.register(PauliY)
@_pauli_word_prefactor.register(PauliZ)
@_pauli_word_prefactor.register(Identity)
def _pw_prefactor_pauli(
    observable: Union[PauliX, PauliY, PauliZ, Identity]
):  # pylint:disable=unused-argument
    return 1


@_pauli_word_prefactor.register
def _pw_prefactor_tensor(observable: Tensor):
    if is_pauli_word(observable):
        return list(pauli_sentence(observable).values())[0]  # only one term,
    raise ValueError(f"Expected a valid Pauli word, got {observable}")


@_pauli_word_prefactor.register(Hamiltonian)
@_pauli_word_prefactor.register(LinearCombination)
def _pw_prefactor_ham(observable: Union[Hamiltonian, LinearCombination]):
    if is_pauli_word(observable):
        return observable.coeffs[0]
    raise ValueError(f"Expected a valid Pauli word, got {observable}")


@_pauli_word_prefactor.register(Prod)
@_pauli_word_prefactor.register(SProd)
def _pw_prefactor_prod_sprod(observable: Union[Prod, SProd]):
    ps = observable.pauli_rep
    if ps is not None and len(ps) == 1:
        return list(ps.values())[0]

    raise ValueError(f"Expected a valid Pauli word, got {observable}")
