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
"""A module containing utility functions and reduced representation classes for working with Pauli operators. """

from .pauli_arithmetic import PauliWord, PauliSentence

from .utils import (
    is_pauli_word,
    are_identical_pauli_words,
    pauli_to_binary,
    binary_to_pauli,
    pauli_word_to_string,
    string_to_pauli_word,
    pauli_word_to_matrix,
    is_qwc,
    are_pauli_words_qwc,
    observables_to_binary_matrix,
    qwc_complement_adj_matrix,
    pauli_group,
    partition_pauli_group,
    qwc_rotation,
    diagonalize_pauli_word,
    diagonalize_qwc_pauli_words,
    diagonalize_qwc_groupings,
    simplify,
)

from .pauli_interface import pauli_word_prefactor

from .conversion import (
    pauli_decompose,
    pauli_sentence,
)

from .grouping import (
    group_observables,
    PauliGroupingStrategy,
    optimize_measurements,
    graph_colouring,
)

from .dla import PauliVSpace, lie_closure, structure_constants, center
