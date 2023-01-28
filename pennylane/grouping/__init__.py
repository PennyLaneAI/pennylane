# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
This subpackage defines functions and classes for Pauli-word partitioning
functionality used in measurement optimization.
"""
from warnings import warn


from . import graph_colouring  # pylint:disable=wrong-import-position
from .group_observables import (  # pylint:disable=wrong-import-position
    group_observables,
    PauliGroupingStrategy,
)
from .optimize_measurements import optimize_measurements  # pylint:disable=wrong-import-position
from .transformations import (  # pylint:disable=wrong-import-position
    qwc_rotation,
    diagonalize_pauli_word,
    diagonalize_qwc_pauli_words,
    diagonalize_qwc_groupings,
)
from .utils import (  # pylint:disable=wrong-import-position
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
)
from .pauli import (  # pylint:disable=wrong-import-position
    pauli_group,
    pauli_mult,
    pauli_mult_with_phase,
    partition_pauli_group,
)
