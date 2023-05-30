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
"""module docstring"""

from collections import defaultdict
from itertools import combinations

import rustworkx as rx
import numpy as onp

import pennylane as qml

from .batch_transform import batch_transform


def _commutes(pw1: qml.pauli.PauliWord, pw2: qml.pauli.PauliWord) -> bool:
    return all(pw2[key] in {value, "I"} for key, value in pw1.items())


# pylint: disable=protected-access, no-member
@batch_transform
def measurement_shot_reuse(tape):
    """Expand the circuit into a batch that samples in all the required bases. Postprocess the returned
    samples into the expectation value using all available samples.

    """
    if len(tape.measurements) > 1 or not isinstance(tape[-1], qml.measurements.ExpectationMP):
        raise NotImplementedError
    # we now have a single expectation value to consider

    obs = tape[-1].obs
    wire_order = obs.wires
    obs_pauli_rep = qml.pauli.pauli_sentence(obs)

    empty_pw = qml.pauli.PauliWord({})
    offset = obs_pauli_rep.pop(empty_pw) if empty_pw in obs_pauli_rep else 0

    g = rx.PyGraph()
    word_to_node = {}
    node_to_word = {}
    for w in obs_pauli_rep:
        node = g.add_node(w)
        word_to_node[w] = node
        node_to_word[node] = w

    for w1, w2 in combinations(obs_pauli_rep, 2):
        if not _commutes(w1, w2):
            g.add_edge(word_to_node[w1], word_to_node[w2], None)

    partitions = rx.graph_greedy_color(g)
    # partitions is map from pauli word to integers

    # map from integers to corresponding pauli word as dictionary
    bases_dict = defaultdict(dict)
    for word, color in partitions.items():
        bases_dict[color].update(node_to_word[word])

    # make sure the bases are on all wires
    # could optimize the choice on unoccupied wire, but just using Z is easy for now
    for num, basis in bases_dict.items():
        if len(basis) != len(wire_order):
            bases_dict[num] = qml.pauli.PauliWord(
                {w: (basis[w] if w in basis else "Z") for w in wire_order}
            )

    # convert to just the pauli words for each basis
    bases = tuple(qml.pauli.PauliWord(val) for val in bases_dict.values())

    num_partitions = len(bases)

    mp_in_partitions = [[] for _ in bases]
    for word, coeff in obs_pauli_rep.items():
        word_bases_indexes = [i for i, basis in enumerate(bases) if _commutes(word, basis)]
        num_partitions = len(word_bases_indexes)
        partial_coeff = coeff / num_partitions

        for i in word_bases_indexes:
            mp = qml.expval(qml.s_prod(partial_coeff, word.operation()))
            mp_in_partitions[i].append(mp)

    batch = []
    for measurement_set in mp_in_partitions:
        batch.append(qml.tape.QuantumScript(tape._ops, measurement_set, tape._prep))

    def post_processing_fn(results):
        """Post processing function for shot_efficient_hamiltonian_expand that processes samples
        back into an observable expectation value.

        Relies upon from outer namespace:

        * `offset`
        """
        return sum(onp.sum(r) for r in results) + offset

    return batch, post_processing_fn
