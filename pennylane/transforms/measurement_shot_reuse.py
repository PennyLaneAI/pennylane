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

    sample_measurement = qml.sample(wires=wire_order)
    batch = []
    for basis in bases:
        basis_op = basis.operation()
        circuit = qml.tape.QuantumScript(
            tape._ops + basis_op.diagonalizing_gates(), [sample_measurement], tape._prep
        )
        batch.append(circuit)

    def post_processing_fn(results):
        """Post processing function for shot_efficient_hamiltonian_expand that processes samples
        back into an observable expectation value.

        Relies upon from outer namespace:

        * `obs_pauli_rep`
        * `bases`
        """
        accumulated_sum = 0

        for term, coeff in obs_pauli_rep.items():

            shots_for_measurement = None
            for sample_basis, samples in zip(bases, results):
                if _commutes(sample_basis, term):
                    if shots_for_measurement is None:
                        shots_for_measurement = samples
                    else:
                        shots_for_measurement = onp.vstack((shots_for_measurement, samples))

            if len(term) > 0:
                term_m = qml.expval(term.operation())
                term_m_expval = term_m.process_samples(shots_for_measurement, wire_order)
                accumulated_sum += coeff * term_m_expval
            else:
                accumulated_sum += coeff

        return accumulated_sum

    return batch, post_processing_fn
