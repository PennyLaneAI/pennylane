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
"""
Contains the hamiltonian tape transform
"""
import itertools
import pennylane as qml


def expand_hamiltonian(tape):
    """Expands a tape that ends in a collection of Hamiltonians"""

    combined_obs = []
    measurements = []

    for m in tape.measurements:

        if isinstance(m.obs, qml.Hamiltonian):
            combined_obs.extend(m.obs.ops)
            measurements.extend([m.return_type for _ in range(len(m.obs.ops))])

        else:
            combined_obs.append(m.obs)
            measurements.append(m.return_type)

    tapes = []

    # Generates the partitioned sets of observables
    new_obs, numbering = qml.grouping.group_observables(
        combined_obs, list(range(len(combined_obs))), grouping_type="commuting"
    )

    merge_numbering = list(itertools.chain.from_iterable(numbering))

    # Generates the new set of tapes
    for num, u in zip(numbering, new_obs):
        new_tape = qml.tape.QuantumTape()
        with new_tape:
            [o.queue() for o in tape.operations]
            [qml.tape.MeasurementProcess(measurements[c], obs=i).queue() for c, i in zip(num, u)]

        tapes.append(new_tape)

    # Generates an order set of identifiers for each observable
    new_obs_id = [frozenset(combined_obs[i]._obs_data()) for i in merge_numbering]


    def processing_fn(results):

        lookup = dict(zip(new_obs_id, list(results)))

        # Reconstructs each of the observables to be returned
        res = []
        for o in tape.observables:
            if isinstance(o, qml.Hamiltonian):
                res.append(sum([c * lookup[frozenset(op._obs_data())] for c, op in zip(o.coeffs, o.ops)]))
            else:
                res.append(lookup[frozenset(o._obs_data())])

        return res

    return tapes, processing_fn
