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

"""This module contains functions for preprocessing `QuantumScript`s to ensure
that they are supported for execution by a device."""

import pennylane as qml
from pennylane.tape import QuantumScript


def _stopping_condition(op):
    """Specify whether or not an Operator is supported"""
    return getattr(op, "has_matrix", False)


def expand_fn(circuit, dev, max_expansion=10):
    """Method for expanding or decomposing an input circuit. Can be the default or
    a custom expansion method, see :meth:`.Device.custom_expand` for more details.

    By default, this method expands the tape if:

    - nested tapes are present,
    - any operations are not supported on the device, or
    - multiple observables are measured on the same wire.

    Args:
        circuit (.QuantumTape): the circuit to expand.
        dev (.Device): the device to execute circuit(s) on.
        max_expansion (int): The number of times the circuit should be
            expanded. Expansion occurs when an operation or measurement is not
            supported, and results in a gate decomposition. If any operations
            in the decomposition remain unsupported by the device, another
            expansion occurs.

    Returns:
        .QuantumTape: The expanded/decomposed circuit, such that the device
        will natively support all operations.
    """
    # pylint: disable=protected-access

    if dev.custom_expand_fn is not None:
        return dev.custom_expand_fn(circuit, max_expansion=max_expansion)

    comp_basis_sampled_multi_measure = (
        len(circuit.measurements) > 1 and circuit.samples_computational_basis
    )
    obs_on_same_wire = len(circuit._obs_sharing_wires) > 0 or comp_basis_sampled_multi_measure
    obs_on_same_wire &= not any(isinstance(o, qml.Hamiltonian) for o in circuit._obs_sharing_wires)

    ops_not_supported = not all(_stopping_condition(op) for op in circuit.operations)

    if ops_not_supported or obs_on_same_wire:
        circuit = circuit.expand(depth=max_expansion, stop_at=_stopping_condition)

    return circuit


def batch_transform():
    pass


def check_validity():
    pass


def preprocess(tapes, dev, execution_config=None):  # Returns Sequence[QuantumScript], callable
    pass
