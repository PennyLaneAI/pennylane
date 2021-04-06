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
Contains the BasisStatePreparation template.
"""

import pennylane as qml
from pennylane.operation import Operation, AnyWires


class BasisStatePreparation(Operation):
    r"""
    Prepares a basis state on the given wires using a sequence of Pauli X gates.

    .. warning::

        ``basis_state`` influences the circuit architecture and is therefore incompatible with
        gradient computations. Ensure that ``basis_state`` is not passed to the qnode by positional
        arguments.

    Args:
        basis_state (array): Input array of shape ``(N,)``, where N is the number of wires
            the state preparation acts on. ``N`` must be smaller or equal to the total
            number of wires of the device.
        wires (Iterable or Wires): Wires that the template acts on. Accepts an iterable of numbers or strings, or
            a Wires object.

    Raises:
        ValueError: if inputs do not have the correct format
    """
    num_params = 1
    num_wires = AnyWires
    par_domain = "A"

    def __init__(self, basis_state, wires, do_queue=True):

        shape = qml.math.shape(basis_state)

        if len(shape) != 1:
            raise ValueError(f"Basis state must be one-dimensional; got shape {shape}.")

        n_bits = shape[0]
        if n_bits != len(wires):
            raise ValueError(f"Basis state must be of length {len(wires)}; got length {n_bits}.")

        # we can extract a list here, because embedding is not differentiable
        basis_state = list(qml.math.toarray(basis_state))

        if not all(bit in [0, 1] for bit in basis_state):
            raise ValueError(f"Basis state must only consist of 0s and 1s; got {basis_state}")

        super().__init__(basis_state, wires=wires, do_queue=do_queue)

    def expand(self):

        with qml.tape.QuantumTape() as tape:
            for wire, state in zip(self.wires, self.parameters[0]):
                if state == 1:
                    qml.PauliX(wire)
        return tape
