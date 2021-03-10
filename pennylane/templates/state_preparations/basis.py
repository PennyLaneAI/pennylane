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
r"""
Contains the ``BasisStatePreparation`` template.
"""

import pennylane as qml
from pennylane.templates.decorator import template
from pennylane.templates.utils import check_shape, get_shape
from pennylane.wires import Wires


def _preprocess(basis_state, wires):
    """Validate and pre-process inputs as follows:

    * Check the shape of the basis state.
    * Cast basis state to a numpy array.

    Args:
        basis_state (tensor_like): basis state to prepare
        wires (Wires): wires that template acts on

    Returns:
        array: preprocessed basis state
    """

    if qml.tape_mode_active():

        shape = qml.math.shape(basis_state)

        if len(shape) != 1:
            raise ValueError(f"Basis state must be one-dimensional; got shape {shape}.")

        n_bits = shape[0]
        if n_bits != len(wires):
            raise ValueError(f"Basis state must be of length {len(wires)}; got length {n_bits}.")

        basis_state = list(qml.math.toarray(basis_state))

        if not all(bit in [0, 1] for bit in basis_state):
            raise ValueError(f"Basis state must only consist of 0s and 1s; got {basis_state}")

        # we return the input as a list of values, since
        # it is not differentiable
        return basis_state

    expected_shape = (len(wires),)
    check_shape(
        basis_state,
        expected_shape,
        msg="Basis state must be of shape {}; got {}."
        "".format(expected_shape, get_shape(basis_state)),
    )

    # basis_state is guaranteed to be a list of binary values
    if any([b not in [0, 1] for b in basis_state]):
        raise ValueError(
            "Basis state must only contain values of 0 and 1; got {}".format(basis_state)
        )

    return basis_state


@template
def BasisStatePreparation(basis_state, wires):
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
    wires = Wires(wires)

    basis_state = _preprocess(basis_state, wires)

    for wire, state in zip(wires, basis_state):
        if state == 1:
            qml.PauliX(wire)
