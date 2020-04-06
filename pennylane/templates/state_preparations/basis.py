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
from pennylane.templates.utils import check_wires, check_no_variable, check_shape, get_shape


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
        wires (Sequence[int]): sequence of qubit indices that the template acts on

    Raises:
        ValueError: if inputs do not have the correct format
    """

    ######################
    # Input checks

    wires = check_wires(wires)

    expected_shape = (len(wires),)
    check_shape(
        basis_state,
        expected_shape,
        msg=" 'basis_state' must be of shape {}; got {}."
        "".format(expected_shape, get_shape(basis_state)),
    )

    # basis_state cannot be trainable
    check_no_variable(
        basis_state,
        msg="'basis_state' cannot be differentiable; must be passed as a keyword argument "
        "to the quantum node",
    )

    # basis_state is guaranteed to be a list of binary values
    if any([b not in [0, 1] for b in basis_state]):
        raise ValueError(
            "'basis_state' must only contain values of 0 and 1; got {}".format(basis_state)
        )

    ######################

    for wire, state in zip(wires, basis_state):
        if state == 1:
            qml.PauliX(wire)
