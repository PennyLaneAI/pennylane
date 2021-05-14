# Copyright 2021 Xanadu Quantum Technologies Inc.

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
Contains the quantum_monte_carlo transform.
"""
from functools import wraps
from pennylane import PauliX, Hadamard, MultiControlledX, CZ
from pennylane.wires import Wires
from pennylane.transforms import adjoint


def _apply_controlled_z(wires, control_wire, work_wires):
    r"""Provides the circuit to apply a controlled version of the :math:`Z` gate defined in
    `this <https://arxiv.org/pdf/1805.00109.pdf>`__ paper.

    The multi-qubit gate :math:`Z = I - 2|0\rangle \langle 0|` can be performed using the
    conventional multi-controlled-Z gate with an additional bit flip on each qubit before and after.

    This function performs the multi-controlled-Z gate via a multi-controlled-X gate by picking an
    arbitrary target wire to perform the X and adding a Hadamard on that wire either side of the
    transformation.

    Additional control from ``control_wire`` is then included within the multi-controlled-X gate.

    Args:
        wires (Union[Wires, Sequence[int], or int]): the wires on which the Z gate is applied
        control_wire (int): the control wire from the register of phase estimation qubits
        work_wires (Union[Wires, Sequence[int], or int]): the work wires used in the decomposition
    """
    target_wire = wires[0]
    PauliX(target_wire)
    Hadamard(target_wire)

    control_values = "0" * (len(wires) - 1) + "1"
    control_wires = Wires(wires[1:]) + control_wire
    MultiControlledX(
        control_wires=control_wires,
        wires=target_wire,
        control_values=control_values,
        work_wires=work_wires,
    )

    Hadamard(target_wire)
    PauliX(target_wire)


def _apply_controlled_v(target_wire, control_wire):
    """Provides the circuit to apply a controlled version of the :math:`V` gate defined in
    `this <https://arxiv.org/pdf/1805.00109.pdf>`__ paper.

    The :math:`V` gate is simply a Pauli-Z gate applied to the ``target_wire``, i.e., the ancilla
    wire in which the expectation value is encoded.

    The controlled version of this gate is then simply a CZ gate.

    Args:
        target_wire (int): the ancilla wire in which the expectation value is encoded
        control_wire (int): the control wire from the register of phase estimation qubits
    """
    CZ(wires=[control_wire, target_wire])


def apply_controlled_Q(fn, wires, target_wire, control_wire, work_wires):

    fn_inv = adjoint(fn)

    if target_wire not in wires:
        raise ValueError("The target wire must be contained within wires")

    @wraps(fn)
    def wrapper(*args, **kwargs):
        _apply_controlled_v(target_wire=target_wire, control_wire=control_wire)
        fn_inv(*args, **kwargs)
        _apply_controlled_z(wires=wires, control_wire=control_wire, work_wires=work_wires)
        fn(*args, **kwargs)

        _apply_controlled_v(target_wire=target_wire, control_wire=control_wire)
        fn_inv(*args, **kwargs)
        _apply_controlled_z(wires=wires, control_wire=control_wire, work_wires=work_wires)
        fn(*args, **kwargs)

    return wrapper


def quantum_monte_carlo(fn, wires, target_wire, estimation_wires):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        fn(*args, **kwargs)

    return wrapper
