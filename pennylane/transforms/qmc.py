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
from pennylane import PauliX, Hadamard, MultiControlledX, CZ, QFT
from pennylane.wires import Wires
from pennylane.transforms import adjoint


def _apply_controlled_z(wires, control_wire, work_wires):
    r"""Provides the circuit to apply a controlled version of the :math:`Z` gate defined in
    `this <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.98.022321>`__ paper.

    The multi-qubit gate :math:`Z = I - 2|0\rangle \langle 0|` can be performed using the
    conventional multi-controlled-Z gate with an additional bit flip on each qubit before and after.

    This function performs the multi-controlled-Z gate via a multi-controlled-X gate by picking an
    arbitrary target wire to perform the X and adding a Hadamard on that wire either side of the
    transformation.

    Additional control from ``control_wire`` is then included within the multi-controlled-X gate.

    Args:
        wires (Union[Wires, Sequence[int], or int]): the wires on which the Z gate is applied
        control_wire (Wires): the control wire from the register of phase estimation qubits
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
    `this <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.98.022321>`__ paper.

    The :math:`V` gate is simply a Pauli-Z gate applied to the ``target_wire``, i.e., the ancilla
    wire in which the expectation value is encoded.

    The controlled version of this gate is then simply a CZ gate.

    Args:
        target_wire (Wires): the ancilla wire in which the expectation value is encoded
        control_wire (Wires): the control wire from the register of phase estimation qubits
    """
    CZ(wires=[control_wire[0], target_wire[0]])


def apply_controlled_Q(fn, wires, target_wire, control_wire, work_wires):
    r"""Provides the circuit to apply a controlled version of the :math:`\mathcal{Q}` unitary
    defined in `this <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.98.022321>`__ paper.

    Given a callable ``fn`` input corresponding to the :math:`\mathcal{F}` unitary in the above
    paper, this function transforms the circuit into a controlled-version of the :math:`\mathcal{Q}`
    unitary which forms part of the quantum Monte Carlo algorithm. In this algorithm, one of the
    wires acted upon by :math:`\mathcal{F}`, specified by ``target_wire``, is used to embed a
    Monte Carlo estimation problem. The :math:`\mathcal{Q}` is then designed to contain the target
    expectation value as a phase in one of its eigenvalues. This function transforms to a controlled
    version of :math:`\mathcal{Q}` that is compatible with quantum phase estimation
    (see :class:`~.QuantumPhaseEstimation` for more details).

    Args:
        fn (Callable): a quantum function that applies quantum operations according to the
            :math:`\mathcal{F}` unitary used as part of quantum Monte Carlo estimation
        wires (Union[Wires, Sequence[int], or int]): the wires acted upon by the ``fn`` circuit
        target_wire (Union[Wires, int]): The wire in which the expectation value is encoded. Must be
            contained within ``wires``.
        control_wire (Union[Wires, int]): the control wire from the register of phase estimation
            qubits
        work_wires (Union[Wires, Sequence[int], or int]): additional work wires used when
            decomposing :math:`\mathcal{Q}`

    Returns:
        function: The input function transformed to the :math:`\mathcal{Q}` unitary

    Raises:
        ValueError: if ``target_wire`` is now in ``wires``
    """
    fn_inv = adjoint(fn)

    wires = Wires(wires)
    target_wire = Wires(target_wire)
    control_wire = Wires(control_wire)
    work_wires = Wires(work_wires)

    if not wires.contains_wires(target_wire):
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

    estimation_wires = Wires(estimation_wires)
    wires = Wires(wires)
    target_wire = Wires(target_wire)

    if Wires.shared_wires([wires, estimation_wires]):
        raise ValueError("No wires can be shared between the wires and estimation_wires registers")

    @wraps(fn)
    def wrapper(*args, **kwargs):
        fn(*args, **kwargs)
        for i, control_wire in enumerate(estimation_wires):
            Hadamard(control_wire)

            # Find wires eligible to be used as helper wires
            work_wires = estimation_wires.toset() - {control_wire}
            n_reps = 2 ** (len(estimation_wires) - (i + 1))

            q = apply_controlled_Q(
                fn,
                wires=wires,
                target_wire=target_wire,
                control_wire=control_wire,
                work_wires=work_wires,
            )

            for _ in range(n_reps):
                q(*args, **kwargs)

        QFT(wires=estimation_wires).inv()

    return wrapper
