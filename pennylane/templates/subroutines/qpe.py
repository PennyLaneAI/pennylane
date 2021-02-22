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
"""
Contains the ``QuantumPhaseEstimation`` template.
"""
from numpy.linalg import matrix_power

import pennylane as qml
from pennylane.templates.decorator import template
from pennylane.wires import Wires


@template
def QuantumPhaseEstimation(unitary, target_wires, estimation_wires):
    r"""Performs the
    `quantum phase estimation <https://en.wikipedia.org/wiki/Quantum_phase_estimation_algorithm>`__
    circuit.

    Given a unitary :math:`U`, this template applies the circuit for quantum phase
    estimation. The unitary is applied to the qubits specified by ``target_wires`` and :math:`n`
    qubits are used for phase estimation as specified by ``estimation_wires``.

    .. figure:: ../../_static/templates/subroutines/qpe.svg
        :align: center
        :width: 60%
        :target: javascript:void(0);

    Args:
        unitary (array): the phase estimation unitary
        target_wires (Union[Wires, Sequence[int], or int]): the target wires to apply the unitary
        estimation_wires (Union[Wires, Sequence[int], or int]): the wires to be used for phase
            estimation

    Raises:
        QuantumFunctionError: if the ``target_wires`` and ``estimation_wires`` share a common
            element

    .. UsageDetails::

        TODO
    """

    target_wires = Wires(target_wires)
    estimation_wires = Wires(estimation_wires)

    if len(Wires.shared_wires([target_wires, estimation_wires])) != 0:
        raise qml.QuantumFunctionError("The target wires and estimation wires must be different")

    for i, wire in enumerate(estimation_wires):
        qml.Hadamard(wire)

        # Could we calculate the matrix power more efficiently by diagonalizing?
        u = matrix_power(unitary, 2 ** i)

        qml.ControlledQubitUnitary(u, control_wires=wire, wires=target_wires)

    qml.QFT(wires=estimation_wires).inv()
