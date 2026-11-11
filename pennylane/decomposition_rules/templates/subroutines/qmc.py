# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Built-in decomposition rules for templates.subroutines.qmc."""

# pylint: disable=too-many-arguments,too-many-positional-arguments

from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import QubitUnitary
from pennylane.templates.subroutines.qmc import QuantumMonteCarlo
from pennylane.templates.subroutines.qpe import QuantumPhaseEstimation
from pennylane.typing import Complex, Wire


def _quantum_monte_carlo_resources(num_target_wires, num_estimation_wires, q_shape):
    return {
        QubitUnitary(
            Complex[2 ** (num_target_wires - 1), 2 ** (num_target_wires - 1)],
            wires=Wire[num_target_wires - 1],
        ): 1,
        QubitUnitary(
            Complex[2**num_target_wires, 2**num_target_wires], wires=Wire[num_target_wires]
        ): 1,
        QuantumPhaseEstimation(
            QubitUnitary(Complex[*q_shape], wires=Wire[num_target_wires]),
            estimation_wires=Wire[num_estimation_wires],
        ): 1,
    }


@register_resources(_quantum_monte_carlo_resources)
def _quantum_monte_carlo_decomposition(
    A, R, Q, wires, estimation_wires, target_wires
):  # pylint: disable=unused-argument
    QubitUnitary(A, wires=target_wires[:-1])
    QubitUnitary(R, wires=target_wires)
    QuantumPhaseEstimation(QubitUnitary(Q, wires=target_wires), estimation_wires=estimation_wires)


add_decomps(QuantumMonteCarlo, _quantum_monte_carlo_decomposition)
