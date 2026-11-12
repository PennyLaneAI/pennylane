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

"""Built-in decomposition rules for templates.subroutines.time_evolution.commuting_evolution."""

# pylint: disable-msg=too-many-arguments

from pennylane import capture, ops
from pennylane.core.apply import apply
from pennylane.core.operator import Operator1
from pennylane.decomposition import add_decomps, register_resources, resource_rep
from pennylane.ops.op_math.linear_combination import Hamiltonian
from pennylane.pauli import PauliWord
from pennylane.templates.subroutines.time_evolution.approx_time_evolution import ApproxTimeEvolution
from pennylane.templates.subroutines.time_evolution.commuting_evolution import CommutingEvolution
from pennylane.wires import WiresLike


def _commuting_evolution_resources(words: tuple[PauliWord]):
    return {resource_rep(ApproxTimeEvolution, words=words, n=1): 1}


@register_resources(_commuting_evolution_resources)
def _commuting_evolution_decomposition(
    time: list, *_, wires: WiresLike, hamiltonian: Hamiltonian, **__
):  # pylint: disable=unused-argument
    if capture.enabled() and isinstance(hamiltonian, Operator1):
        # Reconstruct the closed-over Hamiltonian as a captured value. It is consumed by
        # ``ApproxTimeEvolution`` below, so it is not collected as a separate circuit operation.
        hamiltonian = apply(hamiltonian)
    ApproxTimeEvolution(hamiltonian, time, 1)


add_decomps(CommutingEvolution, _commuting_evolution_decomposition)


@register_resources(lambda base_params, **_: {resource_rep(CommutingEvolution, **base_params): 1})
def _adjoint_commuting_evolution(time, *params, base, **__):
    hamiltonian = ops.functions.bind_new_parameters(base.hyperparameters["hamiltonian"], params)
    CommutingEvolution(
        hamiltonian,
        -time,
        frequencies=base.hyperparameters["frequencies"],
        shifts=base.hyperparameters["shifts"],
    )


add_decomps("Adjoint(CommutingEvolution)", _adjoint_commuting_evolution)
