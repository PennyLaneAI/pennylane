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

"""Built-in decomposition rules for templates.subroutines.qubitization."""

from pennylane.core.operator import abstractify
from pennylane.decomposition import add_decomps, register_resources, resource_rep
from pennylane.ops import I
from pennylane.templates.subroutines.prepselprep import PrepSelPrep
from pennylane.templates.subroutines.qubitization import Qubitization
from pennylane.templates.subroutines.reflection import Reflection
from pennylane.typing import Wire


def _qubitization_resources(num_control_wires, hamiltonian):
    return {
        resource_rep(
            Reflection,
            base_rep=I(Wire[num_control_wires]),
            num_wires=1,
            num_reflection_wires=1,
        ): 1,
        resource_rep(
            PrepSelPrep,
            op_reps=(abstractify(hamiltonian),),
            num_control=num_control_wires,
        ): 1,
    }


@register_resources(_qubitization_resources)
def _qubitization_decomposition(*_, **kwargs):
    hamiltonian = kwargs["hamiltonian"]
    control = kwargs["control"]

    Reflection(I(control))
    PrepSelPrep(hamiltonian, control=control)


add_decomps(Qubitization, _qubitization_decomposition)
