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

"""Built-in decomposition rules for templates.subroutines.ffft."""

from collections import defaultdict

from pennylane import capture, math
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import FermionicSWAP, PauliZ
from pennylane.ops.op_math.pow2 import _pow_abstract
from pennylane.templates.subroutines.ffft import FFFT, TwoWireFFT, _recursive_decompose
from pennylane.wires import WiresLike


def _fast_fermionic_fourier_transform_resources(num_wires):
    resources = defaultdict(int)

    log2_wires = int(math.log2(num_wires))

    two_qubit_gates = num_wires * log2_wires // 2
    resources[TwoWireFFT] = two_qubit_gates

    def _count_one_recursive(wires, resources):
        if wires > 2:
            for mode in range(wires // 2):
                resources[_pow_abstract(PauliZ, z=2 * mode / wires)] += 1
            resources = _count_one_recursive(wires // 2, resources)
            resources = _count_one_recursive(wires // 2, resources)
        return resources

    resources = _count_one_recursive(num_wires, resources)

    if num_wires > 2:
        resources[FermionicSWAP] = num_wires * (num_wires - log2_wires - 1) // 2

    return resources


@register_resources(_fast_fermionic_fourier_transform_resources)
def _fast_fermionic_fourier_transform_decomposition(*_, wires: WiresLike, **__):
    if capture.enabled():
        wires = math.array(wires, like="jax")

    # Rather than performing a bit-reversal permutation, we expect the user to label their wires
    # correctly at the beginning.

    _recursive_decompose(wires)


add_decomps(FFFT, _fast_fermionic_fourier_transform_decomposition)
