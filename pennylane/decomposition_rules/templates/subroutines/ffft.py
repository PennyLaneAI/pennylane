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

import numpy as np

from pennylane import capture, math
from pennylane.capture import hint
from pennylane.control_flow import for_loop
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import FermionicSWAP, PauliZ, pow
from pennylane.ops.op_math.pow2 import _pow_abstract
from pennylane.templates.subroutines.ffft import FFFT, TwoWireFFT
from pennylane.wires import WiresLike


def _fast_fermionic_fourier_transform_standard_resources(num_wires):
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


@register_resources(
    _fast_fermionic_fourier_transform_standard_resources, name="recursive_butterfly"
)
def fast_fermionic_fourier_transform_standard(*_, wires: WiresLike, **__):
    """Decompose ``FFFT`` using the ``recursive_butterfly`` strategy."""
    if capture.enabled():
        wires = math.array(wires, like="jax")

    # Rather than performing a bit-reversal permutation, we expect the user to label their wires
    # correctly at the beginning.

    _recursive_decompose(wires)


def _recursive_decompose(wires: WiresLike):
    # base case is that we have two wires
    if len(wires) == 2:
        TwoWireFFT(wires)
    else:
        _recursive_decompose(wires[: len(wires) // 2])
        _recursive_decompose(wires[len(wires) // 2 :])

        for mode in range(len(wires) // 2):
            pow(PauliZ(wires[len(wires) // 2 + mode]), z=2 * mode / len(wires))

        _permute_and_apply_parallel(wires, TwoWireFFT)


def _permute_and_apply_parallel(wires, operator):
    """
    A permutation algorithm specific to permuting all the 2-site fourier inputs into range
    at the same time, parallelizing as many FSWAPs as possible and cutting the total number
    of them in half.

    Args:
        wires (WiresLike): The wires to permute.
        operator (Type[Operator]): The operator to apply once the Fermions are adjacent in the encoding.
    """
    # pylint: disable=no-value-for-parameter

    num_layers = len(wires) // 2 - 1

    def apply_swaps(start, num_parallel_swaps):

        # The layers contain 1, ..., num_layers parallel swaps
        @hint({"num-iters": (num_layers + 1) / 2})
        @for_loop(start, start + 2 * num_parallel_swaps, 2)
        def swap_layer(i):
            FermionicSWAP(np.pi, [wires[i], wires[i + 1]])

        swap_layer()

    @for_loop(1, len(wires) // 2)
    def permutation_in_layers(num_parallel_swaps):
        # applies a layer of parallel FSWAPs
        apply_swaps(num_layers - (num_parallel_swaps - 1), num_parallel_swaps)

    # applies several layers of FSWAPs that achieve the parallel permutation of all needed indices
    permutation_in_layers()

    # applies the operator on the indices that are now in range
    @for_loop(len(wires) // 2)
    def apply_op(i):
        # apply the op
        operator([wires[2 * i], wires[2 * i + 1]])

    apply_op()

    # number of parallel swaps and their starting index is decreasing this time
    @for_loop(num_layers, 0, -1)
    def permutation_out_layers(num_parallel_swaps):
        # applies a layer of parallel FSWAPs
        apply_swaps(num_layers - (num_parallel_swaps - 1), num_parallel_swaps)

    # applies the inverse permutation
    permutation_out_layers()


add_decomps(FFFT, fast_fermionic_fourier_transform_standard)
