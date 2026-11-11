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

"""Built-in decomposition rules for templates.subroutines.permute."""

from collections import Counter

from jax import numpy as jnp

from pennylane import capture
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import SWAP, cond
from pennylane.templates.subroutines.permute import Permute


def _permute_resources(wires, permutation):
    resources = Counter()
    working_order = wires.tolist()

    for idx_here, here in enumerate(permutation):
        if working_order[idx_here] != here:
            idx_there = working_order.index(permutation[idx_here])

            resources[SWAP] += 1

            working_order[idx_here], working_order[idx_there] = (
                working_order[idx_there],
                working_order[idx_here],
            )

    return dict(resources)


@register_resources(_permute_resources)
def _permute_decomposition(wires, permutation):
    # Temporary storage to keep track as we permute
    working_order = wires.tolist()

    if capture.enabled():
        # The swap index is traced inside ``cond``. JAX arrays support indexing with that
        # tracer and the functional updates needed to carry the new order between branches.
        wires = jnp.array(wires)
        working_order = wires

        for idx_here, here in enumerate(permutation):

            def swap(order, idx_here=idx_here, here=here):
                idx_there = jnp.argmax(order == here)
                SWAP(wires=[wires[idx_here], wires[idx_there]])

                here_value = order[idx_here]
                order = order.at[idx_here].set(order[idx_there])
                return order.at[idx_there].set(here_value)

            working_order = cond(working_order[idx_here] != here, swap, lambda order: order)(
                working_order
            )
        return

    # Go through the new order and shuffle things one by one
    for idx_here, here in enumerate(permutation):
        if working_order[idx_here] != here:
            # Where do we need to send the qubit at this location?
            idx_there = working_order.index(permutation[idx_here])

            # SWAP based on the labels of the wires
            SWAP(wires=wires.subset([idx_here, idx_there]))

            # Update the working order to account for the SWAP
            working_order[idx_here], working_order[idx_there] = (
                working_order[idx_there],
                working_order[idx_here],
            )


add_decomps(Permute, _permute_decomposition)
