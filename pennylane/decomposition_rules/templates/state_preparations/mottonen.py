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


"""Built-in decomposition rules for templates.state_preparations.mottonen."""

from pennylane import decomposition, ops
from pennylane.templates.state_preparations.mottonen import MottonenStatePreparation


def _mottonen_resources(num_wires):
    n = 2**num_wires - 1  # Equal to `sum(2**i for i in range(num_wires))`

    return {ops.GlobalPhase: 1, ops.RY: n, ops.RZ: n, ops.CNOT: 2 * (n - 1)}


@decomposition.register_resources(_mottonen_resources, exact=False, name="standard")
def mottonen_decomp(state_vector, wires, **_):
    """Decompose a state vector using the Mottonen preparation algorithm."""
    return MottonenStatePreparation.compute_decomposition(state_vector, wires, **_)


decomposition.add_decomps(MottonenStatePreparation, mottonen_decomp)
