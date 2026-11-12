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

"""Built-in decomposition rules for qubit.state_preparation."""

# pylint: disable=too-many-branches,arguments-differ

import pennylane as qp
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops.qubit.state_preparation import BasisState, StatePrep


def _basis_state_decomp_resources(state, wires):  # pylint: disable=unused-argument
    num_wires = len(wires)
    return {qp.X: num_wires - num_wires // 2}  # Estimate 50% of wires (rounded up) to get flipped


@register_resources(_basis_state_decomp_resources, exact=False)
def _basis_state_decomp(state, wires):
    if qp.capture.enabled() or qp.compiler.active():
        # This branch makes sure that state and wires are cast to objects into which
        # a traced loop index is allowed to index (if they aren't already traced)
        state = qp.math.array(state, like="jax")
        wires = qp.math.array(wires, like="jax")

    @qp.for_loop(len(state))
    def _loop(i):
        qp.cond(qp.math.allclose(state[i], 1), qp.X)(wires[i])

    _loop()  # pylint: disable=no-value-for-parameter


add_decomps(BasisState, _basis_state_decomp)


def _stateprep_resources(num_wires):
    return {qp.resource_rep(qp.MottonenStatePreparation, num_wires=num_wires): 1}


@register_resources(_stateprep_resources)
def _state_prep_decomp(state, wires, **_):
    qp.MottonenStatePreparation(state, wires)


add_decomps(StatePrep, _state_prep_decomp)
