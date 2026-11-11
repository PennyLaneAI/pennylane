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

"""Built-in decomposition rules for templates.subroutines.fable."""

# pylint: disable=no-value-for-parameter

from collections import Counter

import numpy as np

from pennylane import math
from pennylane.control_flow import for_loop
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import CNOT, RY, SWAP, Hadamard, cond
from pennylane.templates.state_preparations.mottonen import compute_theta, gray_code
from pennylane.templates.subroutines.fable import FABLE


def _fable_resources(wires, thetas, control_wires, tol):

    resources = Counter({Hadamard: len(wires) // 2 * 2})

    wires_i = wires[1 : 1 + len(wires) // 2][::-1]
    wires_j = wires[1 + len(wires) // 2 : len(wires)][::-1]

    wire_map = dict(enumerate(wires_j + wires_i))

    nots = {}
    for theta, control_index in zip(thetas, control_wires, strict=True):

        if math.abs(2 * theta) > tol:
            for _ in nots:
                resources[CNOT] += 1
            resources[RY] += 1
            nots = {}

        if wire_map[control_index] in nots:
            del nots[wire_map[control_index]]
        else:
            nots[wire_map[control_index]] = 1

    for _ in range(len(nots.keys())):
        resources[CNOT] += 1

    resources[SWAP] = min(len(wires_i), len(wires_j))

    return dict(resources)


@register_resources(_fable_resources)
def _fable_decomposition(input_matrix, wires, tol=0):
    alphas = math.arccos(input_matrix).flatten()
    thetas = compute_theta(alphas)

    auxilliary = [wires[0]]
    wires_i = wires[1 : 1 + len(wires) // 2][::-1]
    wires_j = wires[1 + len(wires) // 2 : len(wires)][::-1]

    code = gray_code(int(2 * math.log2(len(input_matrix))))
    control_wires = np.log2(code ^ np.roll(code, -1)).astype(int)

    wire_map = dict(enumerate(wires_j + wires_i))

    @for_loop(len(wires_i))
    def wires_i_hadamard_loop(i):
        w = wires_i[i]
        Hadamard(w)

    wires_i_hadamard_loop()

    nots = {}

    @for_loop(min(len(thetas), len(control_wires)))
    def theta_controls_loop(j, nots):
        theta = thetas[j]
        control_index = control_wires[j]

        def in_tol_branch(nots):

            @for_loop(len(nots.keys()))
            def concrete_nots_loop(l, nots):
                c_wire = list(nots.keys())[l]
                CNOT(wires=[c_wire] + auxilliary)
                return nots

            concrete_nots_loop(nots)

            RY(2 * theta, wires=auxilliary)
            nots = {}

            return nots

        def trivial_branch(nots):
            return nots

        nots = cond(math.abs(2 * theta) > tol, in_tol_branch, trivial_branch)(nots)

        def del_branch(nots):
            del nots[wire_map[control_index]]
            return nots

        def add_branch(nots):
            nots[wire_map[control_index]] = 1
            return nots

        nots = cond(wire_map[control_index] in nots, del_branch, add_branch)(nots)

        return nots

    nots = theta_controls_loop(nots)

    @for_loop(len(nots.keys()))
    def outer_nots_loop(m, nots):
        c_wire = list(nots.keys())[m]
        CNOT([c_wire] + auxilliary)

    outer_nots_loop(nots)

    @for_loop(min(len(wires_i), len(wires_j)))
    def swap_loop(n):
        w_i = wires_i[n]
        w_j = wires_j[n]
        SWAP(wires=[w_i, w_j])

    swap_loop()

    wires_i_hadamard_loop()


add_decomps(FABLE, _fable_decomposition)
