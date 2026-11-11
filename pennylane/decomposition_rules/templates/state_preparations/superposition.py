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

"""Built-in decomposition rules for templates.state_preparations.superposition."""

from collections import Counter
from functools import reduce

import pennylane as qp
from pennylane.control_flow import for_loop
from pennylane.decomposition import add_decomps, register_resources, resource_rep
from pennylane.templates.state_preparations.superposition import Superposition, order_states
from pennylane.typing import Bool, Wire
from pennylane.wires import concatenate_wires


def _permutation_operator_qfunc(basis1, basis2, wires, work_wire):
    r"""
    Creates operations that map an initial basis state to a target basis state using an auxiliary qubit.

    Args:
        basis1 (List): The initial basis state, represented as a list of binary digits.
        basis2 (List): The target basis state, represented as a list of binary digits.
        wires (Sequence[int]): The list of wires that the operator acts on.
        work_wire (Union[Wires, int, str]): The auxiliary wire used for the permutation.
    """

    qp.MultiControlledX(concatenate_wires(wires, work_wire), control_values=basis1)

    @for_loop(len(basis1))
    def apply_cnots(i):
        b = basis1[i]

        def apply_cnot():
            qp.CNOT(wires=work_wire + wires[i])

        qp.cond(b != basis2[i], apply_cnot)()

    apply_cnots()  # pylint: disable=no-value-for-parameter

    qp.MultiControlledX(concatenate_wires(wires, work_wire), control_values=basis2)


def _superposition_resources(num_wires, num_coeffs, bases):
    perms = order_states(bases)

    resources = Counter()

    resources[resource_rep(qp.StatePrep, num_wires=qp.math.ceil_log2(num_coeffs))] += 1

    for basis2, basis1 in perms.items():
        if not qp.math.allclose(basis1, basis2):
            resources[qp.MultiControlledX(Wire[num_wires + 1], control_values=Bool[num_wires])] += 2

            resources[qp.CNOT] += reduce(
                lambda acc, ib: acc
                + int(ib[1] != basis2[ib[0]]),  # pylint: disable=cell-var-from-loop
                enumerate(basis1),
                0,
            )

    return dict(resources)


@register_resources(_superposition_resources)
def _superposition_decomposition(
    coeffs, bases, wires, target_wires, work_wire  # pylint: disable=unused-argument
):
    dic_state = dict(zip(bases, coeffs, strict=True))
    perms = order_states(bases)
    new_dic_state = {perms[key]: dic_state[key] for key in dic_state if key in perms}

    sorted_coefficients = [
        value
        for key, value in sorted(
            new_dic_state.items(), key=lambda item: int("".join(map(str, item[0])), 2)
        )
    ]

    qp.StatePrep(
        qp.math.stack(sorted_coefficients),
        wires=target_wires[-qp.math.ceil_log2(len(coeffs)) :],
        pad_with=0,
    )

    bas = list(perms.items())

    @for_loop(len(list(perms.keys())))
    def apply_permutations(i):
        basis2, basis1 = bas[i][0], bas[i][1]
        if not qp.math.allclose(basis1, basis2):
            _permutation_operator_qfunc(basis1, basis2, target_wires, work_wire)

    apply_permutations()  # pylint: disable=no-value-for-parameter


add_decomps(Superposition, _superposition_decomposition)
