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

"""Built-in decomposition rules for templates.subroutines.time_evolution.trotter."""

from pennylane import ops as qp_ops
from pennylane.decomposition import add_decomps, register_resources, resource_rep
from pennylane.templates.subroutines.time_evolution.trotter import TrotterProduct, _scalar


def _trotter_product_decomposition_resources(n, order, ops):
    reps = {}

    def _count(op, ops):
        count = 0
        for nxt in ops:
            if nxt == op:
                count += 1
        return count

    if order == 1:
        for op in ops:
            reps[resource_rep(qp_ops.op_math.Evolution, base=op)] = n * _count(op, ops)
        return reps
    if order == 2:
        for op in ops:
            reps[resource_rep(qp_ops.op_math.Evolution, base=op)] = n * 2 * _count(op, ops)
        return reps
    for op in ops:
        reps[resource_rep(qp_ops.op_math.Evolution, base=op)] = (
            n * _count(op, ops) * 2 * 5 * (order - 2) // 2
        )
    return reps


@register_resources(_trotter_product_decomposition_resources)
def _trotter_product_decomposition(*args, **kwargs):
    time = args[-1]
    n = kwargs["n"]
    order = kwargs["order"]
    ops = kwargs["base"].operands

    def _recursive(x, order, ops):
        if order == 1:
            for op in ops[::-1]:
                qp_ops.Evolution(op, -x)
            return

        if order == 2:
            for op in ops + ops[::-1]:
                qp_ops.Evolution(op, -x * 0.5)
            return

        scalar_1 = _scalar(order)
        scalar_2 = 1 - 4 * scalar_1

        for _ in range(2):
            _recursive(scalar_1 * x, order - 2, ops)
        _recursive(scalar_2 * x, order - 2, ops)
        for _ in range(2):
            _recursive(scalar_1 * x, order - 2, ops)

    for _ in range(n):
        _recursive(time / n, order, ops)


add_decomps(TrotterProduct, _trotter_product_decomposition)
