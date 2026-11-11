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

"""Built-in decomposition rules for templates.subroutines.hilbert_schmidt."""

from collections import defaultdict
from collections.abc import Iterable

from pennylane import capture, math
from pennylane.control_flow import for_loop
from pennylane.core.apply import apply
from pennylane.core.operator import Operator
from pennylane.decomposition import CompressedResourceOp, add_decomps, register_resources
from pennylane.ops import CNOT, Hadamard, QubitUnitary
from pennylane.templates.subroutines.hilbert_schmidt import HilbertSchmidt, LocalHilbertSchmidt
from pennylane.typing import Complex, TensorLike, Wire
from pennylane.wires import Wires


def _hilbert_schmidt_resources(
    num_wires: int, u_reps: list[CompressedResourceOp], v_wires: list[int]
):
    num_first_range = num_wires // 2
    num_second_range = num_wires - num_first_range

    resources = defaultdict(int)

    resources.update(
        {Hadamard: num_first_range * 2, CNOT: min(num_first_range, num_second_range) * 2}
    )

    for op_rep in u_reps:
        resources[op_rep] += 1

    for n_wires in v_wires:
        resources[QubitUnitary(Complex[2**n_wires, 2**n_wires], wires=Wire[n_wires])] += 1

    return resources


def _local_hilbert_schmidt_resources(
    num_wires: int, u_reps: list[CompressedResourceOp], v_wires: list[int]
):
    num_first_range = num_wires // 2
    num_second_range = num_wires - num_first_range

    resources = defaultdict(int)

    resources.update(
        {Hadamard: num_first_range + 1, CNOT: min(num_first_range, num_second_range) + 1}
    )

    for op_rep in u_reps:
        resources[op_rep] += 1

    for n_wires in v_wires:
        resources[QubitUnitary(Complex[2**n_wires, 2**n_wires], wires=Wire[n_wires])] += 1

    return resources


def _up_to_last_layer(
    wires: int | Iterable[int | str] | Wires,
    U: Operator | Iterable[Operator],
    V: Operator | Iterable[Operator],
) -> tuple[int, Iterable, Iterable, list[int]]:
    u_ops = (U,) if isinstance(U, Operator) else tuple(U)
    v_ops = (V,) if isinstance(V, Operator) else tuple(V)

    def collect_wires_in_order(ops):
        accumulator = []
        for operator in ops:
            for wire_index in range(  # pylint: disable=consider-using-enumerate
                len(operator.wires)
            ):
                if capture.enabled() and isinstance(operator.wires, Wires):
                    wire = operator.wires.labels[wire_index]
                else:
                    wire = operator.wires[wire_index]

                if wire not in accumulator:
                    accumulator.append(wire)

        return accumulator

    u_wires = collect_wires_in_order(u_ops)
    v_wires = collect_wires_in_order(v_ops)

    n_wires = len(u_wires + v_wires)
    first_range = range(n_wires // 2)
    second_range = range(n_wires // 2, n_wires)

    if capture.enabled():
        if isinstance(wires, Wires):
            wires = wires.labels
        wires, u_wires, v_wires, first_range, second_range = (
            math.array(wires, like="jax"),
            math.array(u_wires, like="jax"),
            math.array(v_wires, like="jax"),
            math.array(first_range, like="jax"),
            math.array(second_range, like="jax"),
        )

    @for_loop(len(first_range))
    def layer_hadamards(i):
        Hadamard(wires[i])

    layer_hadamards()  # pylint: disable=no-value-for-parameter

    @for_loop(min(len(first_range), len(second_range)))
    def layer_cnots(j):
        CNOT(wires=[wires[first_range[j]], wires[second_range[j]]])

    layer_cnots()  # pylint: disable=no-value-for-parameter

    for op_u in u_ops:
        apply(op_u)

    for op_v in v_ops:
        mat = op_v.matrix().conjugate()
        QubitUnitary(mat, wires=op_v.wires)

    return n_wires, first_range, second_range, wires


@register_resources(_hilbert_schmidt_resources)
def _hilbert_schmidt_decomposition(
    *params: TensorLike,
    wires: int | Iterable[int | str] | Wires,
    U: Operator | Iterable[Operator],
    V: Operator | Iterable[Operator],
):  # pylint: disable=unused-argument
    _, first_range, second_range, wires = _up_to_last_layer(wires, U, V)

    @for_loop(min(len(first_range), len(second_range)))
    def layer_cnots(j):
        CNOT(wires=[wires[first_range[j]], wires[second_range[j]]])

    layer_cnots()  # pylint: disable=no-value-for-parameter

    @for_loop(len(first_range))
    def layer_hadamards(i):
        Hadamard(wires[i])

    layer_hadamards()  # pylint: disable=no-value-for-parameter


@register_resources(_local_hilbert_schmidt_resources)
def _local_hilbert_schmidt_decomposition(
    *params: TensorLike,
    wires: int | Iterable[int | str] | Wires,
    U: Operator | Iterable[Operator],
    V: Operator | Iterable[Operator],
):  # pylint: disable=unused-argument
    n_wires, _, _, wires = _up_to_last_layer(wires, U, V)

    CNOT(wires=[wires[0], wires[n_wires // 2]])
    Hadamard(wires[0])


add_decomps(HilbertSchmidt, _hilbert_schmidt_decomposition)

add_decomps(LocalHilbertSchmidt, _local_hilbert_schmidt_decomposition)
