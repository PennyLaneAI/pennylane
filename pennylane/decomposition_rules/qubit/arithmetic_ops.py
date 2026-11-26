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


"""Built-in decomposition rules for qubit.arithmetic_ops."""

from pennylane import ops
from pennylane.decomposition import (
    add_decomps,
    register_condition,
    register_resources,
    resource_rep,
)
from pennylane.decomposition.symbolic_decomposition import pow_involutory, self_adjoint_legacy
from pennylane.ops.qubit.arithmetic_ops import (
    IntegerComparator,
    QubitCarry,
    QubitSum,
    _integer_comparator_ge_impl,
    _integer_comparator_greater_equal_resources,
    _integer_comparator_less_than_resources,
    _integer_comparator_lt_impl,
)
from pennylane.wires import WiresLike


@register_resources({ops.CNOT: 1, ops.Toffoli: 2}, name="to_cnot_toffolis")
def qubitcarry_to_cnot_toffolis(wires: WiresLike, **__):
    """Decompose ``QubitCarry`` using the ``to_cnot_toffolis`` strategy."""
    ops.Toffoli(wires=wires[1:])
    ops.CNOT(wires=[wires[1], wires[2]])
    ops.Toffoli(wires=[wires[0], wires[2], wires[3]])


add_decomps(QubitCarry, qubitcarry_to_cnot_toffolis)


@register_resources({ops.CNOT: 2}, name="to_cnots")
def qubitsum_to_cnots(wires: WiresLike, **__):
    """Decompose ``QubitSum`` using the ``to_cnots`` strategy."""
    ops.CNOT(wires=[wires[1], wires[2]])
    ops.CNOT(wires=[wires[0], wires[2]])


add_decomps(QubitSum, qubitsum_to_cnots)
add_decomps("Adjoint(QubitSum)", self_adjoint_legacy)
add_decomps("Pow(QubitSum)", pow_involutory)


@register_condition(lambda geq, **_: not geq)
@register_resources(_integer_comparator_less_than_resources, name="less_than")
def integer_comparator_less_than(wires, value, work_wires, **_):
    """Decompose ``IntegerComparator`` for the ``n < value`` condition."""
    return _integer_comparator_lt_impl(wires, value, work_wires)


@register_condition(lambda geq, **_: geq)
@register_resources(_integer_comparator_greater_equal_resources, name="greater_equal")
def integer_comparator_greater_equal(wires, value, work_wires, **_):
    """Decompose ``IntegerComparator`` for the ``n >= value`` condition."""
    return _integer_comparator_ge_impl(wires, value, work_wires)


def _integer_comparator_flip_greater_equal_resources(num_wires, value, num_work_wires, geq, **_):
    """Resource estimation for flipping the geq condition."""
    return {
        ops.X: 1,
        resource_rep(
            IntegerComparator,
            num_wires=num_wires,
            value=value,
            geq=not geq,
            num_work_wires=num_work_wires,
        ): 1,
    }


@register_resources(_integer_comparator_flip_greater_equal_resources, name="flip_greater_equal")
def integer_comparator_flip_greater_equal(value, geq, wires, work_wires, **_):
    """Decompose the IntegerComparator by flipping geq to lt or vice versa."""
    ops.X(wires[-1])
    IntegerComparator(value, wires, geq=not geq, work_wires=work_wires)


add_decomps(
    IntegerComparator,
    integer_comparator_less_than,
    integer_comparator_greater_equal,
    integer_comparator_flip_greater_equal,
)
add_decomps("Adjoint(IntegerComparator)", self_adjoint_legacy)
