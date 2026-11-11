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

"""Built-in decomposition rules for templates.subroutines.arithmetic.temporary_and."""

from collections.abc import Sequence

from pennylane import math, ops
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops.op_math.change_op_basis2 import _change_op_basis_abstract
from pennylane.templates.subroutines.arithmetic.temporary_and import TemporaryAND
from pennylane.typing import Wire
from pennylane.wires import WiresLike

# The number of X gates we put into resource estimates for flipping control qubits.
# We choose 2 here because it is the average over the four possible scenarios.
_number_xs = 2


def _temporary_and_resources(*_, **__):
    compute_rep = ops.prod(
        ops.adjoint(ops.T(Wire[1])),
        ops.CNOT(Wire[2]),
        ops.T(Wire[1]),
        ops.Hadamard(Wire[1]),
    )
    uncompute_rep = ops.prod(
        ops.Hadamard(Wire[1]),
        ops.adjoint(ops.T(Wire[1])),
        ops.CNOT(Wire[2]),
        ops.T(Wire[1]),
    )

    resources = {
        ops.X: _number_xs,
        _change_op_basis_abstract(compute_rep, ops.CNOT, uncompute_rep): 1,
        ops.adjoint(ops.S(Wire[1])): 1,
    }
    return resources


@register_resources(_temporary_and_resources, exact=False)
def _temporary_and(wires: WiresLike, control_values: Sequence[bool]):
    ops.cond(math.logical_not(control_values[0]), ops.X)(wires[0])
    ops.cond(math.logical_not(control_values[1]), ops.X)(wires[1])

    def _compute_fn():
        ops.H(wires[2])
        ops.T(wires=wires[2])
        ops.CNOT(wires=[wires[1], wires[2]])
        ops.adjoint(ops.T(wires=wires[2]))

    def _uncompute_fn():
        ops.T(wires=wires[2])
        ops.CNOT(wires=[wires[1], wires[2]])
        ops.adjoint(ops.T(wires=wires[2]))
        ops.H(wires[2])

    ops.change_op_basis(_compute_fn, ops.CNOT(wires=[wires[0], wires[2]]), _uncompute_fn)
    ops.adjoint(ops.S(wires=wires[2]))

    ops.cond(math.logical_not(control_values[0]), ops.X)(wires[0])
    ops.cond(math.logical_not(control_values[1]), ops.X)(wires[1])


def _toffoli_with_cvals(wires, cvals):
    """Conjugate a Toffoli by Pauli X depending on control values."""
    ops.cond(math.logical_not(cvals[0]), ops.X)(wires[0])
    ops.cond(math.logical_not(cvals[1]), ops.X)(wires[1])
    ops.Toffoli(wires)
    ops.cond(math.logical_not(cvals[0]), ops.X)(wires[0])
    ops.cond(math.logical_not(cvals[1]), ops.X)(wires[1])


@register_resources({ops.Toffoli: 1, ops.X: _number_xs}, exact=False)
def _temporary_and_to_toffoli(wires: WiresLike, control_values: Sequence[bool]):
    _toffoli_with_cvals(wires, control_values)


def _temporary_and_ppr_resources(*_, **__):
    return {
        ops.X: _number_xs,
        ops.PPR_8(1, "ZZY", Wire[3]): 1,
        ops.PPR_8(-1, "ZY", Wire[2]): 2,
        ops.PPR_8(1, "Y", Wire[1]): 1,
    }


@register_resources(_temporary_and_ppr_resources, exact=False)
def _temporary_and_ppr(wires: WiresLike, control_values: Sequence[bool]):
    ops.cond(math.logical_not(control_values[0]), ops.X)(wires[0])
    ops.cond(math.logical_not(control_values[1]), ops.X)(wires[1])
    ops.PPR_8(1, "ZZY", wires)
    ops.PPR_8(-1, "ZY", wires[1:])
    ops.PPR_8(-1, "ZY", wires[::2])
    ops.PPR_8(1, "Y", wires[2:])
    ops.cond(math.logical_not(control_values[0]), ops.X)(wires[0])
    ops.cond(math.logical_not(control_values[1]), ops.X)(wires[1])


add_decomps(TemporaryAND, _temporary_and, _temporary_and_to_toffoli, _temporary_and_ppr)


def _adjoint_temporary_and_resources(*_, **__):
    return {
        ops.Hadamard: 1,
        ops.MidMeasure(Wire[1], reset=True): 1,
        ops.CZ: 1,
        ops.X: _number_xs,
    }


@register_resources(_adjoint_temporary_and_resources, exact=False)
def _adjoint_temporary_and(base):
    r"""The implementation of adjoint TemporaryAND by mid-circuit measurements
    as found in https://arxiv.org/abs/1805.03662."""
    cvals = base.control_values
    ops.cond(math.logical_not(cvals[0]), ops.X)(base.wires[0])
    ops.cond(math.logical_not(cvals[1]), ops.X)(base.wires[1])
    ops.Hadamard(wires=base.wires[2])
    m_0 = ops.measure(base.wires[2], reset=True)
    ops.cond(m_0, ops.CZ)(wires=base.wires[:2])
    ops.cond(math.logical_not(cvals[0]), ops.X)(base.wires[0])
    ops.cond(math.logical_not(cvals[1]), ops.X)(base.wires[1])


@register_resources(
    {
        ops.PauliMeasure("X", Wire[1]): 1,
        ops.PauliMeasure("Z", Wire[1]): 1,
        ops.CZ: 1,
        ops.X: _number_xs + 1,
    },
    exact=False,
)
def _adjoint_temporary_and_ppm(base):
    r"""The implementation of adjoint TemporaryAND by Pauli product measurements + Pauli/Clifford
    corrections, similar to _adjoint_temporary_and above from https://arxiv.org/abs/1805.03662."""
    cvals = base.control_values
    ops.cond(math.logical_not(cvals[0]), ops.X)(base.wires[0])
    ops.cond(math.logical_not(cvals[1]), ops.X)(base.wires[1])

    m_0 = ops.pauli_measure("X", base.wires[2])
    ops.cond(m_0, ops.CZ)(wires=base.wires[:2])

    # Reset aux qubit to |0>
    m_1 = ops.pauli_measure("Z", base.wires[2])
    ops.cond(m_1, ops.X)(base.wires[2])

    ops.cond(math.logical_not(cvals[0]), ops.X)(base.wires[0])
    ops.cond(math.logical_not(cvals[1]), ops.X)(base.wires[1])


@register_resources({ops.Toffoli: 1, ops.X: _number_xs}, exact=False)
def _adjoint_temporary_and_to_toffoli(base):
    _toffoli_with_cvals(base.wires, base.control_values)


add_decomps(
    "Adjoint(TemporaryAND)",
    _adjoint_temporary_and_to_toffoli,
    _adjoint_temporary_and_ppm,
    _adjoint_temporary_and,
)
