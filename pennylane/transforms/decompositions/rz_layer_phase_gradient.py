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
r"""
Factory that produces a decomposition rule for :class:`~.RZLayer` in terms of
`phase gradient states <https://pennylane.ai/compilation/phase-gradient/b-rotations>`__, and
private helpers for loading lookup tables with dynamic entries through their algebraic normal
form (ANF).
"""

from collections import Counter

import numpy as np

import pennylane as qp
from pennylane.ops.op_math import ChangeOpBasis2
from pennylane.ops.subcircuit import subcircuit
from pennylane.templates.subroutines.arithmetic.hamming_weight import _HammingWeight
from pennylane.typing import Bool, Float, Wire

from .rz_phase_gradient import make_rz_to_phase_gradient_decomp, validate_phase_gradient_wires


def _anf_bits(table, num_bits):
    r"""Bits of the coefficients of the algebraic normal form (ANF) of a lookup table.

    The ANF writes the table as an XOR of coefficients times products of its input bits, see the
    details on lookup tables in :func:`~.make_rzlayer_to_phase_gradient_decomp`. This lets
    :func:`~._signed_add` load the table with one controlled ``MultiX`` per product.

    The transform uses XORs and static loops only, so that ``table`` may be a traced jax array.

    Args:
        table (TensorLike): integer array of length ``2**r``, whose entry ``v`` is the value at
            the input bits ``v_j`` with ``v = sum(v_j * 2**j)``. The entries are taken modulo
            ``2**num_bits``, so that negative entries are mapped to their two's complement
        num_bits (int): number of bits of the entries

    Returns:
        TensorLike: array of shape ``(2**r, num_bits)``, whose row ``sum(2**j for j in S)``
        holds the bits of the coefficient of the product of the input bits ``v_j`` with ``j`` in
        ``S``, most significant bit first as in :func:`~.math.int_to_binary`

    **Example**

    >>> from pennylane.transforms.decompositions.rz_layer_phase_gradient import _anf_bits
    >>> _anf_bits(np.array([1, 3, 2, 7]), num_bits=3)
    array([[0, 0, 1],
           [0, 1, 0],
           [0, 1, 1],
           [1, 1, 1]])

    The coefficients are ``1, 2, 3, 7``. For example, ``table[3] = 1 ^ 2 ^ 3 ^ 7 = 7`` is the XOR
    of all coefficients, because all products are ``1`` at the input ``v = 3``.
    """
    rows = list(qp.math.int_to_binary(table, num_bits))
    for j in range(len(rows).bit_length() - 1):
        bit = 1 << j
        rows = [row ^ rows[mask ^ bit] if mask & bit else row for mask, row in enumerate(rows)]
    return qp.math.stack(rows)


def _monomial_wires(var_wires, mono_wires):
    """The wires holding the monomials of the variables on ``var_wires``, in the order of the rows
    of :func:`~._anf_bits`: ``v0, v1, v0v1, v2, v0v2, v1v2, v0v1v2, ...``. Each variable is
    followed by its products with the monomials before it, which are held by ``mono_wires``, as
    long as these last."""
    monomials = []
    for var in var_wires:
        products, mono_wires = mono_wires[: len(monomials)], mono_wires[len(monomials) :]
        monomials += [var, *products]
    return monomials


@subcircuit(wire_argnames=("var_wires", "mono_wires"))
@qp.register_resources(lambda mono_wires, **_: {qp.TemporaryAND: len(mono_wires)})
def _Monomials(var_wires, mono_wires):
    r"""Compute the monomials of degree two or higher of the variables on ``var_wires`` into the
    zeroed ``mono_wires``, in the order of :func:`~._monomial_wires`, as long as the
    ``mono_wires`` last. For three variables and four ``mono_wires``:

    .. code-block:: text

        v0v1   = AND(v0, v1)
        v0v2   = AND(v0, v2)
        v1v2   = AND(v1, v2)
        v0v1v2 = AND(v0v1, v2)
    """
    monomials = _monomial_wires(var_wires, mono_wires)
    for j, var in enumerate(var_wires):
        # The 2^j - 1 monomials of the previous variables come first, and their products with var
        # follow right after var itself
        for i in range(min(2**j - 1, len(monomials) - 2**j)):
            qp.TemporaryAND([monomials[i], var, monomials[2**j + i]])


def _rzlayer_phase_gradient_compute_resources(
    control_wires, sign_wires, angle_wires, phase_grad_wires, **_
):
    """Resources of :func:`~._RZLayerPhaseGradientCompute`."""
    load = qp.MultiX(Bool[len(angle_wires)], Wire[len(angle_wires)])
    flip = qp.MultiX(Bool[len(phase_grad_wires)], Wire[len(phase_grad_wires)])
    resources = Counter({load: 1, qp.ctrl(load, control=Wire[1]): len(control_wires)})
    resources[qp.ctrl(flip, control=Wire[1])] += len(sign_wires)
    return resources


@subcircuit(
    dynamic_argnames=("bits",),
    wire_argnames=("control_wires", "sign_wires", "angle_wires", "phase_grad_wires"),
    arg_specs={"bits": Bool[-1, -1]},
    grad_method=None,
)
@qp.register_resources(_rzlayer_phase_gradient_compute_resources)
def _RZLayerPhaseGradientCompute(bits, control_wires, sign_wires, angle_wires, phase_grad_wires):
    r"""XOR a lookup table into ``angle_wires`` and flip ``phase_grad_wires`` if the sign wire, if
    any, is in the state :math:`|1\rangle`. ``sign_wires`` holds zero or one wire.

    ``bits`` holds the ANF coefficients of the table, see :func:`~._anf_bits`. The constant
    coefficient ``bits[0]`` is loaded unconditionally, and ``bits[i]`` controlled on the monomial
    on ``control_wires[i - 1]``. The bits are big-endian, i.e., ``angle_wires[0]`` receives the
    most significant bit, as in :func:`~.math.binary_decimals` and :class:`~.SemiAdder`.
    """
    qp.MultiX(bits[0], angle_wires)
    for i, wire in enumerate(control_wires):
        qp.ctrl(qp.MultiX(bits[i + 1], angle_wires), control=wire)
    for wire in sign_wires:
        qp.ctrl(qp.MultiX([1] * len(phase_grad_wires), phase_grad_wires), control=wire)


def _signed_add(table, var_wires, mono_wires, sign_wires, registers):
    r"""Add the entry of the lookup table ``table`` at the input on ``var_wires`` to the phase
    gradient register, or subtract it if the sign wire, if any, is in the state :math:`|1\rangle`.
    ``sign_wires`` holds zero or one wire, and ``registers`` the angle, phase gradient and adder
    work wires.

    The table is loaded into the angle wires from its ANF, controlled on the monomials of the
    variables, which are computed into the zeroed ``mono_wires``. The monomials that do not fit
    into ``mono_wires`` are skipped. For inputs up to :math:`n` and :math:`r` variables, with
    :math:`r` the bit length of :math:`n`, :math:`n - r` wires in ``mono_wires`` thus leave the
    entries at these inputs intact. The subtraction complements the
    phase gradient register around the addition. The load and the complement are their own
    inverse, because they only target the angle and phase gradient wires, controlled on wires
    that they leave unchanged.

    The integer dtype of ``table`` limits the number of bits: with 32-bit integers, the jax
    default without x64, keep ``len(angle_wires) <= 28``, which leaves headroom for computing the
    tables, or enable x64.
    """
    angle_wires, phase_grad_wires, work_wires = registers
    control_wires = _monomial_wires(var_wires, mono_wires)
    bits = qp.math.cast(_anf_bits(table, len(angle_wires)), bool)[: len(control_wires) + 1]
    load = _RZLayerPhaseGradientCompute(
        bits, control_wires, sign_wires, angle_wires, phase_grad_wires
    )
    add = qp.change_op_basis(load, qp.SemiAdder(angle_wires, phase_grad_wires, work_wires), load)
    return qp.change_op_basis(_Monomials(var_wires, mono_wires), add)


def _signed_add_rep(max_input, num_bits, num_work_wires, num_sign_wires):
    """Resource representation of :func:`~._signed_add` for a table with the inputs up to
    ``max_input``, ``num_bits`` angle wires, ``num_work_wires`` adder work wires and
    ``num_sign_wires`` sign wires."""
    num_vars = max_input.bit_length()
    load = _RZLayerPhaseGradientCompute(
        Bool[max_input + 1, num_bits],
        Wire[max_input],
        Wire[num_sign_wires],
        Wire[num_bits],
        Wire[num_bits],
    )
    target = qp.SemiAdder(Wire[num_bits], Wire[num_bits], Wire[num_work_wires])
    add = ChangeOpBasis2(load, target, load)
    return ChangeOpBasis2(_Monomials(Wire[num_vars], Wire[max_input - num_vars]), add)


def _angle_ints(angles, num_bits):
    r"""Integers whose ``num_bits`` bits are the binary decimals of ``angles`` in units of
    :math:`2\pi`."""
    powers_of_two = 2 ** qp.math.arange(num_bits - 1, -1, -1)
    return qp.math.binary_decimals(angles, num_bits, unit=2 * np.pi) @ powers_of_two


def _equal_angle_block(theta, wires, work_wires, registers):
    r"""Apply :math:`R_Z(\theta)` to each of the :math:`m > 1` ``wires`` with a single
    :func:`~._signed_add`. For odd :math:`m`, this omits the ``GlobalPhase(m * theta / 2)``, which
    the rule applies separately.

    For odd :math:`m`, the rotations apply the phase :math:`\exp(-i\theta (m/2 - h))`, where
    :math:`h` is the Hamming weight of the wires. We compute :math:`h` in place, see
    :func:`~._HammingWeight`, and add the lookup table :math:`L(h) = k h \bmod 2^b` to the phase
    gradient register of :math:`b` wires, where :math:`\theta \approx 2\pi k / 2^b`.

    For even :math:`m`, we avoid the global phase with the sign wire :math:`s` = ``wires[-1]``.
    With the parities :math:`y_i = x_i \oplus x_s` of the other wires, the rotations apply the
    phase :math:`\exp(-i\theta z_s (m/2 - h))`, where :math:`z_s = (-1)^{x_s}` and :math:`h` is
    the Hamming weight of the parities. We compute the parities and :math:`h` in place, and add
    the lookup table :math:`L(h) = -k (m/2 - h) \bmod 2^b`, or subtract it if :math:`x_s = 1`.

    As :math:`h \leq n` for the :math:`n` wires of the Hamming weight, we skip the monomials of
    its bits that vanish for all such :math:`h`. Each AND of the Hamming weight and of the
    remaining monomials targets one of the ``work_wires``.
    """
    m = len(wires)
    # For even m, we compute the Hamming weight of all but the sign wire
    n = m - 1 + m % 2
    hw_wires, sign_wires = wires[:n], wires[n:]
    num_hw_work_wires = n - n.bit_count()
    var_wires = [hw_wires[-1], *work_wires[: n.bit_length() - 1]]
    mono_wires = work_wires[num_hw_work_wires : num_hw_work_wires + n - n.bit_length()]
    k = _angle_ints(theta, len(registers[0]))
    weights = qp.math.arange(2 ** len(var_wires))
    table = k * weights if m % 2 else -k * (m // 2 - weights)
    signed_add = _signed_add(table, var_wires, mono_wires, sign_wires, registers)
    hamming_weight = _HammingWeight(hw_wires, work_wires[:num_hw_work_wires])
    if m % 2:
        qp.change_op_basis(hamming_weight, signed_add)
        return
    # XOR the sign wire into the other wires, which is its own inverse
    parities = qp.ctrl(qp.MultiX([1] * n, hw_wires), control=sign_wires[0])
    qp.change_op_basis(parities, qp.change_op_basis(hamming_weight, signed_add), parities)


def _equal_angle_block_rep(num_wires, num_bits, num_work_wires):
    """Resource representation of :func:`~._equal_angle_block` on ``num_wires`` wires, with
    ``num_bits`` angle wires and ``num_work_wires`` adder work wires."""
    n = num_wires - 1 + num_wires % 2
    num_sign_wires = 1 - num_wires % 2
    hamming_weight = _HammingWeight(Wire[n], Wire[n - n.bit_count()])
    signed_add = _signed_add_rep(n, num_bits, num_work_wires, num_sign_wires)
    if num_wires % 2:
        return ChangeOpBasis2(hamming_weight, signed_add)
    parities = qp.ctrl(qp.MultiX(Bool[n], Wire[n]), control=Wire[1])
    return ChangeOpBasis2(parities, ChangeOpBasis2(hamming_weight, signed_add), parities)


def _num_and_work_wires(num_wires):
    """Number of work wires that the TemporaryAND gates of the phase gradient rule use on top of
    the adder."""
    if num_wires == 1:
        return 0
    n = num_wires - 1 + num_wires % 2
    return 2 * n - n.bit_count() - n.bit_length()


def make_rzlayer_to_phase_gradient_decomp(angle_wires, phase_grad_wires, work_wires):
    r"""
    Create a custom decomposition rule for :class:`~.RZLayer` operators, in which the rotations by
    an equal angle share one :class:`~.SemiAdder` of the
    `phase gradient decomposition <https://pennylane.ai/compilation/phase-gradient/b-rotations>`__.

    As for :func:`~.make_rz_to_phase_gradient_decomp`, the auxiliary wires of the phase gradient
    decomposition are taken as inputs and bound into the created decomposition rule, which can
    then be registered with :func:`~pennylane.decomposition.add_decomps`. The rule decomposes an
    :class:`~.RZLayer` on :math:`m` wires as follows:

    * **m > 1.** The :math:`m` rotations by an equal angle use one addition. For odd :math:`m`,
      they apply a multiple of the angle that depends on the Hamming weight of the wires, and a
      global phase, which the rule applies with a :class:`~.GlobalPhase`.
      For even :math:`m`, the multiple depends on the Hamming weight of the parities of the
      first :math:`m-1` wires with the last one, and is added or subtracted depending on the last
      wire, without a global phase. The rule computes the Hamming weight in place, loads the
      multiple from a lookup table into the angle wires, and adds it. On top of the adder,
      this uses :math:`2n - \|n\|_H - r` :class:`~.TemporaryAND` gates, with :math:`n=m` for odd
      and :math:`n=m-1` for even :math:`m`, :math:`\|n\|_H` the Hamming weight of :math:`n`, and
      :math:`r` the number of bits of :math:`n`.
    * **m = 1.** The rotation uses the circuit of :func:`~.make_rz_to_phase_gradient_decomp`,
      including its :class:`~.GlobalPhase`.

    Each of these :class:`~.TemporaryAND` gates targets a work wire. The rule takes these work
    wires from the ``work_wires`` that the adder does not need, together with the ``work_wires``
    of the :class:`~.RZLayer`. If there are too few,
    it allocates the missing ones dynamically, which excludes it from compilation with Catalyst.
    The ``work_wires`` of the rule and those of the :class:`~.RZLayer` must not overlap.

    The rotations have the precision of :func:`~.make_rz_to_phase_gradient_decomp` with the same
    registers: their angle is rounded to :math:`b` = ``len(angle_wires)`` binary decimals in units
    of :math:`2\pi`.

    Parameters:
        angle_wires (Wires): wires that encode the binary representation of the rotation angle
        phase_grad_wires (Wires): wires that carry a phase gradient state. Should have the same
            length as ``angle_wires``.
        work_wires (Wires): work wires for the :class:`~.SemiAdder` decomposition, of which at
            least ``len(angle_wires) - 1`` are required. Any further ones are used by the
            :class:`~.TemporaryAND` gates described above.

    Returns:
        qp.decomposition.DecompositionRule: decomposition rule to be used within
        :func:`~.pennylane.decompose`.

    .. seealso:: :func:`~.make_rz_to_phase_gradient_decomp`

    **Example**

    We decompose four rotations by an equal angle, with two work wires. We register the rule
    and make rotations expensive in the gate set, so that the decomposition favours the rule:

    .. code-block:: python

        import pennylane as qp
        from pennylane.transforms.decompositions import make_rzlayer_to_phase_gradient_decomp

        custom_decomp = make_rzlayer_to_phase_gradient_decomp(
            angle_wires=[6, 7, 8], phase_grad_wires=[9, 10, 11], work_wires=[12, 13]
        )
        qp.add_decomps(qp.RZLayer, custom_decomp)
        gate_set = {
            "RZ": 100, "PhaseShift": 100, "GlobalPhase": 1, "CNOT": 1, "MultiX": 1,
            "C(MultiX)": 1, "TemporaryAND": 1, "Adjoint(TemporaryAND)": 1, "SemiAdder": 1,
        }

        def rotations(theta):
            qp.RZLayer(theta, wires=[0, 1, 2, 3], work_wires=[4, 5])
            return qp.state()

        circuit = qp.qnode(qp.device("lightning.qubit", wires=14))(rotations)
        compiled = qp.qjit(qp.decompose(circuit, gate_set=gate_set), capture=True)

    The rotations use one :class:`~.SemiAdder` and two :class:`~.TemporaryAND` gates:

    >>> dict(sorted(qp.analyze(compiled)(2.4).resources.quantum_operations.items()))
    {'1C(MultiX)': 10, 'Adjoint(TemporaryAND)': 2, 'CNOT': 10, 'MultiX': 2, 'SemiAdder': 1,
     'TemporaryAND': 2}
    >>> drawn = qp.transforms._tape_decompose(circuit, gate_set=gate_set)
    >>> print(qp.draw(drawn, max_length=112, show_matrices=False)(2.4))
     0: ─╭MultiX(M0)─╭X────╭●───────╭●────────────────────────────────────────────────────────────────────────── ···
     1: ─├MultiX(M0)─╰●─╭●─│──╭●─╭●─│─────────────────────────────────────────────────────────────────────────── ···
     2: ─├MultiX(M0)────╰X─├●─│──╰X─╰X─╭●─╭●──────────────────────────────────────────────────────────────────── ···
     3: ─╰●────────────────│──│────────│──│───────────────────────────────────╭●──────────────────────────────── ···
     4: ───────────────────╰⊕─╰X───────├●─│───────────╭●──────────────────────│───────────────────────────────── ···
     5: ───────────────────────────────╰⊕─│───────────│───────────╭●──────────│───────────────────────────────── ···
     6: ─╭MultiX(M1)──────────────────────├MultiX(M0)─├MultiX(M1)─├MultiX(M2)─│───────────╭SemiAdder─╭MultiX(M1) ···
     7: ─├MultiX(M1)──────────────────────├MultiX(M0)─├MultiX(M1)─├MultiX(M2)─│───────────├SemiAdder─├MultiX(M1) ···
     8: ─╰MultiX(M1)──────────────────────╰MultiX(M0)─╰MultiX(M1)─╰MultiX(M2)─│───────────├SemiAdder─╰MultiX(M1) ···
     9: ──────────────────────────────────────────────────────────────────────├MultiX(M0)─├SemiAdder──────────── ···
    10: ──────────────────────────────────────────────────────────────────────├MultiX(M0)─├SemiAdder──────────── ···
    11: ──────────────────────────────────────────────────────────────────────╰MultiX(M0)─├SemiAdder──────────── ···
    12: ──────────────────────────────────────────────────────────────────────────────────├SemiAdder──────────── ···
    13: ──────────────────────────────────────────────────────────────────────────────────╰SemiAdder──────────── ···
    <BLANKLINE>
     0: ··· ─────────────────────────────────────────────────────╭●────────●╮────╭X─╭MultiX(M0)─┤ ╭State
     1: ··· ─────────────────────────────────────────────────────│──╭●─╭●───│─╭●─╰●─├MultiX(M0)─┤ ├State
     2: ··· ─╭●───────────────────────────────────────────────●╮─╰X─╰X─│───●┤─╰X────├MultiX(M0)─┤ ├State
     3: ··· ─│───────────────────────────────────╭●────────────│───────│────│───────╰●──────────┤ ├State
     4: ··· ─│───────────╭●──────────────────────│────────────●┤───────╰X──⊕╯───────────────────┤ ├State
     5: ··· ─│───────────│───────────╭●──────────│────────────⊕╯────────────────────────────────┤ ├State
     6: ··· ─├MultiX(M0)─├MultiX(M1)─├MultiX(M2)─│──────────────────────────────────────────────┤ ├State
     7: ··· ─├MultiX(M0)─├MultiX(M1)─├MultiX(M2)─│──────────────────────────────────────────────┤ ├State
     8: ··· ─╰MultiX(M0)─╰MultiX(M1)─╰MultiX(M2)─│──────────────────────────────────────────────┤ ├State
     9: ··· ─────────────────────────────────────├MultiX(M0)────────────────────────────────────┤ ├State
    10: ··· ─────────────────────────────────────├MultiX(M0)────────────────────────────────────┤ ├State
    11: ··· ─────────────────────────────────────╰MultiX(M0)────────────────────────────────────┤ ├State
    12: ··· ────────────────────────────────────────────────────────────────────────────────────┤ ├State
    13: ··· ────────────────────────────────────────────────────────────────────────────────────┤ ╰State

    The wire ``3`` is XORed into the wires ``0, 1, 2``, and wire ``2`` and the work wire ``4``
    end up holding the Hamming weight of the parities. A :class:`~.TemporaryAND` computes their
    product into the work wire ``5``. The lookup table is loaded into the angle wires
    ``6, 7, 8``, with its constant loaded unconditionally and its other coefficients
    controlled on the wires ``2``, ``4`` and ``5``, see the details on lookup tables below. The
    phase gradient wires ``9, 10, 11`` are flipped if wire ``3`` is in the state
    :math:`|1\rangle`, so that the :class:`~.SemiAdder` subtracts instead of adding. Afterwards,
    everything is uncomputed.

    .. details::
        :title: Lookup tables

        Each addition adds the entry of a lookup table :math:`T` that is selected by :math:`r`
        bits :math:`v_0, \dots, v_{r-1}`, such as the bits of a Hamming weight. Instead of loading
        every entry controlled on its input, the rule writes :math:`T` as an XOR of constant bit
        strings times products of the bits. For example, for :math:`r=2` and the table
        :math:`T = (1, 3, 2, 7)`, indexed by :math:`v_0 + 2 v_1`, we have

        .. math:: T(v_0, v_1) = 1 \oplus 2 v_0 \oplus 3 v_1 \oplus 7 v_0 v_1,

        where :math:`\oplus` is the bitwise XOR. The rule computes the product :math:`v_0 v_1`
        into a work wire with a :class:`~.TemporaryAND`, loads :math:`1` with :class:`~.PauliX`
        gates, and loads the other coefficients with :class:`~.PauliX` gates controlled on
        :math:`v_0`, :math:`v_1` and :math:`v_0 v_1`, respectively. For :math:`r` bits, the
        products of two or more bits take :math:`2^r - r - 1` :class:`~.TemporaryAND` gates.
        If the bits encode a number of at most :math:`n`, such as the Hamming weight of
        :math:`n` wires, the rule skips the products that vanish for all such numbers, which
        leaves :math:`n - r` of them.

    """
    angle_wires, phase_grad_wires, rule_work_wires = validate_phase_gradient_wires(
        angle_wires, phase_grad_wires, work_wires
    )
    registers = (angle_wires, phase_grad_wires, rule_work_wires[: len(angle_wires) - 1])
    spare_work_wires = rule_work_wires[len(angle_wires) - 1 :]
    rz_rule = make_rz_to_phase_gradient_decomp(*registers)

    def _resource_fn(wires, **_):
        """Resources of the rule."""
        if len(wires) == 1:
            return rz_rule.compute_resources(Float, Wire[1]).gate_counts
        block = _equal_angle_block_rep(len(wires), len(angle_wires), len(registers[2]))
        if len(wires) % 2:
            return {block: 1, qp.GlobalPhase: 1}
        return {block: 1}

    def _work_wires_fn(wires, work_wires, **_):
        """Work wires that the rule needs on top of the spare ones and those of the RZLayer."""
        needed = _num_and_work_wires(len(wires))
        return {"zeroed": max(needed - len(spare_work_wires) - len(work_wires), 0)}

    @qp.register_resources(_resource_fn, work_wires=_work_wires_fn, name="phase_gradient")
    def _rzlayer_phase_gradient_decomp(angle, wires, work_wires):
        """Phase gradient decomposition of an :class:`~.RZLayer`."""
        if len(wires) == 1:
            rz_rule(angle, wires[0])
            return
        work_wires = [*work_wires, *spare_work_wires]
        needed = _num_and_work_wires(len(wires))
        with qp.allocate(max(needed - len(work_wires), 0), restored=True) as extra_work_wires:
            _equal_angle_block(angle, wires, [*work_wires, *extra_work_wires], registers)
        if len(wires) % 2:
            qp.GlobalPhase(len(wires) * angle / 2)

    return _rzlayer_phase_gradient_decomp
