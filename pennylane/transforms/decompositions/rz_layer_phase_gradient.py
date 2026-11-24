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

    A table :math:`T` over the Boolean variables :math:`v_0, \dots, v_{r-1}` is the XOR of
    :math:`c_S \prod_{j\in S} v_j` over all subsets :math:`S`, with coefficients :math:`c_S`
    given by the Möbius transform of the table over :math:`\mathbb{Z}_2`, bit by bit. Subsets
    are encoded as masks, ``S = sum(2**j for j in S)``, and inputs as integers,
    ``v = sum(v_j * 2**j)``. That is, ``table[v]`` is the entry for the input ``v``, and the
    monomial of the mask ``S`` is ``1`` at the input ``v`` if ``S & v == S``.

    The transform uses XORs and static loops only, so that ``table`` may be a traced jax array.

    Args:
        table (TensorLike): integer array of length ``2**r``, whose entries are taken modulo
            ``2**num_bits``, so that negative entries are mapped to their two's complement
        num_bits (int): number of bits of the entries

    Returns:
        TensorLike: array of shape ``(2**r, num_bits)``, whose row ``S`` holds the bits of
        :math:`c_S`, most significant bit first as in :func:`~.math.int_to_binary`

    **Example**

    >>> from pennylane.transforms.decompositions.rz_layer_phase_gradient import _anf_bits
    >>> _anf_bits(np.array([1, 3, 2, 7]), num_bits=3)
    array([[0, 0, 1],
           [0, 1, 0],
           [0, 1, 1],
           [1, 1, 1]])

    The coefficients are ``1, 2, 3, 7``. For example, ``table[3] = 1 ^ 2 ^ 3 ^ 7 = 7`` is the XOR
    of all coefficients, because all monomials are ``1`` at the input ``v = 3``.
    """
    rows = list(qp.math.int_to_binary(table, num_bits))
    for j in range(len(rows).bit_length() - 1):
        bit = 1 << j
        rows = [row ^ rows[mask ^ bit] if mask & bit else row for mask, row in enumerate(rows)]
    return qp.math.stack(rows)


def _num_monomials(num_vars):
    """Number of monomials of degree two or higher of ``num_vars`` variables."""
    return 2**num_vars - num_vars - 1


def _monomial_wires(var_wires, mono_wires):
    """The wires holding the monomials of the variables on ``var_wires``, in the order of the rows
    of :func:`~._anf_bits`: ``v0, v1, v0v1, v2, v0v2, v1v2, v0v1v2, ...``. Each variable is
    followed by its products with all monomials before it, which are held by ``mono_wires``."""
    mono_wires = iter(mono_wires)
    monomials = []
    for var in var_wires:
        monomials += [var, *(next(mono_wires) for _ in monomials)]
    return monomials


@subcircuit(wire_argnames=("var_wires", "mono_wires"))
@qp.register_resources(lambda var_wires, **_: {qp.TemporaryAND: _num_monomials(len(var_wires))})
def _Monomials(var_wires, mono_wires):
    r"""Compute the monomials of degree two or higher of the variables on ``var_wires`` into the
    zeroed ``mono_wires``, in the order of :func:`~._monomial_wires`. For three variables:

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
        for i in range(2**j - 1):
            qp.TemporaryAND([monomials[i], var, monomials[2**j + i]])


def _rzlayer_phase_gradient_compute_resources(control_wires, angle_wires, phase_grad_wires, **_):
    """Resources of :func:`~._RZLayerPhaseGradientCompute`."""
    load = qp.MultiX(Bool[len(angle_wires)], Wire[len(angle_wires)])
    flip = qp.MultiX(Bool[len(phase_grad_wires)], Wire[len(phase_grad_wires)])
    resources = Counter({load: 1, qp.ctrl(load, control=Wire[1]): len(control_wires)})
    resources[qp.ctrl(flip, control=Wire[1])] += 1
    return resources


@subcircuit(
    dynamic_argnames=("bits",),
    wire_argnames=("control_wires", "sign_wire", "angle_wires", "phase_grad_wires"),
    arg_specs={"bits": Bool[-1, -1]},
    grad_method=None,
)
@qp.register_resources(_rzlayer_phase_gradient_compute_resources)
def _RZLayerPhaseGradientCompute(bits, control_wires, sign_wire, angle_wires, phase_grad_wires):
    r"""XOR a lookup table into ``angle_wires`` and flip ``phase_grad_wires`` if ``sign_wire`` is
    in the state :math:`|1\rangle`.

    ``bits`` holds the ANF coefficients of the table, see :func:`~._anf_bits`. The constant
    coefficient ``bits[0]`` is loaded unconditionally, and ``bits[i]`` controlled on the monomial
    on ``control_wires[i - 1]``. The bits are big-endian, i.e., ``angle_wires[0]`` receives the
    most significant bit, as in :func:`~.math.binary_decimals` and :class:`~.SemiAdder`.
    """
    qp.MultiX(bits[0], angle_wires)
    for i, wire in enumerate(control_wires):
        qp.ctrl(qp.MultiX(bits[i + 1], angle_wires), control=wire)
    qp.ctrl(qp.MultiX([1] * len(phase_grad_wires), phase_grad_wires), control=sign_wire)


def _signed_add(table, var_wires, mono_wires, sign_wire, registers):
    r"""Add the entry of the lookup table ``table`` at the input on ``var_wires`` to the phase
    gradient register if ``sign_wire`` is in the state :math:`|0\rangle`, and subtract it
    otherwise. ``registers`` holds the angle, phase gradient and adder work wires.

    The table is loaded into the angle wires from its ANF, controlled on the monomials of the
    variables, which are computed into the zeroed ``mono_wires``. The subtraction complements the
    phase gradient register around the addition. The load and the complement are their own
    inverse, because they only target the angle and phase gradient wires, controlled on wires
    that they leave unchanged.

    The integer dtype of ``table`` limits the number of bits: with 32-bit integers, the jax
    default without x64, keep ``len(angle_wires) <= 28``, which leaves headroom for computing the
    tables, or enable x64.
    """
    angle_wires, phase_grad_wires, work_wires = registers
    bits = qp.math.cast(_anf_bits(table, len(angle_wires)), bool)
    control_wires = _monomial_wires(var_wires, mono_wires)
    load = _RZLayerPhaseGradientCompute(
        bits, control_wires, [sign_wire], angle_wires, phase_grad_wires
    )
    add = qp.change_op_basis(load, qp.SemiAdder(angle_wires, phase_grad_wires, work_wires), load)
    return qp.change_op_basis(_Monomials(var_wires, mono_wires), add)


def _signed_add_rep(num_vars, num_bits, num_work_wires):
    """Resource representation of :func:`~._signed_add` for a table over ``num_vars`` variables,
    ``num_bits`` angle wires and ``num_work_wires`` adder work wires."""
    load = _RZLayerPhaseGradientCompute(
        Bool[2**num_vars, num_bits], Wire[2**num_vars - 1], Wire[1], Wire[num_bits], Wire[num_bits]
    )
    target = qp.SemiAdder(Wire[num_bits], Wire[num_bits], Wire[num_work_wires])
    add = ChangeOpBasis2(load, target, load)
    return ChangeOpBasis2(_Monomials(Wire[num_vars], Wire[_num_monomials(num_vars)]), add)


def _angle_ints(angles, num_bits, unit):
    """Integers whose ``num_bits`` bits are the binary decimals of ``angles`` in units of
    ``unit``. They are computed with a dot product, so that ``angles`` may be traced."""
    powers_of_two = 2 ** qp.math.arange(num_bits - 1, -1, -1)
    return qp.math.binary_decimals(angles, num_bits, unit=unit) @ powers_of_two


def _equal_angle_block(theta, wires, work_wires, registers):
    r"""Apply :math:`R_Z(\theta)` to each of an even number :math:`m` of ``wires`` with a single
    :func:`~._signed_add`, with the sign wire :math:`s` = ``wires[-1]``.

    With the parities :math:`y_i = x_i \oplus x_s` of the other wires, the rotations apply the
    phase :math:`\exp(-i\theta z_s (m/2 - h))`, where :math:`z_s = (-1)^{x_s}` and :math:`h` is
    the Hamming weight of the parities. The parities and :math:`h` are computed in place, see
    :func:`~._HammingWeight`, and the lookup table :math:`L(h) = -k (m/2 - h) \bmod 2^b` is added
    to the phase gradient register of :math:`b` wires, where
    :math:`\theta \approx 2\pi k / 2^b`. Each AND of the Hamming weight and of the monomials of
    its bits targets one of the ``work_wires``.
    """
    sign_wire, others = wires[-1], wires[:-1]
    n = len(others)
    num_hw_wires = n - n.bit_count()
    var_wires = [others[-1], *work_wires[: n.bit_length() - 1]]
    mono_wires = work_wires[num_hw_wires : num_hw_wires + _num_monomials(len(var_wires))]
    k = _angle_ints(theta, len(registers[0]), 2 * np.pi)
    table = -k * (len(wires) // 2 - qp.math.arange(2 ** len(var_wires)))
    signed_add = _signed_add(table, var_wires, mono_wires, sign_wire, registers)
    hamming_weight = _HammingWeight(others, work_wires[:num_hw_wires])
    # XOR the sign wire into the other wires, which is its own inverse
    parities = qp.ctrl(qp.MultiX([1] * n, others), control=sign_wire)
    qp.change_op_basis(parities, qp.change_op_basis(hamming_weight, signed_add), parities)


def _equal_angle_block_rep(block_size, num_bits, num_work_wires):
    """Resource representation of :func:`~._equal_angle_block` on ``block_size`` wires, with
    ``num_bits`` angle wires and ``num_work_wires`` adder work wires."""
    n = block_size - 1
    parities = qp.ctrl(qp.MultiX(Bool[n], Wire[n]), control=Wire[1])
    hamming_weight = _HammingWeight(Wire[n], Wire[n - n.bit_count()])
    signed_add = _signed_add_rep(n.bit_length(), num_bits, num_work_wires)
    return ChangeOpBasis2(parities, ChangeOpBasis2(hamming_weight, signed_add), parities)


def _distinct_angle_batch(thetas, wires, work_wires, registers):
    r"""Apply :math:`R_Z(\theta_i)` to each of the ``wires`` with a single
    :func:`~._signed_add`, with the sign wire :math:`s` = ``wires[-1]``.

    With the parities :math:`y_i = x_i \oplus x_s` of the other wires, the rotations apply the
    phase :math:`\exp(-\tfrac{i}{2} z_s (\theta_s + \sum_{i<s} \theta_i (1 - 2 y_i)))`, where
    :math:`z_s = (-1)^{x_s}`. The parities are computed in place, and the lookup table
    :math:`L(y) = -(k_s + \sum_{i<s} k_i (1 - 2 y_i)) \bmod 2^b` is added to the phase gradient
    register of :math:`b` wires, where :math:`\theta_i \approx 4\pi k_i / 2^b`. Each AND of the
    monomials of the parities targets one of the ``work_wires``.
    """
    sign_wire, others = wires[-1], wires[:-1]
    k = _angle_ints(thetas, len(registers[0]), 4 * np.pi)
    inputs = qp.math.arange(2 ** len(others))
    signs = 1 - 2 * ((inputs[:, None] >> qp.math.arange(len(others))) & 1)
    table = -(k[-1] + signs @ k[:-1])
    mono_wires = work_wires[: _num_monomials(len(others))]
    signed_add = _signed_add(table, others, mono_wires, sign_wire, registers)
    if others:
        parities = qp.ctrl(qp.MultiX([1] * len(others), others), control=sign_wire)
        qp.change_op_basis(parities, signed_add, parities)


def _distinct_angle_batch_rep(size, num_bits, num_work_wires):
    """Resource representation of :func:`~._distinct_angle_batch` on ``size`` wires, with
    ``num_bits`` angle wires and ``num_work_wires`` adder work wires."""
    signed_add = _signed_add_rep(size - 1, num_bits, num_work_wires)
    if size == 1:
        return signed_add
    parities = qp.ctrl(qp.MultiX(Bool[size - 1], Wire[size - 1]), control=Wire[1])
    return ChangeOpBasis2(parities, signed_add, parities)


def _num_and_work_wires(num_angles, block_size, batch_size):
    """Number of work wires that the TemporaryAND gates of the phase gradient rule use on top of
    the adder."""
    if block_size % 2 == 0:
        n = block_size - 1
        return n - n.bit_count() + _num_monomials(n.bit_length())
    if block_size == 1 and batch_size is not None:
        return _num_monomials(min(batch_size, num_angles) - 1)
    return 0


def make_rzlayer_to_phase_gradient_decomp(
    angle_wires, phase_grad_wires, work_wires, batch_size=None
):
    r"""
    Create a custom decomposition rule for :class:`~.RZLayer` operators, in which rotations by
    equal angles or batches of rotations share one :class:`~.SemiAdder` of the
    `phase gradient decomposition <https://pennylane.ai/compilation/phase-gradient/b-rotations>`__.

    As for :func:`~.make_rz_to_phase_gradient_decomp`, the auxiliary wires of the phase gradient
    decomposition are taken as inputs and bound into the created decomposition rule, which can
    then be registered with :func:`~pennylane.decomposition.add_decomps`. The rule decomposes an
    :class:`~.RZLayer` with :math:`k` angles on :math:`km` wires depending on the block size
    :math:`m` of wires per angle:

    * **Even m.** The :math:`m` rotations by an equal angle use one addition. They apply a
      multiple of the angle that only depends on the parities of the first :math:`m-1` wires with
      the last one, through their Hamming weight. The rule computes the Hamming weight in place,
      loads the multiple from a lookup table into the angle wires, and adds or subtracts it,
      depending on the last wire. On top of the adder, this uses :math:`(m-1) -
      \operatorname{popcount}(m-1) + 2^r - r - 1` :class:`~.TemporaryAND` gates, with :math:`r`
      the number of bits of :math:`m-1`, i.e., 0, 2 and 7 for :math:`m = 2, 4, 6`.
    * **m = 1 with batch_size.** Consecutive batches of :math:`s` = ``batch_size`` rotations with
      distinct angles use one addition, from a lookup table over the parities of the first
      :math:`s-1` wires with the last one. On top of the adder, this uses :math:`2^{s-1} - s`
      :class:`~.TemporaryAND` gates, i.e., 0, 1 and 4 for :math:`s = 2, 3, 4`. The last batch is
      smaller if ``batch_size`` does not divide :math:`k`.
    * **Otherwise**, i.e., for odd :math:`m > 1` and for :math:`m=1` without ``batch_size``,
      each rotation uses the circuit of :func:`~.make_rz_to_phase_gradient_decomp`, including its
      :class:`~.GlobalPhase`.

    Each of these :class:`~.TemporaryAND` gates targets a work wire, which is reused for every
    block and batch. The rule takes these work wires from the ``work_wires`` that the adder does
    not need, together with the ``work_wires`` of the :class:`~.RZLayer`. If there are too few,
    it allocates the missing ones dynamically, which excludes it from compilation with Catalyst.

    Parameters:
        angle_wires (Wires): wires that encode the binary representation of the rotation angles
        phase_grad_wires (Wires): wires that carry a phase gradient state. Should have the same
            length as ``angle_wires``.
        work_wires (Wires): work wires for the :class:`~.SemiAdder` decomposition, of which at
            least ``len(angle_wires) - 1`` are required. Any further ones are used by the
            :class:`~.TemporaryAND` gates described above.
        batch_size (int | None): Number of rotations with distinct angles that share one adder
            if the block size is :math:`m=1`. If given, the registers hold one more wire than the
            other rotations use, see below. Defaults to ``None``, which applies the rotations
            individually.

    Returns:
        qp.decomposition.DecompositionRule: decomposition rule to be used within
        :func:`~.pennylane.decompose`.

    .. seealso:: :func:`~.make_rz_to_phase_gradient_decomp`

    **Example**

    We decompose four rotations by an equal angle, with two work wires. They use one
    :class:`~.SemiAdder` and two :class:`~.TemporaryAND` gates:

    .. code-block:: python

        import pennylane as qp
        from pennylane.transforms.decompositions import make_rzlayer_to_phase_gradient_decomp

        custom_decomp = make_rzlayer_to_phase_gradient_decomp(
            angle_wires=[6, 7, 8], phase_grad_wires=[9, 10, 11], work_wires=[12, 13]
        )
        gate_set = {
            "CNOT", "MultiX", "C(MultiX)", "TemporaryAND", "Adjoint(TemporaryAND)", "SemiAdder"
        }

        @qp.transforms._tape_decompose(
            gate_set=gate_set, fixed_decomps={qp.RZLayer: custom_decomp}
        )
        @qp.qnode(qp.device("null.qubit"))
        def circuit(theta):
            qp.RZLayer([theta], wires=[0, 1, 2, 3], work_wires=[4, 5])
            return qp.state()

    >>> tape = qp.workflow.construct_tape(circuit)(0.5)
    >>> qp.resource.resources_from_tape(tape).counts
    {'C(MultiX)': 10, 'CNOT': 10, 'TemporaryAND': 2, 'MultiX': 2, 'SemiAdder': 1,
     'Adjoint(TemporaryAND)': 2}
    >>> print(qp.draw(circuit, max_length=88)(0.5))
     0: ─╭MultiX(M0)─╭X────╭●───────╭●─────────────────────────────────────── ···
     1: ─├MultiX(M0)─╰●─╭●─│──╭●─╭●─│──────────────────────────────────────── ···
     2: ─├MultiX(M0)────╰X─├●─│──╰X─╰X─╭●─╭●───────────────────────────────── ···
     3: ─╰●────────────────│──│────────│──│────────────────────────────────── ···
     4: ───────────────────╰⊕─╰X───────├●─│───────────╭●───────────────────── ···
     5: ───────────────────────────────╰⊕─│───────────│───────────╭●───────── ···
     6: ─╭MultiX(M1)──────────────────────├MultiX(M2)─├MultiX(M1)─├MultiX(M3) ···
     7: ─├MultiX(M1)──────────────────────├MultiX(M2)─├MultiX(M1)─├MultiX(M3) ···
     8: ─╰MultiX(M1)──────────────────────╰MultiX(M2)─╰MultiX(M1)─╰MultiX(M3) ···
     9: ───────────────────────────────────────────────────────────────────── ···
    10: ───────────────────────────────────────────────────────────────────── ···
    11: ───────────────────────────────────────────────────────────────────── ···
    12: ───────────────────────────────────────────────────────────────────── ···
    13: ───────────────────────────────────────────────────────────────────── ···
    <BLANKLINE>
     0: ··· ─────────────────────────────────────────────────────────────────────── ···
     1: ··· ─────────────────────────────────────────────────────────────────────── ···
     2: ··· ────────────────────────────────────╭●───────────────────────────────── ···
     3: ··· ─╭●─────────────────────────────────│────────────────────────────────── ···
     4: ··· ─│──────────────────────────────────│───────────╭●───────────────────── ···
     5: ··· ─│──────────────────────────────────│───────────│───────────╭●───────── ···
     6: ··· ─│───────────╭SemiAdder─╭MultiX(M1)─├MultiX(M2)─├MultiX(M1)─├MultiX(M3) ···
     7: ··· ─│───────────├SemiAdder─├MultiX(M1)─├MultiX(M2)─├MultiX(M1)─├MultiX(M3) ···
     8: ··· ─│───────────├SemiAdder─╰MultiX(M1)─╰MultiX(M2)─╰MultiX(M1)─╰MultiX(M3) ···
     9: ··· ─├MultiX(M0)─├SemiAdder──────────────────────────────────────────────── ···
    10: ··· ─├MultiX(M0)─├SemiAdder──────────────────────────────────────────────── ···
    11: ··· ─╰MultiX(M0)─├SemiAdder──────────────────────────────────────────────── ···
    12: ··· ─────────────├SemiAdder──────────────────────────────────────────────── ···
    13: ··· ─────────────╰SemiAdder──────────────────────────────────────────────── ···
    <BLANKLINE>
     0: ··· ─────────────────╭●────────●╮────╭X─╭MultiX(M0)─┤ ╭State
     1: ··· ─────────────────│──╭●─╭●───│─╭●─╰●─├MultiX(M0)─┤ ├State
     2: ··· ──────────────●╮─╰X─╰X─│───●┤─╰X────├MultiX(M0)─┤ ├State
     3: ··· ─╭●────────────│───────│────│───────╰●──────────┤ ├State
     4: ··· ─│────────────●┤───────╰X──⊕╯───────────────────┤ ├State
     5: ··· ─│────────────⊕╯────────────────────────────────┤ ├State
     6: ··· ─│──────────────────────────────────────────────┤ ├State
     7: ··· ─│──────────────────────────────────────────────┤ ├State
     8: ··· ─│──────────────────────────────────────────────┤ ├State
     9: ··· ─├MultiX(M0)────────────────────────────────────┤ ├State
    10: ··· ─├MultiX(M0)────────────────────────────────────┤ ├State
    11: ··· ─╰MultiX(M0)────────────────────────────────────┤ ├State
    12: ··· ────────────────────────────────────────────────┤ ├State
    13: ··· ────────────────────────────────────────────────┤ ╰State
    <BLANKLINE>
    M0 =
    [ True  True  True]
    M1 =
    [ True  True False]
    M2 =
    [False False  True]
    M3 =
    [False False False]

    The wire ``3`` is XORed into the wires ``0, 1, 2``, and wire ``2`` and the work wire ``4``
    end up holding the Hamming weight of the parities. A :class:`~.TemporaryAND` computes their
    product into the work wire ``5``. The lookup table is loaded into the angle wires
    ``6, 7, 8``, with its constant ``M1`` loaded unconditionally and its other coefficients
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

    .. details::
        :title: Precision

        Without ``batch_size``, every rotation has the precision of
        :func:`~.make_rz_to_phase_gradient_decomp` with the same registers: its angle is rounded
        to :math:`b` = ``len(angle_wires)`` binary decimals in units of :math:`2\pi`.

        With ``batch_size``, ``angle_wires`` and ``phase_grad_wires`` hold :math:`b+1` wires. The
        batches of distinct angles use all of them and round the angles to :math:`b+1` binary
        decimals in units of :math:`4\pi`, i.e., with the same precision, which costs one more
        :class:`~.TemporaryAND` in each adder. All other rotations only use the leading :math:`b`
        wires of the registers, as above.

        In both cases, the rotations by equal angles and the batches apply the product of
        :class:`~.RZ` rotations by the rounded angles exactly, without a global phase.
    """
    angle_wires, phase_grad_wires, rule_work_wires = validate_phase_gradient_wires(
        angle_wires, phase_grad_wires, work_wires
    )
    num_bits = len(angle_wires) - (batch_size is not None)
    rz_registers = (
        angle_wires[:num_bits],
        phase_grad_wires[:num_bits],
        rule_work_wires[: num_bits - 1],
    )
    batch_registers = (angle_wires, phase_grad_wires, rule_work_wires[: len(angle_wires) - 1])
    rz_rule = make_rz_to_phase_gradient_decomp(*rz_registers)

    def _spare_work_wires(block_size):
        """The ``work_wires`` of the rule that the adder does not need."""
        registers = batch_registers if block_size == 1 and batch_size is not None else rz_registers
        return rule_work_wires[len(registers[2]) :]

    def _resource_fn(angles, wires, **_):
        """Resources of the rule."""
        num_angles = len(angles)
        block_size = len(wires) // num_angles
        if block_size % 2 == 0:
            block = _equal_angle_block_rep(block_size, num_bits, len(rz_registers[2]))
            return {block: num_angles}
        if block_size == 1 and batch_size is not None:
            batch_args = (len(angle_wires), len(batch_registers[2]))
            reps = {_distinct_angle_batch_rep(batch_size, *batch_args): num_angles // batch_size}
            if num_angles % batch_size:
                reps[_distinct_angle_batch_rep(num_angles % batch_size, *batch_args)] = 1
            return reps
        rz_resources = rz_rule.compute_resources(Float, Wire[1]).gate_counts
        return {op: count * len(wires) for op, count in rz_resources.items()}

    def _work_wires_fn(angles, wires, work_wires):
        """Work wires that the rule needs on top of the spare ones and those of the RZLayer."""
        num_angles = len(angles)
        block_size = len(wires) // num_angles
        needed = _num_and_work_wires(num_angles, block_size, batch_size)
        available = len(_spare_work_wires(block_size)) + len(work_wires)
        return {"zeroed": max(needed - available, 0)}

    def _equal_angle_blocks(angles, wires, work_wires):
        """Apply the blocks of rotations by an equal angle."""
        block_size = len(wires) // len(angles)

        def _block(i):
            block = [wires[i * block_size + j] for j in range(block_size)]
            _equal_angle_block(angles[i], block, work_wires, rz_registers)

        qp.for_loop(len(angles))(_block)()

    def _distinct_batches(angles, wires, work_wires):
        """Apply the batches of rotations by distinct angles."""

        def _batch(start, size):
            batch = [wires[start + j] for j in range(size)]
            thetas = qp.math.stack([angles[start + j] for j in range(size)])
            _distinct_angle_batch(thetas, batch, work_wires, batch_registers)

        num_batches, rest = divmod(len(angles), batch_size)
        qp.for_loop(num_batches)(lambda i: _batch(i * batch_size, batch_size))()
        if rest:
            _batch(num_batches * batch_size, rest)

    def _rotations(angles, wires):
        """Apply the rotations individually."""
        block_size = len(wires) // len(angles)
        qp.for_loop(len(wires))(lambda i: rz_rule(angles[i // block_size], wires[i]))()

    @qp.register_resources(_resource_fn, work_wires=_work_wires_fn)
    def _rzlayer_phase_gradient_decomp(angles, wires, work_wires):
        """Phase gradient decomposition of an :class:`~.RZLayer`."""
        num_angles = len(angles)
        block_size = len(wires) // num_angles
        if qp.capture.enabled() or qp.compiler.active():
            angles = qp.math.array(angles, like="jax")
            wires = qp.math.array(wires, like="jax")

        work_wires = [*work_wires, *_spare_work_wires(block_size)]
        needed = _num_and_work_wires(num_angles, block_size, batch_size)
        with qp.allocate(max(needed - len(work_wires), 0), restored=True) as extra_work_wires:
            work_wires = [*work_wires, *extra_work_wires]
            if block_size % 2 == 0:
                _equal_angle_blocks(angles, wires, work_wires)
            elif block_size == 1 and batch_size is not None:
                _distinct_batches(angles, wires, work_wires)
            else:
                _rotations(angles, wires)

    return _rzlayer_phase_gradient_decomp
