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

"""Built-in decomposition rules for templates.subroutines.time_evolution.trotter_vibronic."""

from collections import defaultdict

import numpy as np

from pennylane import capture, compiler, math
from pennylane.allocation import allocate
from pennylane.capture import hint
from pennylane.control_flow import for_loop
from pennylane.decomposition import add_decomps, register_resources
from pennylane.decomposition_rules.templates.subroutines.arithmetic.signed_out_multiplier import (
    _twos_complement_helper,
)
from pennylane.ops import CNOT, Hadamard, adjoint, cond, ctrl
from pennylane.templates.subroutines.aqft import AQFT
from pennylane.templates.subroutines.arithmetic.incrementer import Incrementer
from pennylane.templates.subroutines.arithmetic.out_multiplier import OutMultiplier
from pennylane.templates.subroutines.arithmetic.semi_adder import SemiAdder
from pennylane.templates.subroutines.arithmetic.signed_out_multiplier import SignedOutMultiplier
from pennylane.templates.subroutines.arithmetic.signed_out_square import SignedOutSquare
from pennylane.templates.subroutines.multix import MultiX
from pennylane.templates.subroutines.qrom import QROM
from pennylane.templates.subroutines.time_evolution.trotter_vibronic import (
    TrotterVibronic,
    _momentum_coefficients,
    _preprocess_data,
)
from pennylane.typing import Wire


# pylint: disable-next=too-many-arguments
def _trotter_vibronic_resources(
    evolution_time,
    num_trotter_steps,
    hamiltonian,
    electronic_wires,
    vib_wires,
    coefficient_wires,
    phase_gradient_wires,
    cache_wires,
    work_wires,
    aqft_order,
    branch_probs,
):
    """Gate counts for the vibronic Trotter circuit.

    Each term is counted as often as it is expected to be applied according to ``branch_probs``,
    or as often as it is visited if its probability is unknown. This estimate is inexact
    (``exact=False``): the expected counts are rounded up, the electronic diagonalization is
    counted as an upper bound, and the sub-operations are counted at their top level.
    """
    # ``evolution_time`` and ``cache_wires`` are part of the shared resource/decomposition
    # signature but do not affect the gate counts.
    # pylint: disable=unused-argument
    num_fragments = hamiltonian.num_fragments
    n_states = hamiltonian.num_states
    n_modes = hamiltonian.num_modes
    n_elec = len(electronic_wires)
    # ``coefficient_wires`` may be dynamically allocated (empty here); fall back to
    # ``len(phase_gradient_wires)`` so the sub-operations are counted consistently.
    b = _coefficient_register_size(coefficient_wires, phase_gradient_wires)
    n_pg = len(phase_gradient_wires)
    k = len(vib_wires) // n_modes
    # ``work_wires`` may be dynamically allocated (empty here), in which case fall back to the
    # required size so the sub-operations are counted consistently with the decomposition.
    n_work = len(work_wires) if len(work_wires) > 0 else max(n_elec - 1, 2 * k, 2 * b + 2)
    num_pairs = n_modes * (n_modes - 1) // 2

    # Each Trotter step visits every position fragment twice (forward + backward).
    position_visits = 2 * num_fragments * num_trotter_steps
    num_constant = _num_applied_terms(branch_probs, "constant", position_visits)
    num_linear = _num_applied_terms(branch_probs, "linear", position_visits * n_modes)
    num_quadratic = _num_applied_terms(branch_probs, "quadratic", position_visits * n_modes)
    num_bilinear = _num_applied_terms(branch_probs, "bilinear", position_visits * num_pairs)
    num_kinetic = _num_applied_terms(branch_probs, "kinetic", num_trotter_steps * n_modes)

    # Data loading (QROM) and the arithmetic primitives, sized to match ``_extract_registers``.
    # ``Wire[size]`` builds an abstract wire register of the given size; only the sizes matter
    # for resource counting, so registers of the same operator need not be kept disjoint.
    qrom = QROM(
        np.zeros((n_states, b), dtype=int),
        control_wires=Wire[n_elec],
        target_wires=Wire[b],
        work_wires=Wire[max(n_elec - 1, 0)],
    )
    semi_adder = SemiAdder(x_wires=Wire[b], y_wires=Wire[n_pg], work_wires=Wire[n_work])

    # ``OutMultiplier`` appears with three different ``y`` register sizes: inside the half-signed
    # multiplier for the linear terms (``y = k``, the mode register) and the bilinear terms
    # (``y = 2k``, the cache register), and directly for the quadratic/kinetic terms
    # (``y = 2k - 1``, the cache register minus its sign bit).
    out_mult_linear = OutMultiplier(
        x_wires=Wire[b],
        y_wires=Wire[k],
        output_wires=Wire[n_pg],
        work_wires=Wire[max(n_work - 1, 0)],
    )
    out_mult_bilinear = OutMultiplier(
        x_wires=Wire[b],
        y_wires=Wire[2 * k],
        output_wires=Wire[n_pg],
        work_wires=Wire[max(n_work - 1, 0)],
    )
    out_mult_quad = OutMultiplier(
        x_wires=Wire[b],
        y_wires=Wire[max(2 * k - 1, 1)],
        output_wires=Wire[n_pg],
        work_wires=Wire[n_work],
    )
    signed_out_mult = SignedOutMultiplier(
        x_wires=Wire[k],
        y_wires=Wire[k],
        output_wires=Wire[2 * k],
        work_wires=Wire[n_work],
        output_wires_zeroed=True,
    )
    signed_square = SignedOutSquare(
        x_wires=Wire[k],
        output_wires=Wire[max(2 * k - 1, 1)],
        work_wires=Wire[n_work],
        output_wires_zeroed=True,
    )
    # Mirror the decomposition's resolution of ``aqft_order`` (``k - 1`` when unset) so the
    # resource estimate and the emitted circuit agree.
    aqft = AQFT(order=(aqft_order if aqft_order is not None else k - 1), wires=Wire[k])

    # The compute/uncompute pairs are emitted as ``adjoint(...)`` wrappers in the decomposition, so
    # they must be declared as such (not as their bare base ops) for the graph to find a path.
    adj_signed_square = adjoint(signed_square)
    adj_signed_out_mult = adjoint(signed_out_mult)
    adj_aqft = adjoint(aqft)

    # The sign-bit-controlled two's complement (compute + uncompute) acts on the ``k``-wire mode
    # register (linear terms) and the ``2k``-wire cache register (bilinear terms).
    ctrl_incrementer_linear = ctrl(Incrementer(Wire[k], Wire[max(n_work - 1, 0)]), Wire[1])
    ctrl_incrementer_bilinear = ctrl(Incrementer(Wire[2 * k], Wire[max(n_work - 1, 0)]), Wire[1])

    resources = defaultdict(int)

    # -- Electronic diagonalization (forward + adjoint per visit) --
    resources[Hadamard(wires=Wire[1])] += 2 * position_visits

    # -- CNOTs: electronic diagonalization plus the half-signed multipliers, which cache the sign
    #    bit, invert the register for the two's complement, and apply the controlled output flips --
    resources[CNOT(wires=Wire[2])] += (
        2 * position_visits * max(n_elec - 1, 0)
        + num_linear * (2 + 2 * k + 2 * n_pg)
        + num_bilinear * (2 + 4 * k + 2 * n_pg)
    )

    # -- Data loading: each applied constant/linear/quadratic/bilinear term + final unload --
    resources[qrom] += num_constant + num_linear + num_quadratic + num_bilinear + position_visits
    resources[semi_adder] += num_constant

    # -- Linear terms: one half-signed multiplier each --
    resources[out_mult_linear] += num_linear
    resources[ctrl_incrementer_linear] += num_linear * 2

    # -- Quadratic terms: two SignedOutSquares (compute + uncompute) and one OutMultiplier --
    resources[signed_square] += num_quadratic
    resources[adj_signed_square] += num_quadratic
    resources[out_mult_quad] += num_quadratic

    # -- Bilinear terms: two SignedOutMultipliers (compute + uncompute) and one half-signed
    #    multiplier --
    resources[signed_out_mult] += num_bilinear
    resources[adj_signed_out_mult] += num_bilinear
    resources[out_mult_bilinear] += num_bilinear
    resources[ctrl_incrementer_bilinear] += num_bilinear * 2

    # -- Kinetic fragment (once per Trotter step): per mode, load the momentum coefficients as a
    #    basis state (``MultiX``), then AQFT + SignedOutSquare + OutMultiplier and their
    #    uncomputation --
    multi_x = MultiX(np.zeros(b, dtype=bool), wires=Wire[b])
    resources[multi_x] += num_kinetic * 2
    resources[aqft] += num_kinetic
    resources[adj_aqft] += num_kinetic
    resources[signed_square] += num_kinetic
    resources[adj_signed_square] += num_kinetic
    resources[out_mult_quad] += num_kinetic

    return dict(resources)


def _num_applied_terms(branch_probs, term, num_visits):
    """Number of applications of ``term`` out of ``num_visits``, weighted with its probability in
    ``branch_probs`` and rounded up.

    A visited term is counted at least once, because the resources must contain the gates in its
    conditional branch, which is traced under capture even if it is never taken.
    """
    prob = dict(branch_probs or ()).get(term)
    if prob is None or num_visits == 0:
        return num_visits
    # Rounding first keeps float errors in exact products, e.g. ``6 * (2 / 3)``, from rounding up
    return max(int(np.ceil(round(prob * num_visits, 9))), 1)


def _coefficient_register_size(coefficient_wires, phase_gradient_wires):
    """Size ``b`` of the coefficient register: its own length, or ``len(phase_gradient_wires)``
    when dynamically allocated (both hold ``b``-bit precision data, see the class docstring)."""
    return len(coefficient_wires) if len(coefficient_wires) > 0 else len(phase_gradient_wires)


def _required_work_wire_sizes(hamiltonian, vib_wires, b):
    """Required ``(cache, work)`` register sizes for the vibronic Trotter circuit."""
    n_states = hamiltonian.num_states
    n_modes = hamiltonian.num_modes
    k = len(vib_wires) // n_modes
    n = math.ceil_log2(n_states)
    return 2 * k, max(n - 1, 2 * k, 2 * b + 2)


# pylint: disable-next=too-many-arguments
def _trotter_vibronic_work_wires(
    hamiltonian, vib_wires, coefficient_wires, phase_gradient_wires, cache_wires, work_wires, **_
):
    """Number of zeroed work wires allocated when ``coefficient_wires``/``cache_wires``/
    ``work_wires`` are omitted."""
    b = _coefficient_register_size(coefficient_wires, phase_gradient_wires)
    cache_size, work_size = _required_work_wire_sizes(hamiltonian, vib_wires, b)
    num_alloc = 0
    if len(coefficient_wires) == 0:
        num_alloc += b
    if len(cache_wires) == 0:
        num_alloc += cache_size
    if len(work_wires) == 0:
        num_alloc += work_size
    return {"zeroed": num_alloc}


@register_resources(
    _trotter_vibronic_resources, exact=False, work_wires=_trotter_vibronic_work_wires
)
# pylint: disable-next=too-many-arguments
def _trotter_vibronic_decomposition(
    evolution_time,
    num_trotter_steps,
    hamiltonian,
    electronic_wires,
    vib_wires,
    coefficient_wires,
    phase_gradient_wires,
    cache_wires,
    work_wires,
    aqft_order,
    branch_probs,
):
    n_states = hamiltonian.num_states
    n_modes = hamiltonian.num_modes
    n_elec = len(electronic_wires)

    vib = list(vib_wires)
    # Reshape flat ``vib_wires`` into per-mode registers for dynamic-index access.
    if compiler.active() or capture.enabled():
        mode_registers = math.array(vib, like="jax").reshape(n_modes, -1)
    else:
        mode_registers = np.array(vib).reshape(n_modes, -1)

    resolved_aqft_order = aqft_order if aqft_order is not None else mode_registers.shape[1] - 1

    def _run(coefficients, cache, work):
        registers = {
            "electronic": electronic_wires,
            "cache": cache,
            "coefficients": coefficients,
            "phase_gradient": phase_gradient_wires,
            "work": work,
        }
        _validate_registers(registers, mode_registers, n_modes, n_states)
        _run_trotter_vibronic(
            evolution_time,
            num_trotter_steps,
            hamiltonian,
            registers,
            mode_registers,
            resolved_aqft_order,
            n_states,
            n_modes,
            n_elec,
            branch_probs,
        )

    # Dynamically allocate ``coefficient_wires``/``cache_wires``/``work_wires`` when omitted.
    b = _coefficient_register_size(coefficient_wires, phase_gradient_wires)
    cache_size, work_size = _required_work_wire_sizes(hamiltonian, vib_wires, b)
    coefficients = coefficient_wires
    cache = cache_wires
    work = work_wires
    need_coefficients = len(coefficients) == 0
    need_cache = len(cache) == 0
    need_work = len(work) == 0
    num_alloc = (
        (b if need_coefficients else 0)
        + (cache_size if need_cache else 0)
        + (work_size if need_work else 0)
    )

    with allocate(num_alloc, state="zero", restored=True) as allocated:
        start = 0
        if need_coefficients:
            coefficients = allocated[start : (start := start + b)]
        if need_cache:
            cache = allocated[start : (start := start + cache_size)]
        if need_work:
            work = allocated[start : (start := start + work_size)]
        _run(coefficients, cache, work)


add_decomps(TrotterVibronic, _trotter_vibronic_decomposition)


def _diagonalize_probs(num_fragments, n):
    """Compute the probabilities for the outer loop and the CNOT loop in
    ``_diagonalize_vibronic_circuit``."""
    # at least a Hadamard is applied for all but the first fragment
    apply_anything_prob = (num_fragments - 1) / num_fragments
    # CNOTs are applied whenever more than one bit is set in the fragment index, overall we compute
    # all set bits and subtract `num_fragments-1`, the number of "first set bits":
    num_cnots = np.sum(np.bitwise_count(np.arange(num_fragments))) - (num_fragments - 1)
    # For the probability we have to normalize by the inner loop size (n) and the effective outer
    # loop size (num_fragments - 1)
    cnot_prob = num_cnots / (n * (num_fragments - 1))
    return apply_anything_prob, cnot_prob


def _diagonalize_vibronic_circuit(fragment_idx, wires, num_fragments, inverse=False):
    r"""Diagonalize position fragment ``fragment_idx`` by applying Clifford operations on the
    electronic register, following the ``key = (0, fragment_idx)`` convention (see
    :func:`_diagonalization_matrix`).

    Based on Fig. 2 of `Motlagh et al, arXiv:2411.13669 <https://arxiv.org/abs/2411.13669>`__.
    Requires one :class:`~.Hadamard` and at most :math:`\lceil\log_2(N)\rceil - 1`
    :class:`~.CNOT` gates, where :math:`N` is the number of electronic states.

    ``fragment_idx`` may be a traced integer, so the circuit structure (whether any gate is
    applied, which wire is the control, which wires receive a CNOT) is expressed with
    ``cond``/``for_loop`` rather than Python-level branching.

    Args:
        fragment_idx (int): index of the position fragment. The circuit is the identity if
            ``fragment_idx == 0``.
        wires (WiresLike): electronic wires on which the fragment acts.
        inverse (bool): Apply the inverse gate sequence.
        num_fragments (int): total number of position fragments. The conditionals carry branch
            probability hints that assume ``fragment_idx`` to be uniformly distributed over
            ``range(num_fragments)``.
    """
    if num_fragments == 1:
        # Only have fragment_idx 0, which does not need a circuit.
        return
    n = len(wires)

    # Compute resource hints probabilities
    apply_anything_prob, cnot_prob = _diagonalize_probs(num_fragments, n)

    # ``key = (0, fragment_idx)``, so the bits that differ between ``key[0]`` and ``key[1]``
    # (see ``_diagonalization_support``) are exactly the bits of ``fragment_idx`` itself.
    shifts = np.arange(n - 1, -1, -1)
    if compiler.active() or capture.enabled():
        wires_arr = math.array(list(wires), like="jax")
        shifts = math.array(shifts, like="jax")
    else:
        wires_arr = list(wires)
    diff = (fragment_idx >> shifts) & 1
    any_diff = math.sum(diff) > 0
    # Largest wire index with a set bit == least-significant differing bit, since wire 0 holds
    # the most significant bit (matching ``qp.math.int_to_binary``).
    control_pos = math.max(math.where(diff > 0, math.arange(n), -1))
    control_wire = wires_arr[control_pos]

    def _cnots(start, stop, step):
        @for_loop(start, stop, step)
        def loop(w):
            @cond((diff[w] > 0) & (w != control_pos))
            @hint({"branch-prob": cnot_prob})
            def _maybe_cnot():
                CNOT(wires=[control_wire, wires_arr[w]])

            _maybe_cnot()

        loop()  # pylint: disable=no-value-for-parameter

    @cond(any_diff)
    @hint({"branch-prob": apply_anything_prob})
    def _apply():
        Hadamard(wires=[control_wire])
        _cnots(n - 1, -1, -1)

    @cond(any_diff)
    @hint({"branch-prob": apply_anything_prob})
    def _adjoint_apply():
        _cnots(0, n, 1)
        Hadamard(wires=[control_wire])

    if inverse:
        _adjoint_apply()
    else:
        _apply()


def _half_signed_out_multiplier(x_wires, y_wires, output_wires, work_wires):
    r"""Out-of-place multiplier of an unsigned register ``x`` and a signed register ``y``.

    Computes :math:`|z\rangle \mapsto |(z + xy) \bmod 2^k\rangle` on ``output_wires``, where
    ``k = len(output_wires)``, ``x`` is an unsigned integer and ``y`` is a two's-complement
    signed integer. This is a specialized primitive used by :class:`~.TrotterVibronic`; it is
    a plain quantum function that queues its constituent operations.

    The sign bit of ``y`` is cached on the first work wire, the two's complement of ``y`` is
    computed controlled on that cached bit, the (unsigned) magnitudes are multiplied into the
    output register with a :class:`~.OutMultiplier` (with the output wires flipped before and
    after the multiplication if ``y`` is negative), and finally the two's complement and the
    cached sign bit are uncomputed.
    """
    y_aux, work_wires = work_wires[0], work_wires[1:]

    if compiler.active() or capture.enabled():
        output_arr = math.array(list(output_wires), like="jax")
    else:
        output_arr = list(output_wires)

    def _flip_outputs():
        # CNOT from the cached sign bit onto every output wire.
        @for_loop(len(output_wires))
        def _flip(w):
            CNOT([y_aux, output_arr[w]])

        # pylint: disable-next=no-value-for-parameter
        _flip()

    # Cache the sign bit of ``y`` on the auxiliary wire.
    CNOT([y_wires[0], y_aux])

    # Take the two's complement of ``y`` controlled on the cached sign bit.
    _twos_complement_helper(y_wires, y_aux, work_wires)

    # Multiply the magnitudes into the output register. If ``y`` was negative, flip all output
    # qubits before and after the (unsigned) multiplication, effectively subtracting the product.
    _flip_outputs()
    OutMultiplier(x_wires, y_wires, output_wires, work_wires=work_wires, output_wires_zeroed=False)
    _flip_outputs()

    # Uncompute the two's complement and the cached sign bit.
    _twos_complement_helper(y_wires, y_aux, work_wires)
    CNOT([y_wires[0], y_aux])


def _load_coefficients(coefficients, precision, prev_bitstrings, qrom_wires):
    """Load ``coefficients`` into the data-loading register via a differential :class:`~.QROM`.

    The coefficients are converted into ``precision``-bit strings, XOR-ed with the currently
    loaded ``prev_bitstrings`` reference point, and loaded with a :class:`~.QROM`. The newly
    loaded bit strings are returned so that they can be used as the reference point for the next
    data-loading step.

    The coefficients are negated before encoding: adding an integer into the standard
    phase-gradient state :math:`|\\nabla n\\rangle` imprints :math:`e^{+2\\pi i m/2^b}`, so the
    negation is what turns the accumulated phase into :math:`e^{-i\\theta}` (i.e. :math:`e^{-iHt}`).
    """
    new_bitstrings = math.binary_decimals(-coefficients, precision, unit=2 * np.pi)
    change_bitstrings = (prev_bitstrings + new_bitstrings) % 2
    QROM(change_bitstrings, **qrom_wires)
    return new_bitstrings


def _extract_registers(registers, mode_registers, term, *mode_ids):
    r"""Extract the wire registers required for a specific term of the Trotter step.

    See :class:`~.TrotterVibronic` for the meaning of the individual registers. ``term`` is one
    of ``"constant"``, ``"linear"``, ``"quadratic"``, ``"bilinear"`` or ``"QROM"``, and
    ``mode_ids`` are the indices of the mode(s) involved in the term.
    """
    if term == "quadratic":
        (k,) = mode_ids
        square_wires = {"output_wires": "cache", "work_wires": "work"}
        square_wires = {new: registers[old] for new, old in square_wires.items()}
        square_wires["x_wires"] = mode_registers[k]
        # The cache contains 2k wires, we just need 2k-1 here
        square_wires["output_wires"] = square_wires["output_wires"][1:]
        mult_wires = {
            "x_wires": "coefficients",
            "y_wires": "cache",
            "output_wires": "phase_gradient",
            "work_wires": "work",
        }
        mult_wires = {new: registers[old] for new, old in mult_wires.items()}
        # The cache contains 2k wires, we just need 2k-1 here, see above
        mult_wires["y_wires"] = mult_wires["y_wires"][1:]
        return square_wires, mult_wires

    if term == "bilinear":
        k, ell = mode_ids
        mode_mult_wires = {
            "output_wires": "cache",
            "work_wires": "work",
        }
        mode_mult_wires = {new: registers[old] for new, old in mode_mult_wires.items()}
        mode_mult_wires["x_wires"] = mode_registers[k]
        mode_mult_wires["y_wires"] = mode_registers[ell]
        # The signed register for _half_signed_out_multiplier must be the _second_ input
        coeff_mult_wires = {
            "x_wires": "coefficients",
            "y_wires": "cache",
            "output_wires": "phase_gradient",
            "work_wires": "work",
        }
        coeff_mult_wires = {new: registers[old] for new, old in coeff_mult_wires.items()}
        return mode_mult_wires, coeff_mult_wires

    if term == "QROM":
        reg = {"control_wires": "electronic", "target_wires": "coefficients", "work_wires": "work"}
        qrom_wires = {new: registers[old] for new, old in reg.items()}
        # Fix lambda=1
        qrom_wires["work_wires"] = qrom_wires["work_wires"][: len(qrom_wires["control_wires"]) - 1]
        return qrom_wires

    if term == "linear":
        (k,) = mode_ids
        # The signed register for _half_signed_out_multiplier must be the _second_ input
        reg = {
            "x_wires": "coefficients",
            "output_wires": "phase_gradient",
            "work_wires": "work",
        }
        mult_wires = {new: registers[old] for new, old in reg.items()}
        mult_wires["y_wires"] = mode_registers[k]
        return mult_wires

    if term == "constant":
        reg = {"x_wires": "coefficients", "y_wires": "phase_gradient", "work_wires": "work"}
    return {new: registers[old] for new, old in reg.items()}


# pylint: disable-next=too-many-arguments,too-many-statements
def _trotter_step_second_order(
    time,
    hamiltonian,
    registers,
    mode_registers,
    aqft_order,
    n_states,
    n_modes,
    n_elec,
    branch_probs,
):
    r"""Emit a single symmetric second-order Trotter step.

    The (first-Trotter-order) kinetic evolution is placed in the middle, wrapped by the
    second-Trotter-order position-fragment evolutions (forward, then backward). Within each
    position fragment we iterate over the linear, quadratic and bilinear terms; within each of
    these we iterate over the modes (or mode pairs) using ``for_loop``\ s. The fragment loop
    itself is also a ``for_loop``, so ``fragment_idx`` may be a traced value.

    ``branch_probs`` holds the ``branch-prob`` hints for applying each term, see
    :class:`~.TrotterVibronic`.
    """
    branch_probs = dict(branch_probs or ())
    precision = len(registers["phase_gradient"])
    all_coeffs, bilinear_indices = _preprocess_data(time, hamiltonian, n_elec, n_states, n_modes)
    if compiler.active() or capture.enabled():
        all_coeffs = tuple(math.array(c, like="jax") for c in all_coeffs)
        bilinear_indices = math.array(bilinear_indices, like="jax")
    all_constant, all_linear, all_quadratic, all_bilinear = all_coeffs
    qrom_wires = _extract_registers(registers, mode_registers, "QROM")
    num_position_fragments = hamiltonian.num_fragments

    def position_fragments(i):
        const_coeffs = all_constant[i]
        linear_coeffs = all_linear[i]
        quadratic_coeffs = all_quadratic[i]
        bilinear_coeffs = all_bilinear[i]

        _diagonalize_vibronic_circuit(
            fragment_idx=i,
            wires=registers["electronic"],
            num_fragments=num_position_fragments,
            inverse=True,
        )

        def constant_term(prev_bitstrings):
            def skip_fn():
                return prev_bitstrings

            @hint({"branch-prob": branch_probs.get("constant")})
            def actual_fn():
                new_bitstrings = _load_coefficients(
                    const_coeffs, precision, prev_bitstrings, qrom_wires
                )
                SemiAdder(**_extract_registers(registers, mode_registers, "constant"))
                return new_bitstrings

            return cond(math.allclose(const_coeffs, 0.0), skip_fn, actual_fn)()

        @for_loop(n_modes)
        def linear_terms(k, prev_bitstrings):
            _coeffs = linear_coeffs[k]

            def skip_fn():
                return prev_bitstrings

            @hint({"branch-prob": branch_probs.get("linear")})
            def actual_fn():
                new_bitstrings = _load_coefficients(_coeffs, precision, prev_bitstrings, qrom_wires)
                _half_signed_out_multiplier(
                    **_extract_registers(registers, mode_registers, "linear", k)
                )
                return new_bitstrings

            return cond(math.allclose(_coeffs, 0.0), skip_fn, actual_fn)()

        @for_loop(n_modes)
        def quadratic_terms(k, prev_bitstrings):
            _coeffs = quadratic_coeffs[k]

            def skip_fn():
                return prev_bitstrings

            @hint({"branch-prob": branch_probs.get("quadratic")})
            def actual_fn():
                new_bitstrings = _load_coefficients(_coeffs, precision, prev_bitstrings, qrom_wires)
                square_wires, mult_wires = _extract_registers(
                    registers, mode_registers, "quadratic", k
                )
                SignedOutSquare(**square_wires, output_wires_zeroed=True)
                OutMultiplier(**mult_wires)
                adjoint(SignedOutSquare(**square_wires, output_wires_zeroed=True))
                return new_bitstrings

            return cond(math.allclose(_coeffs, 0.0), skip_fn, actual_fn)()

        @for_loop(bilinear_indices.shape[1])
        def bilinear_terms(k, prev_bitstrings):
            _coeffs = bilinear_coeffs[k]
            ids = bilinear_indices[:, k]

            def skip_fn():
                return prev_bitstrings

            @hint({"branch-prob": branch_probs.get("bilinear")})
            def actual_fn():
                mode_mult_wires, coeff_mult_wires = _extract_registers(
                    registers, mode_registers, "bilinear", *ids
                )
                new_bitstrings = _load_coefficients(_coeffs, precision, prev_bitstrings, qrom_wires)
                SignedOutMultiplier(**mode_mult_wires, output_wires_zeroed=True)
                _half_signed_out_multiplier(**coeff_mult_wires)
                adjoint(SignedOutMultiplier(**mode_mult_wires, output_wires_zeroed=True))
                return new_bitstrings

            return cond(math.allclose(_coeffs, 0.0), skip_fn, actual_fn)()

        prev_bitstrings = np.zeros((n_states, precision), dtype=int)
        prev_bitstrings = constant_term(prev_bitstrings)
        # pylint: disable-next=no-value-for-parameter
        prev_bitstrings = linear_terms(prev_bitstrings)
        # pylint: disable-next=no-value-for-parameter
        prev_bitstrings = quadratic_terms(prev_bitstrings)
        # Skip empty bilinear loop: zero-iteration ``for_loop`` still traces under capture.
        if bilinear_indices.shape[1] > 0:
            # pylint: disable-next=no-value-for-parameter
            prev_bitstrings = bilinear_terms(prev_bitstrings)

        # Finish up the coefficients register by unloading the last loaded coefficients
        QROM(prev_bitstrings, **qrom_wires)
        _diagonalize_vibronic_circuit(
            fragment_idx=i, wires=registers["electronic"], num_fragments=num_position_fragments
        )

    def kinetic_fragment():
        # use ``time``, not ``first_order_time_step`` because the kinetic fragment is the
        # middle one in second-order Trotter, so the two neighbouring first-order steps merge.
        kinetic_coeffs = _momentum_coefficients(hamiltonian.kinetic) * time
        if compiler.active() or capture.enabled():
            kinetic_coeffs = math.array(kinetic_coeffs, like="jax")

        @for_loop(n_modes)
        def kinetic_terms(k):
            _coeffs = kinetic_coeffs[k]

            def skip_fn():
                """Do nothing."""

            @hint({"branch-prob": branch_probs.get("kinetic")})
            def actual_fn():
                square_wires, mult_wires = _extract_registers(
                    registers, mode_registers, "quadratic", k
                )
                # Negated for the same phase-gradient sign convention as ``_load_coefficients``.
                bitstring = math.binary_decimals(-_coeffs, precision, unit=2 * np.pi)
                MultiX(bitstring, registers["coefficients"])
                AQFT(order=aqft_order, wires=mode_registers[k])
                SignedOutSquare(**square_wires, output_wires_zeroed=True)
                OutMultiplier(**mult_wires)
                adjoint(SignedOutSquare(**square_wires, output_wires_zeroed=True))
                adjoint(AQFT(order=aqft_order, wires=mode_registers[k]))
                # ``MultiX`` is self-adjoint, so the unload is the same call.
                MultiX(bitstring, registers["coefficients"])

            cond(math.allclose(_coeffs, 0.0), skip_fn, actual_fn)()

        # pylint: disable-next=no-value-for-parameter
        kinetic_terms()

    # pylint: disable-next=no-value-for-parameter
    for_loop(num_position_fragments)(position_fragments)()
    kinetic_fragment()
    # pylint: disable-next=no-value-for-parameter
    for_loop(num_position_fragments - 1, -1, -1)(position_fragments)()


# pylint: disable-next=too-many-arguments
def _run_trotter_vibronic(
    evolution_time,
    num_trotter_steps,
    hamiltonian,
    registers,
    mode_registers,
    aqft_order,
    n_states,
    n_modes,
    n_elec,
    branch_probs,
):
    """Emit ``num_trotter_steps`` symmetric second-order Trotter steps."""
    trotter_time_step = evolution_time / num_trotter_steps

    def _step(_step_idx, hamiltonian):
        # Carry ``hamiltonian`` through the loop for valid traced loop-body inputs.
        _trotter_step_second_order(
            trotter_time_step,
            hamiltonian,
            registers,
            mode_registers,
            aqft_order,
            n_states,
            n_modes,
            n_elec,
            branch_probs,
        )
        return hamiltonian

    for_loop(num_trotter_steps)(_step)(hamiltonian)


def _validate_registers(registers, mode_registers, n_modes, n_states):
    """Light validation of the wire register sizes. See :class:`~.TrotterVibronic`."""
    b = len(registers["coefficients"])
    k = len(mode_registers[0])
    n = math.ceil_log2(n_states)
    needed_work_wires = max(n - 1, 2 * k, 2 * b + 2)

    if len(registers["electronic"]) != n:
        raise ValueError(
            f"Expected {n} qubits for {n_states} electronic states, but got "
            f"{len(registers['electronic'])}."
        )
    if len(registers["cache"]) != 2 * k:
        raise ValueError(
            f"Expected exactly {2 * k} cache qubits for {k} qubits per vibrational mode, "
            f"but got {len(registers['cache'])}."
        )
    if len(registers["phase_gradient"]) < b:
        raise ValueError(
            "Expected the phase-gradient register to have at least as many qubits as the "
            f"coefficients register ({b} qubits), but got {len(registers['phase_gradient'])}."
        )
    if len(registers["work"]) < needed_work_wires:
        raise ValueError(
            f"Expected at least {needed_work_wires} work qubits, but got {len(registers['work'])}."
        )
    vibr_sizes = [len(mode_registers[i]) for i in range(n_modes)]
    if any(size != vibr_sizes[0] for size in vibr_sizes[1:]):
        raise ValueError(
            f"Expected all vibrational mode registers to have the same size, but got {vibr_sizes}."
        )
