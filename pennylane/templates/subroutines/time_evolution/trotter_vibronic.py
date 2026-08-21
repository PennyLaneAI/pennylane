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
"""Contains the ``TrotterVibronic`` template for vibronic Hamiltonians."""

from collections import defaultdict
from functools import reduce

import numpy as np

from pennylane import capture, compiler, math
from pennylane.allocation import allocate
from pennylane.control_flow import for_loop
from pennylane.core.operator import Operator2
from pennylane.decomposition import (
    add_decomps,
    adjoint_resource_rep,
    controlled_resource_rep,
    register_resources,
    resource_rep,
)
from pennylane.ops import CNOT, Hadamard, X, adjoint, cond
from pennylane.typing import AbstractWires, Float, Wire
from pennylane.wires import Wires, WiresLike

from ..aqft import AQFT
from ..arithmetic.incrementer import Incrementer
from ..arithmetic.out_multiplier import OutMultiplier
from ..arithmetic.semi_adder import SemiAdder
from ..arithmetic.signed_out_multiplier import SignedOutMultiplier, _twos_complement_helper
from ..arithmetic.signed_out_square import SignedOutSquare
from ..qrom import QROM

#: Keys expected in the dense vibronic Hamiltonian dictionary.
HAMILTONIAN_KEYS = ("constant", "linear", "quadratic", "kinetic")


class TrotterVibronic(Operator2):
    r"""Second-order Trotter circuit for vibronic Hamiltonian simulation using phase-gradient
    arithmetic.

    This template realizes :math:`U \approx e^{-iHt}` for a vibronic Hamiltonian, using the
    phase-gradient arithmetic construction of `Motlagh et al, arXiv:2411.13669
    <https://arxiv.org/abs/2411.13669>`__. The Hamiltonian acts on :math:`N` electronic states
    (represented on :math:`n = \lceil\log_2(N)\rceil` qubits) and :math:`M` vibrational modes,
    each discretized on a grid represented with :math:`k` qubits. The vibronic Hamiltonian is
    partitioned into position fragments (diagonal in a fragment-specific electronic basis and
    polynomial in the mode positions up to second order) and a single trailing kinetic fragment
    (quadratic in the mode momenta).

    Args:
        evolution_time (float): time :math:`t` for which to evolve under the vibronic Hamiltonian.
        num_trotter_steps (int): number of second-order Trotter steps to use.
        hamiltonian (dict): the vibronic Hamiltonian as a dictionary of dense coefficient tensors
            (the stacked output of ``fragment_to_dense``). The expected keys and shapes are

            * ``"constant"``: ``(F, N, N)`` -- the constant (mode-independent) coefficients of the
              ``F`` position fragments;
            * ``"linear"``: ``(F, N, N, M)`` -- the linear-in-position coefficients;
            * ``"quadratic"``: ``(F, N, N, M, M)`` -- the quadratic-in-position coefficients;
            * ``"kinetic"``: ``(N, N, M, M)`` -- the quadratic-in-momentum coefficients of the
              single kinetic fragment.

            Here ``N`` is the number of electronic states, ``M`` the number of vibrational modes,
            and ``F`` the number of position fragments.
        electronic (WiresLike): the :math:`n` electronic-state wires.
        vib_wires (WiresLike): the :math:`M \cdot k` vibrational-mode wires, provided as a single
            flattened register. Internally these are reshaped into ``M`` registers of ``k`` wires,
            one per mode, via ``np.array(vib_wires).reshape(M, -1)``.
        coefficients (WiresLike): the :math:`b` wires of the Hamiltonian-coefficient register.
        phase_gradient (WiresLike): the :math:`b` wires holding the phase-gradient resource state.
        cache (WiresLike): the :math:`2k` cache wires for squared/multiplied mode registers. If
            omitted (default), they are dynamically allocated in the decomposition.
        work (WiresLike): the :math:`\max(n-1, 2k, 2b+2)` work wires for data loading and
            arithmetic. If omitted (default), they are dynamically allocated in the decomposition.
        aqft_order (int): approximation order of the :class:`~.AQFT` used to transform between
            position and momentum space. If ``None`` (default), no approximation is made
            (``aqft_order = k - 1``).
        diag_keys (tuple[tuple[int, int]]): the per-fragment electronic diagonalization keys. If
            ``None`` (default), they are derived from the non-zero structure of ``hamiltonian``.

    .. warning::

        This template is tailored to the vibronic fragments produced by
        ``pennylane.labs.trotter_error.vibronic_fragments``. In particular, the electronic
        diagonalization assumes the populated coefficients follow the structure imposed there.

    .. details::
        :title: Register sizes
        :href: register-sizes

        With :math:`N` electronic states (:math:`n = \lceil\log_2(N)\rceil` qubits), :math:`M`
        vibrational modes (:math:`k` qubits each), and :math:`b`-qubit coefficient and
        phase-gradient registers, the wire registers should have the following sizes:

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - argument
             - expected size
             - information content
           * - ``electronic``
             - :math:`n`
             - electronic state
           * - ``vib_wires``
             - :math:`M \cdot k`
             - positions of all vibrational modes (signed), flattened
           * - ``cache``
             - :math:`2k`
             - cached squares/products of modes (signed/unsigned)
           * - ``coefficients``
             - :math:`b`
             - Hamiltonian coefficients (unsigned)
           * - ``phase_gradient``
             - :math:`b`
             - phase-gradient state (unsigned)
           * - ``work``
             - :math:`\max(n-1, 2k, 2b+2)`
             - data-loading/arithmetic scratch

    **Example**

    .. code-block:: python

        import numpy as np
        import pennylane as qp

        n_states, n_modes, k, b = 2, 1, 2, 3
        n = int(np.ceil(np.log2(n_states)))
        # A single diagonal position fragment and a diagonal kinetic fragment
        hamiltonian = {
            "constant": np.zeros((1, n_states, n_states)),
            "linear": np.zeros((1, n_states, n_states, n_modes)),
            "quadratic": np.zeros((1, n_states, n_states, n_modes, n_modes)),
            "kinetic": np.einsum(
                "ab,cd->abcd", np.eye(n_states), np.diag(0.3 * np.ones(n_modes))
            ),
        }
        wires = qp.registers({
            "electronic": n, "vib_wires": n_modes * k, "cache": 2 * k,
            "coefficients": b, "phase_gradient": b, "work": max(n - 1, 2 * k, 2 * b + 2),
        })

        op = qp.TrotterVibronic(
            evolution_time=1.0, num_trotter_steps=1, hamiltonian=hamiltonian,
            electronic=wires["electronic"], vib_wires=wires["vib_wires"],
            cache=wires["cache"], coefficients=wires["coefficients"],
            phase_gradient=wires["phase_gradient"], work=wires["work"], aqft_order=1,
        )

    """

    dynamic_argnames = ("evolution_time",)
    hybrid_argnames = ("hamiltonian",)
    wire_argnames = ("electronic", "vib_wires", "coefficients", "phase_gradient", "cache", "work")
    # ``num_trotter_steps`` drives Python-level control flow (the number of Trotter steps), and
    # ``aqft_order``/``diag_keys`` describe the compile-time circuit structure, so all three are
    # treated as (non-compilable) static arguments.
    static_argnames = ("num_trotter_steps", "aqft_order", "diag_keys")

    # ``cache`` and ``work`` are optional work registers: when omitted they are dynamically
    # allocated in the decomposition (see ``sum_of_slaters`` for the same pattern), so they are
    # not part of the declared ``arg_specs``.
    arg_specs = {
        "evolution_time": Float,
        "electronic": Wire[-1],
        "vib_wires": Wire[-1],
        "coefficients": Wire[-1],
        "phase_gradient": Wire[-1],
    }

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        evolution_time,
        num_trotter_steps,
        hamiltonian,
        electronic: WiresLike,
        vib_wires: WiresLike,
        coefficients: WiresLike,
        phase_gradient: WiresLike,
        cache: WiresLike = (),
        work: WiresLike = (),
        aqft_order=None,
        diag_keys=None,
    ):
        _validate_hamiltonian(hamiltonian)
        # Canonicalize the key order to match how ``jax.tree_util`` flattens dicts (sorted keys),
        # so the operator round-trips through its pytree registration (required by ``assert_valid``).
        hamiltonian = {key: hamiltonian[key] for key in sorted(hamiltonian)}
        if not isinstance(num_trotter_steps, int) or num_trotter_steps <= 0:
            raise ValueError(
                "The number of Trotter steps should be a positive integer, "
                f"but got {num_trotter_steps}."
            )

        # Abstract inputs are handled here (rather than in an ``__abstract_init__`` override): the
        # register-size validation only needs the wire *counts* (skipped for abstract/traced wire
        # registers), and ``diag_keys`` depends on the concrete non-zero structure of the
        # Hamiltonian (so it can only be derived when the Hamiltonian is concrete).
        if _wires_are_concrete(vib_wires) and _wires_are_concrete(electronic):
            # Validate the register sizes fixed by the Hamiltonian shape at construction time, so
            # the failure is clear and attributed to ``TrotterVibronic`` (rather than surfacing
            # later as an opaque reshape/wire-count error from the decomposition or estimator).
            n_states = hamiltonian["constant"].shape[1]
            n_modes = hamiltonian["linear"].shape[-1]
            num_vib_wires = len(Wires(vib_wires))
            if num_vib_wires % n_modes != 0:
                raise ValueError(
                    f"The number of vibrational wires ({num_vib_wires}) must be divisible by the "
                    f"number of modes ({n_modes})."
                )
            num_electronic = math.ceil_log2(n_states)
            if len(Wires(electronic)) != num_electronic:
                raise ValueError(
                    f"Expected {num_electronic} electronic qubits for {n_states} electronic "
                    f"states, but got {len(Wires(electronic))}."
                )

        if diag_keys is None:
            if any(math.is_abstract(v) for v in hamiltonian.values()):
                raise ValueError(
                    "TrotterVibronic requires `diag_keys` to be provided explicitly when the "
                    "`hamiltonian` is traced (abstract)."
                )
            diag_keys = _derive_diag_keys(hamiltonian)

        super().__init__(
            evolution_time,
            num_trotter_steps,
            hamiltonian,
            electronic,
            vib_wires,
            coefficients,
            phase_gradient,
            cache,
            work,
            aqft_order,
            diag_keys,
        )


# pylint: disable-next=too-many-arguments
def _trotter_vibronic_resources(
    evolution_time,
    num_trotter_steps,
    hamiltonian,
    electronic,
    vib_wires,
    coefficients,
    phase_gradient,
    cache,
    work,
    aqft_order,
    diag_keys,
):
    """Coarse (upper-bound) gate counts for the vibronic Trotter circuit.

    This estimate is intentionally inexact (``exact=False``): terms whose coefficients happen to
    vanish are skipped at runtime, and the sub-operations are counted at their top level. The keys
    are abstractifiable operator instances / resource representations sized from the actual wire
    registers, so the estimate can drive the decomposition graph.
    """
    # ``evolution_time``, ``cache`` and ``diag_keys`` are part of the shared resource/decomposition
    # signature but do not affect the (structural) gate counts.
    # pylint: disable=unused-argument
    num_fragments = hamiltonian["constant"].shape[0]
    n_states = hamiltonian["constant"].shape[1]
    n_modes = hamiltonian["linear"].shape[-1]
    n_elec = len(electronic)
    b = len(coefficients)
    n_pg = len(phase_gradient)
    k = len(vib_wires) // n_modes
    # ``work`` may be dynamically allocated (empty here), in which case fall back to the required
    # size so the sub-operations are counted consistently with the decomposition.
    n_work = len(work) if len(work) > 0 else max(n_elec - 1, 2 * k, 2 * b + 2)
    num_pairs = n_modes * (n_modes - 1) // 2

    # Disjoint dummy-wire allocator: only the wire *counts* survive abstractification, so the
    # concrete labels are irrelevant as long as each operator instance is internally consistent.
    _next = [0]

    def ww(size):
        size = max(size, 0)
        wires = list(range(_next[0], _next[0] + size))
        _next[0] += size
        return wires

    # Each Trotter step visits every position fragment twice (forward + backward).
    position_visits = 2 * num_fragments * num_trotter_steps

    # Data loading (QROM) and the arithmetic primitives, sized to match ``_extract_registers``.
    qrom = QROM(
        np.zeros((n_states, b), dtype=int),
        control_wires=ww(n_elec),
        target_wires=ww(b),
        work_wires=ww(max(n_elec - 1, 0)),
    )
    semi_adder = SemiAdder(x_wires=ww(b), y_wires=ww(n_pg), work_wires=ww(n_work))

    # ``OutMultiplier`` appears with three different ``y`` register sizes: inside the half-signed
    # multiplier for the linear terms (``y = k``, the mode register) and the bilinear terms
    # (``y = 2k``, the cache register), and directly for the quadratic/kinetic terms
    # (``y = 2k - 1``, the cache register minus its sign bit).
    out_mult_linear = OutMultiplier(
        x_wires=ww(b), y_wires=ww(k), output_wires=ww(n_pg), work_wires=ww(max(n_work - 1, 0))
    )
    out_mult_bilinear = OutMultiplier(
        x_wires=ww(b), y_wires=ww(2 * k), output_wires=ww(n_pg), work_wires=ww(max(n_work - 1, 0))
    )
    out_mult_quad = OutMultiplier(
        x_wires=ww(b), y_wires=ww(max(2 * k - 1, 1)), output_wires=ww(n_pg), work_wires=ww(n_work)
    )
    signed_out_mult = SignedOutMultiplier(
        x_wires=ww(k),
        y_wires=ww(k),
        output_wires=ww(2 * k),
        work_wires=ww(n_work),
        output_wires_zeroed=True,
    )
    # ``SignedOutSquare`` is a legacy ``Operation`` (no abstractifiable instance form), so it is
    # declared via ``resource_rep``; every other primitive is an ``Operator2`` declared by instance.
    signed_square_params = {
        "num_x_wires": k,
        "num_output_wires": max(2 * k - 1, 1),
        "num_work_wires": n_work,
        "output_wires_zeroed": True,
    }
    signed_square = resource_rep(SignedOutSquare, **signed_square_params)
    # Mirror the decomposition's resolution of ``aqft_order`` (``k - 1`` when unset) so the
    # resource estimate and the emitted circuit agree.
    aqft = AQFT(order=(aqft_order if aqft_order is not None else k - 1), wires=ww(k))

    # The compute/uncompute pairs are emitted as ``adjoint(...)`` wrappers in the decomposition, so
    # they must be declared as such (not as their bare base ops) for the graph to find a path.
    adj_signed_square = adjoint_resource_rep(SignedOutSquare, signed_square_params)
    adj_signed_out_mult = adjoint(signed_out_mult)
    adj_aqft = adjoint(aqft)

    # The half-signed multiplier takes the two's complement of its signed input via a
    # sign-bit-controlled ``Incrementer`` (twice per call: compute + uncompute). It acts on the
    # ``k``-wire mode register (linear terms) and the ``2k``-wire cache register (bilinear terms).
    ctrl_incrementer_linear = controlled_resource_rep(
        Incrementer,
        {"num_wires": k, "num_work_wires": max(n_work - 1, 0)},
        num_control_wires=1,
    )
    ctrl_incrementer_bilinear = controlled_resource_rep(
        Incrementer,
        {"num_wires": 2 * k, "num_work_wires": max(n_work - 1, 0)},
        num_control_wires=1,
    )

    resources = defaultdict(int)

    # -- Electronic diagonalization (forward + adjoint per visit) --
    resources[Hadamard(wires=ww(1))] += 2 * position_visits

    # -- CNOTs: electronic diagonalization plus the half-signed multipliers, which cache the sign
    #    bit, invert the register for the two's complement, and apply the controlled output flips --
    resources[CNOT(wires=ww(2))] += (
        2 * position_visits * max(n_elec - 1, 0)
        + position_visits * n_modes * (2 + 2 * k + 2 * n_pg)
        + position_visits * num_pairs * (2 + 4 * k + 2 * n_pg)
    )

    # -- Data loading: constant + each linear/quadratic/bilinear term + final unload --
    resources[qrom] += position_visits * (2 + 2 * n_modes + num_pairs)
    resources[semi_adder] += position_visits

    # -- Linear terms: one half-signed multiplier each --
    resources[out_mult_linear] += position_visits * n_modes
    resources[ctrl_incrementer_linear] += position_visits * n_modes * 2

    # -- Quadratic terms: two SignedOutSquares (compute + uncompute) and one OutMultiplier --
    resources[signed_square] += position_visits * n_modes
    resources[adj_signed_square] += position_visits * n_modes
    resources[out_mult_quad] += position_visits * n_modes

    # -- Bilinear terms: two SignedOutMultipliers (compute + uncompute) and one half-signed
    #    multiplier --
    resources[signed_out_mult] += position_visits * num_pairs
    resources[adj_signed_out_mult] += position_visits * num_pairs
    resources[out_mult_bilinear] += position_visits * num_pairs
    resources[ctrl_incrementer_bilinear] += position_visits * num_pairs * 2

    # -- Kinetic fragment (once per Trotter step): per mode, load the momentum coefficients as a
    #    basis state (conditional PauliX per coefficient wire), then AQFT + SignedOutSquare +
    #    OutMultiplier and their uncomputation --
    kinetic_visits = num_trotter_steps * n_modes
    resources[X(wires=ww(1))] += kinetic_visits * 2 * b
    resources[aqft] += kinetic_visits
    resources[adj_aqft] += kinetic_visits
    resources[signed_square] += kinetic_visits
    resources[adj_signed_square] += kinetic_visits
    resources[out_mult_quad] += kinetic_visits

    return dict(resources)


def _required_work_wire_sizes(hamiltonian, vib_wires, coefficients):
    """Required ``(cache, work)`` register sizes for the vibronic Trotter circuit."""
    n_states = hamiltonian["constant"].shape[1]
    n_modes = hamiltonian["linear"].shape[-1]
    k = len(vib_wires) // n_modes
    b = len(coefficients)
    n = math.ceil_log2(n_states)
    return 2 * k, max(n - 1, 2 * k, 2 * b + 2)


def _trotter_vibronic_work_wires(hamiltonian, vib_wires, coefficients, cache, work, **_):
    """Number of zeroed work wires allocated when ``cache``/``work`` are omitted."""
    cache_size, work_size = _required_work_wire_sizes(hamiltonian, vib_wires, coefficients)
    num_alloc = 0
    if len(cache) == 0:
        num_alloc += cache_size
    if len(work) == 0:
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
    electronic,
    vib_wires,
    coefficients,
    phase_gradient,
    cache,
    work,
    aqft_order,
    diag_keys,
):
    n_states = hamiltonian["constant"].shape[1]
    n_modes = hamiltonian["linear"].shape[-1]
    n_elec = len(electronic)

    vib = list(vib_wires)
    if compiler.active() or capture.enabled():
        mode_registers = math.array(vib, like="jax").reshape(n_modes, -1)
    else:
        mode_registers = np.array(vib).reshape(n_modes, -1)

    resolved_aqft_order = aqft_order if aqft_order is not None else mode_registers.shape[1] - 1

    def _run(cache, work):
        registers = {
            "electronic": electronic,
            "cache": cache,
            "coefficients": coefficients,
            "phase gradient": phase_gradient,
            "work": work,
        }
        _validate_registers(registers, mode_registers, n_modes, n_states)
        _run_trotter_vibronic(
            evolution_time,
            num_trotter_steps,
            hamiltonian,
            diag_keys,
            registers,
            mode_registers,
            resolved_aqft_order,
            n_states,
            n_modes,
            n_elec,
        )

    # ``cache`` and ``work`` are optional: when omitted they are dynamically allocated here (as
    # zeroed, restored wires) instead of being required as explicit register arguments.
    cache_size, work_size = _required_work_wire_sizes(hamiltonian, vib_wires, coefficients)
    need_cache = len(cache) == 0
    need_work = len(work) == 0
    num_alloc = (cache_size if need_cache else 0) + (work_size if need_work else 0)

    if num_alloc == 0:
        _run(cache, work)
        return

    with allocate(num_alloc, state="zero", restored=True) as allocated:
        start = 0
        if need_cache:
            cache = allocated[start : (start := start + cache_size)]
        if need_work:
            work = allocated[start : (start := start + work_size)]
        _run(cache, work)


add_decomps(TrotterVibronic, _trotter_vibronic_decomposition)


# ---------------------------------------------------------------------------
# ---------------------------- Electronic diagonalization -------------------
# ---------------------------------------------------------------------------


def _diagonalization_support(key, n_wires):
    """Return the control wire index and the CNOT target indices for a fragment key.

    ``key`` is the ``(row, column)`` index of the (first) non-zero off-diagonal
    electronic matrix element of a fragment. Following Fig. 2 of
    `Motlagh et al, arXiv:2411.13669 <https://arxiv.org/abs/2411.13669>`__, the
    fragment is diagonalized on the electronic register with a single Hadamard and a
    number of CNOTs, all acting on the wires where the bit strings of ``key[0]`` and
    ``key[1]`` differ.
    """
    bitstrings = [math.int_to_binary(int(k), n_wires) for k in key]
    diagonalization_key = (np.array(bitstrings[0]) + np.array(bitstrings[1])) % 2
    support = np.where(diagonalization_key)[0][::-1]
    return int(support[0]), [int(k) for k in support[1:]]


def _diagonalize_vibronic_circuit(key, wires):
    r"""Diagonalize a vibronic fragment by applying Clifford operations on the electronic register.

    Based on Fig. 2 of `Motlagh et al, arXiv:2411.13669 <https://arxiv.org/abs/2411.13669>`__.
    Requires one :class:`~.Hadamard` and at most :math:`\lceil\log_2(N)\rceil - 1`
    :class:`~.CNOT` gates, where :math:`N` is the number of electronic states.

    Args:
        key (tuple[int]): row and column index of the (first) non-zero off-diagonal element
            of the fragment. The circuit is the identity if ``key[0] == key[1]``.
        wires (WiresLike): electronic wires on which the fragment acts.
    """
    if int(key[0]) == int(key[1]):
        # already diagonal, no operations required
        return

    control, targets = _diagonalization_support(key, len(wires))
    Hadamard(wires=wires[control])
    for target in targets:
        CNOT(wires=[wires[control], wires[target]])


def _diagonalization_matrix(key, n_wires):
    """Dense matrix implemented by :func:`_diagonalize_vibronic_circuit` for ``key``.

    The matrix acts on ``2 ** n_wires`` basis states with wire ``0`` being the most
    significant qubit (matching :func:`pennylane.math.int_to_binary`).
    """
    dim = 2**n_wires
    if int(key[0]) == int(key[1]):
        return np.eye(dim)

    identity = np.eye(2)
    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    proj_0 = np.array([[1.0, 0.0], [0.0, 0.0]])
    proj_1 = np.array([[0.0, 0.0], [0.0, 1.0]])
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]])

    def _embed(gate, wire):
        return reduce(np.kron, [gate if w == wire else identity for w in range(n_wires)])

    def _embed_cnot(control, target):
        term0 = reduce(np.kron, [proj_0 if w == control else identity for w in range(n_wires)])
        term1 = reduce(
            np.kron,
            [
                proj_1 if w == control else (pauli_x if w == target else identity)
                for w in range(n_wires)
            ],
        )
        return term0 + term1

    control, targets = _diagonalization_support(key, n_wires)
    matrix = _embed(hadamard, control)
    for target in targets:
        matrix = _embed_cnot(control, target) @ matrix
    return matrix


# ---------------------------------------------------------------------------
# ------------------------- Half-signed multiplier --------------------------
# ---------------------------------------------------------------------------


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
    cached sign bit are uncomputed. See the ``pennylane.labs`` prototype for the full
    derivation.

    .. note::
        This is logically identical to
        ``pennylane.labs.templates.half_signed_out_multiplier``. ``tach`` forbids importing that
        labs helper from core, so the logic is duplicated here. The two copies should be
        consolidated (e.g. by promoting the helper into
        ``pennylane.templates.subroutines.arithmetic``) to avoid drift.
    """
    y_aux, work_wires = work_wires[0], work_wires[1:]

    if compiler.active() or capture.enabled():
        output_arr = math.array(list(output_wires), like="jax")
    else:
        output_arr = list(output_wires)

    def _flip_outputs():
        # Controlled (on the cached sign bit) broadcasted ``PauliX`` across the output register,
        # i.e. a ``CNOT`` from ``y_aux`` onto every output wire. This replaces the previous
        # ``ctrl(BasisState([1, ...]))``, which is a state-prep operation whose controlled form is
        # ill-defined under program capture.
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


def _load_basis(bitstring, wires):
    r"""Prepare the computational basis state ``bitstring`` on ``wires`` (assumed to start in
    :math:`|0\rangle`).

    Replaces ``BasisState(bitstring, wires)`` with an explicit conditional :class:`~.PauliX` per
    wire (applied iff the corresponding bit is set). Each conditional ``X`` is self-inverse, so
    this loader is its own adjoint -- unlike ``BasisState``, whose adjoint/controlled forms are
    state-prep operations that behave inconsistently between the tape and program-capture paths.
    """
    if compiler.active() or capture.enabled():
        bitstring = math.array(bitstring, like="jax")
        wires_arr = math.array(list(wires), like="jax")
    else:
        wires_arr = list(wires)

    @for_loop(len(wires))
    def _prep(i):
        cond(bitstring[i] != 0, X)(wires_arr[i])

    # pylint: disable-next=no-value-for-parameter
    _prep()


# ---------------------------------------------------------------------------
# ------------------------- Coefficient preprocessing -----------------------
# ---------------------------------------------------------------------------


# pylint: disable-next=too-many-arguments
def _position_coefficients(matrix, constant, linear, quadratic, n_states, n_modes):
    """Diagonalize a single position fragment and extract its phase coefficients.

    Args:
        matrix (np.ndarray): dense ``(n_states, n_states)`` diagonalization matrix of the fragment.
        constant (TensorLike): dense ``(n_states, n_states)`` constant coefficients.
        linear (TensorLike): dense ``(n_states, n_states, n_modes)`` linear coefficients.
        quadratic (TensorLike): dense ``(n_states, n_states, n_modes, n_modes)`` quadratic
            coefficients.
        n_states (int): number of electronic states.
        n_modes (int): number of vibrational modes.

    Returns:
        tuple: ``(constant, linear, quadratic, bilinear)`` diagonalized coefficient tensors of
        shapes ``(n_states,)``, ``(n_modes, n_states)``, ``(n_modes, n_states)`` and
        ``(n_pairs, n_states)``, where ``n_pairs = n_modes * (n_modes - 1) / 2``.
    """
    matrix_t = matrix.T

    rotated_constant = matrix_t @ constant @ matrix
    constant_diag = math.stack([rotated_constant[a, a] for a in range(n_states)])

    linear_full = math.einsum("ba,bcz,cd->zad", matrix, linear, matrix)
    linear_diag = math.stack([linear_full[:, a, a] for a in range(n_states)], axis=1)

    sec_order = math.einsum("ba,bcyz,cd->yzad", matrix, quadratic, matrix)
    sec_order_diag = math.stack([sec_order[:, :, a, a] for a in range(n_states)], axis=2)
    quadratic_diag = math.stack([sec_order_diag[k, k] for k in range(n_modes)])

    rows, cols = np.triu_indices(n_modes, 1)
    if len(rows) > 0:
        bilinear = math.stack([sec_order_diag[int(r), int(c)] for r, c in zip(rows, cols)])
    else:
        bilinear = np.zeros((0, n_states))

    return constant_diag, linear_diag, quadratic_diag, bilinear


def _momentum_coefficients(kinetic, n_modes):
    """Extract per-mode momentum coefficients from a dense kinetic fragment.

    Args:
        kinetic (TensorLike): dense ``(n_states, n_states, n_modes, n_modes)`` kinetic
            coefficients.
        n_modes (int): number of vibrational modes.

    Returns:
        TensorLike: momentum coefficients of shape ``(n_modes,)``.
    """
    return math.stack([kinetic[0, 0, m, m] for m in range(n_modes)])


# pylint: disable-next=too-many-arguments
def _preprocess_data(time, hamiltonian, diag_keys, n_elec, n_states, n_modes):
    """Diagonalize all position fragments and scale their coefficients by the time step.

    Returns the tuple ``((constant, linear, quadratic, bilinear), bilinear_indices)`` where the
    coefficient tensors are stacked over the position fragments and scaled by ``time / 2`` (the
    first-order time step of the symmetric second-order Trotter step).
    """
    first_order_time_step = time / 2
    constant_dense = hamiltonian["constant"]
    linear_dense = hamiltonian["linear"]
    quadratic_dense = hamiltonian["quadratic"]
    num_fragments = constant_dense.shape[0]

    all_constant, all_linear, all_quadratic, all_bilinear = [], [], [], []
    for i in range(num_fragments):
        matrix = _diagonalization_matrix(diag_keys[i], n_elec)[:n_states, :n_states]
        constant, linear, quadratic, bilinear = _position_coefficients(
            matrix,
            constant_dense[i],
            linear_dense[i],
            quadratic_dense[i],
            n_states,
            n_modes,
        )
        all_constant.append(constant * first_order_time_step)
        all_linear.append(linear * first_order_time_step)
        all_quadratic.append(quadratic * first_order_time_step)
        all_bilinear.append(bilinear * first_order_time_step)

    all_constant = math.stack(all_constant)
    all_linear = math.stack(all_linear)
    all_quadratic = math.stack(all_quadratic)
    all_bilinear = math.stack(all_bilinear)
    bilinear_indices = np.array(np.triu_indices(n_modes, 1))

    if compiler.active() or capture.enabled():
        all_constant = math.array(all_constant, like="jax")
        all_linear = math.array(all_linear, like="jax")
        all_quadratic = math.array(all_quadratic, like="jax")
        all_bilinear = math.array(all_bilinear, like="jax")
        bilinear_indices = math.array(bilinear_indices, like="jax")

    return (all_constant, all_linear, all_quadratic, all_bilinear), bilinear_indices


# ---------------------------------------------------------------------------
# ------------------------------ Data loading -------------------------------
# ---------------------------------------------------------------------------


def _load_coefficients(coefficients, precision, prev_bitstrings, qrom_wires):
    """Load ``coefficients`` into the data-loading register via a differential :class:`~.QROM`.

    The coefficients are converted into ``precision``-bit strings, XOR-ed with the currently
    loaded ``prev_bitstrings`` reference point, and loaded with a :class:`~.QROM`. The newly
    loaded bit strings are returned so that they can be used as the reference point for the next
    data-loading step.
    """
    new_bitstrings = math.binary_decimals(coefficients, precision, unit=2 * np.pi)
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
            "output_wires": "phase gradient",
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
            "output_wires": "phase gradient",
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
            "output_wires": "phase gradient",
            "work_wires": "work",
        }
        mult_wires = {new: registers[old] for new, old in reg.items()}
        mult_wires["y_wires"] = mode_registers[k]
        return mult_wires

    if term == "constant":
        reg = {"x_wires": "coefficients", "y_wires": "phase gradient", "work_wires": "work"}
    return {new: registers[old] for new, old in reg.items()}


# ---------------------------------------------------------------------------
# ------------------------------ Trotter step -------------------------------
# ---------------------------------------------------------------------------


# pylint: disable-next=too-many-arguments,too-many-statements
def _trotter_step_second_order(
    time, hamiltonian, diag_keys, registers, mode_registers, aqft_order, n_states, n_modes, n_elec
):
    r"""Emit a single symmetric second-order Trotter step.

    The (first-Trotter-order) kinetic evolution is placed in the middle, wrapped by the
    second-Trotter-order position-fragment evolutions (forward, then backward). Within each
    position fragment we iterate over the linear, quadratic and bilinear terms; within each of
    these we iterate over the modes (or mode pairs) using ``for_loop``\ s.
    """
    precision = len(registers["phase gradient"])
    all_coeffs, bilinear_indices = _preprocess_data(
        time, hamiltonian, diag_keys, n_elec, n_states, n_modes
    )
    all_constant, all_linear, all_quadratic, all_bilinear = all_coeffs
    qrom_wires = _extract_registers(registers, mode_registers, "QROM")
    num_position_fragments = hamiltonian["constant"].shape[0]

    def position_fragments(i):
        diag_key = diag_keys[i]
        const_coeffs = all_constant[i]
        linear_coeffs = all_linear[i]
        quadratic_coeffs = all_quadratic[i]
        bilinear_coeffs = all_bilinear[i]

        adjoint(_diagonalize_vibronic_circuit, lazy=False)(
            key=diag_key, wires=registers["electronic"]
        )

        def constant_term(prev_bitstrings):
            def skip_fn():
                return prev_bitstrings

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
        # The number of mode pairs is static; skip the bilinear loop entirely when there are none
        # (a zero-iteration ``for_loop`` is still traced once under program capture, which would
        # index into an empty coefficient array).
        if bilinear_indices.shape[1] > 0:
            # pylint: disable-next=no-value-for-parameter
            prev_bitstrings = bilinear_terms(prev_bitstrings)

        # Finish up the coefficients register by unloading the last loaded coefficients
        QROM(prev_bitstrings, **qrom_wires)
        _diagonalize_vibronic_circuit(key=diag_key, wires=registers["electronic"])

    def kinetic_fragment():
        # Use ``time`` (not the first-order time step): the kinetic fragment is the middle one in
        # the symmetric second-order step, so the two neighbouring first-order steps merge.
        kinetic_coeffs = _momentum_coefficients(hamiltonian["kinetic"], n_modes) * time
        if compiler.active() or capture.enabled():
            kinetic_coeffs = math.array(kinetic_coeffs, like="jax")

        @for_loop(n_modes)
        def kinetic_terms(k):
            _coeffs = kinetic_coeffs[k]

            def skip_fn():
                """Do nothing."""

            def actual_fn():
                square_wires, mult_wires = _extract_registers(
                    registers, mode_registers, "quadratic", k
                )
                bitstring = math.binary_decimals(_coeffs, precision, unit=2 * np.pi)
                _load_basis(bitstring, registers["coefficients"])
                AQFT(order=aqft_order, wires=mode_registers[k])
                SignedOutSquare(**square_wires, output_wires_zeroed=True)
                OutMultiplier(**mult_wires)
                adjoint(SignedOutSquare(**square_wires, output_wires_zeroed=True))
                adjoint(AQFT(order=aqft_order, wires=mode_registers[k]))
                # ``_load_basis`` is self-inverse, so the unload is the same call (no ``adjoint``).
                _load_basis(bitstring, registers["coefficients"])

            cond(math.allclose(_coeffs, 0.0), skip_fn, actual_fn)()

        # pylint: disable-next=no-value-for-parameter
        kinetic_terms()

    # The number of position fragments is static, so we iterate over them with a plain Python
    # loop (unrolled in the circuit). This keeps the per-fragment diagonalization keys concrete,
    # so the electronic-diagonalization circuit structure is fixed at compile time.
    for i in range(num_position_fragments):
        position_fragments(i)
    kinetic_fragment()
    for i in range(num_position_fragments - 1, -1, -1):
        position_fragments(i)


# pylint: disable-next=too-many-arguments
def _run_trotter_vibronic(
    evolution_time,
    num_trotter_steps,
    hamiltonian,
    diag_keys,
    registers,
    mode_registers,
    aqft_order,
    n_states,
    n_modes,
    n_elec,
):
    """Emit ``num_trotter_steps`` symmetric second-order Trotter steps."""
    trotter_time_step = evolution_time / num_trotter_steps

    def _step(_step_idx, hamiltonian):
        # ``hamiltonian`` is carried through the for-loop (rather than closed over) so the
        # traced tensors remain valid loop-body inputs under program capture.
        _trotter_step_second_order(
            trotter_time_step,
            hamiltonian,
            diag_keys,
            registers,
            mode_registers,
            aqft_order,
            n_states,
            n_modes,
            n_elec,
        )
        return hamiltonian

    for_loop(num_trotter_steps)(_step)(hamiltonian)


# ---------------------------------------------------------------------------
# ------------------------------ Validation ---------------------------------
# ---------------------------------------------------------------------------


def _derive_diag_keys(hamiltonian):
    """Derive the per-fragment diagonalization keys from the dense coefficient structure.

    For each position fragment, the key is the ``(row, column)`` index of the first electronic
    matrix element (in row-major order) that has a non-zero coefficient in any of the constant,
    linear or quadratic tensors.
    """
    constant = np.asarray(hamiltonian["constant"])
    linear = np.asarray(hamiltonian["linear"])
    quadratic = np.asarray(hamiltonian["quadratic"])
    num_fragments = constant.shape[0]

    diag_keys = []
    for i in range(num_fragments):
        mask = np.abs(constant[i]) > 1e-12
        mask = mask | (np.abs(linear[i]) > 1e-12).any(axis=-1)
        mask = mask | (np.abs(quadratic[i]) > 1e-12).any(axis=(-1, -2))
        nonzero = np.argwhere(mask)
        if len(nonzero) == 0:
            diag_keys.append((0, 0))
        else:
            row, col = nonzero[0]
            diag_keys.append((int(row), int(col)))
    return tuple(diag_keys)


def _wires_are_concrete(wires):
    """Whether a wire register is concrete (i.e. its labels are known, not abstract/traced)."""
    if isinstance(wires, AbstractWires):
        return False
    return not any(math.is_abstract(w) for w in Wires(wires))


def _validate_hamiltonian(hamiltonian):
    """Light validation of the dense vibronic Hamiltonian dictionary."""
    if not isinstance(hamiltonian, dict):
        raise ValueError(
            f"Expected `hamiltonian` to be a dictionary, got {type(hamiltonian).__name__}."
        )
    if set(hamiltonian) != set(HAMILTONIAN_KEYS):
        raise ValueError(
            f"Expected the keys in `hamiltonian` to be {set(HAMILTONIAN_KEYS)}, "
            f"but got {set(hamiltonian)}."
        )
    expected_ndim = {"constant": 3, "linear": 4, "quadratic": 5, "kinetic": 4}
    for key, ndim in expected_ndim.items():
        if math.ndim(hamiltonian[key]) != ndim:
            raise ValueError(
                f"Expected `hamiltonian['{key}']` to be a {ndim}-dimensional array, "
                f"but got {math.ndim(hamiltonian[key])} dimensions."
            )


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
    if len(registers["cache"]) < 2 * k:
        raise ValueError(
            f"Expected at least {2 * k} cache qubits for {k} qubits per vibrational mode, "
            f"but got {len(registers['cache'])}."
        )
    if len(registers["phase gradient"]) < b:
        raise ValueError(
            "Expected the phase-gradient register to have at least as many qubits as the "
            f"coefficients register ({b} qubits), but got {len(registers['phase gradient'])}."
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
