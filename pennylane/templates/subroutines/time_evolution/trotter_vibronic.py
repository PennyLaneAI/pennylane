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

import numpy as np

from pennylane import capture, compiler, math
from pennylane.core.operator import Operator2
from pennylane.decomposition import add_decomps, register_resources
from pennylane.ops import CNOT, BasisState, Hadamard
from pennylane.typing import Wire
from pennylane.wires import WiresLike

from ..aqft import AQFT
from ..arithmetic.out_multiplier import OutMultiplier
from ..arithmetic.semi_adder import SemiAdder
from ..arithmetic.signed_out_multiplier import SignedOutMultiplier
from ..arithmetic.signed_out_square import SignedOutSquare
from ..qrom import QROM
from ._trotter_vibronic_utils import (
    _derive_diag_keys,
    _run_trotter_vibronic,
    _validate_hamiltonian,
    _validate_registers,
)

# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
# pylint: disable=no-value-for-parameter, unused-argument, arguments-differ


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
        cache (WiresLike): the :math:`2k` cache wires for squared/multiplied mode registers.
        coefficients (WiresLike): the :math:`b` wires of the Hamiltonian-coefficient register.
        phase_gradient (WiresLike): the :math:`b` wires holding the phase-gradient resource state.
        work (WiresLike): the :math:`\max(n-1, 2k, 2b+2)` work wires for data loading and
            arithmetic.
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
    wire_argnames = ("electronic", "vib_wires", "cache", "coefficients", "phase_gradient", "work")
    # ``num_trotter_steps`` drives Python-level control flow (the number of Trotter steps), and
    # ``aqft_order``/``diag_keys`` describe the compile-time circuit structure, so all three are
    # treated as (non-compilable) static arguments.
    static_argnames = ("num_trotter_steps", "aqft_order", "diag_keys")

    arg_specs = {
        "electronic": Wire[-1],
        "vib_wires": Wire[-1],
        "cache": Wire[-1],
        "coefficients": Wire[-1],
        "phase_gradient": Wire[-1],
        "work": Wire[-1],
    }

    def __init__(
        self,
        evolution_time,
        num_trotter_steps,
        hamiltonian,
        electronic: WiresLike,
        vib_wires: WiresLike,
        cache: WiresLike,
        coefficients: WiresLike,
        phase_gradient: WiresLike,
        work: WiresLike,
        aqft_order=None,
        diag_keys=None,
    ):
        _validate_hamiltonian(hamiltonian)
        if not isinstance(num_trotter_steps, int) or num_trotter_steps <= 0:
            raise ValueError(
                "The number of Trotter steps should be a positive integer, "
                f"but got {num_trotter_steps}."
            )
        if diag_keys is None:
            diag_keys = _derive_diag_keys(hamiltonian)

        super().__init__(
            evolution_time,
            num_trotter_steps,
            hamiltonian,
            electronic,
            vib_wires,
            cache,
            coefficients,
            phase_gradient,
            work,
            aqft_order,
            diag_keys,
        )

    def __abstract_init__(self, *args, **kwargs):
        # ``diag_keys`` depends on the concrete non-zero structure of the Hamiltonian. When this
        # operator is first constructed abstractly (e.g. with a traced ``evolution_time`` but a
        # concrete ``hamiltonian``), derive it here; on later reconstructions it is already set.
        bound = self._sig.bind(*args, **kwargs)
        bound.apply_defaults()
        if bound.arguments.get("diag_keys") is None:
            hamiltonian = bound.arguments["hamiltonian"]
            if any(math.is_abstract(v) for v in hamiltonian.values()):
                raise ValueError(
                    "TrotterVibronic requires `diag_keys` to be provided explicitly when the "
                    "`hamiltonian` is traced (abstract)."
                )
            bound.arguments["diag_keys"] = _derive_diag_keys(hamiltonian)
        super().__abstract_init__(*bound.args, **bound.kwargs)


def _trotter_vibronic_resources(
    evolution_time,
    num_trotter_steps,
    hamiltonian,
    electronic,
    vib_wires,
    cache,
    coefficients,
    phase_gradient,
    work,
    aqft_order,
    diag_keys,
):
    """Coarse (upper-bound) gate counts for the vibronic Trotter circuit.

    This estimate is intentionally inexact (``exact=False``): terms whose coefficients happen to
    vanish are skipped at runtime, and the sub-operations are counted at their top level.
    """
    num_fragments = hamiltonian["constant"].shape[0]
    n_modes = hamiltonian["linear"].shape[-1]
    n_elec = len(electronic)
    num_pairs = n_modes * (n_modes - 1) // 2

    # Each Trotter step visits every position fragment twice (forward + backward).
    position_visits = 2 * num_fragments * num_trotter_steps

    resources = defaultdict(int)
    # Electronic diagonalization (forward + adjoint per visit).
    resources[Hadamard] += 2 * position_visits
    resources[CNOT] += 2 * position_visits * max(n_elec - 1, 0)
    # Data loading: constant + each linear/quadratic/bilinear term + final unload.
    resources[QROM] += position_visits * (2 + 2 * n_modes + num_pairs)
    resources[SemiAdder] += position_visits
    # Linear terms use a half-signed multiplier (one OutMultiplier + broadcasted BasisState).
    resources[OutMultiplier] += position_visits * (n_modes + n_modes)
    resources[BasisState] += position_visits * 2 * (n_modes + num_pairs)
    # Quadratic terms use two SignedOutSquares and one OutMultiplier.
    resources[SignedOutSquare] += position_visits * 2 * n_modes
    # Bilinear terms use two SignedOutMultipliers and one half-signed multiplier.
    resources[SignedOutMultiplier] += position_visits * 2 * num_pairs
    # Kinetic fragment (once per Trotter step).
    resources[AQFT] += num_trotter_steps * 2 * n_modes
    resources[SignedOutSquare] += num_trotter_steps * 2 * n_modes
    resources[OutMultiplier] += num_trotter_steps * n_modes
    resources[BasisState] += num_trotter_steps * 2 * n_modes

    return dict(resources)


@register_resources(_trotter_vibronic_resources, exact=False)
def _trotter_vibronic_decomposition(
    evolution_time,
    num_trotter_steps,
    hamiltonian,
    electronic,
    vib_wires,
    cache,
    coefficients,
    phase_gradient,
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

    registers = {
        "electronic": electronic,
        "cache": cache,
        "coefficients": coefficients,
        "phase gradient": phase_gradient,
        "work": work,
    }
    _validate_registers(registers, mode_registers, n_modes, n_states)

    resolved_aqft_order = aqft_order if aqft_order is not None else mode_registers.shape[1] - 1

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


add_decomps(TrotterVibronic, _trotter_vibronic_decomposition)
