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
"""Utility functions for the :class:`~.TrotterVibronic` template."""

from functools import reduce

import numpy as np

from pennylane import capture, compiler, math
from pennylane.control_flow import for_loop
from pennylane.ops import CNOT, BasisState, Hadamard, adjoint, cond, ctrl

from ..aqft import AQFT
from ..arithmetic.out_multiplier import OutMultiplier
from ..arithmetic.semi_adder import SemiAdder
from ..arithmetic.signed_out_multiplier import SignedOutMultiplier, _twos_complement_helper
from ..arithmetic.signed_out_square import SignedOutSquare
from ..qrom import QROM

# pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
# pylint: disable=no-value-for-parameter, too-many-statements

#: Keys expected in the dense vibronic Hamiltonian dictionary.
HAMILTONIAN_KEYS = ("constant", "linear", "quadratic", "kinetic")


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
    """
    y_aux, work_wires = work_wires[0], work_wires[1:]

    # Cache the sign bit of ``y`` on the auxiliary wire.
    CNOT([y_wires[0], y_aux])

    # Take the two's complement of ``y`` controlled on the cached sign bit.
    _twos_complement_helper(y_wires, y_aux, work_wires)

    # Multiply the magnitudes into the output register. If ``y`` was negative, flip all output
    # qubits before and after the (unsigned) multiplication, effectively subtracting the
    # product. ``BasisState`` acts as a compact broadcasted PauliX across all output wires.
    ctrl(BasisState([1] * len(output_wires), output_wires), control=y_aux)
    OutMultiplier(x_wires, y_wires, output_wires, work_wires=work_wires, output_wires_zeroed=False)
    ctrl(BasisState([1] * len(output_wires), output_wires), control=y_aux)

    # Uncompute the two's complement and the cached sign bit.
    _twos_complement_helper(y_wires, y_aux, work_wires)
    CNOT([y_wires[0], y_aux])


# ---------------------------------------------------------------------------
# ------------------------- Coefficient preprocessing -----------------------
# ---------------------------------------------------------------------------


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


def load_coefficients(coefficients, precision, prev_bitstrings, qrom_wires):
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
                new_bitstrings = load_coefficients(
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
                new_bitstrings = load_coefficients(_coeffs, precision, prev_bitstrings, qrom_wires)
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
                new_bitstrings = load_coefficients(_coeffs, precision, prev_bitstrings, qrom_wires)
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
                new_bitstrings = load_coefficients(_coeffs, precision, prev_bitstrings, qrom_wires)
                SignedOutMultiplier(**mode_mult_wires, output_wires_zeroed=True)
                _half_signed_out_multiplier(**coeff_mult_wires)
                adjoint(SignedOutMultiplier(**mode_mult_wires, output_wires_zeroed=True))
                return new_bitstrings

            return cond(math.allclose(_coeffs, 0.0), skip_fn, actual_fn)()

        prev_bitstrings = np.zeros((n_states, precision), dtype=int)
        prev_bitstrings = constant_term(prev_bitstrings)
        prev_bitstrings = linear_terms(prev_bitstrings)
        prev_bitstrings = quadratic_terms(prev_bitstrings)
        # The number of mode pairs is static; skip the bilinear loop entirely when there are none
        # (a zero-iteration ``for_loop`` is still traced once under program capture, which would
        # index into an empty coefficient array).
        if bilinear_indices.shape[1] > 0:
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
                BasisState(bitstring, registers["coefficients"])
                AQFT(order=aqft_order, wires=mode_registers[k])
                SignedOutSquare(**square_wires, output_wires_zeroed=True)
                OutMultiplier(**mult_wires)
                adjoint(SignedOutSquare(**square_wires, output_wires_zeroed=True))
                adjoint(AQFT)(order=aqft_order, wires=mode_registers[k])
                adjoint(BasisState)(bitstring, registers["coefficients"])

            cond(math.allclose(_coeffs, 0.0), skip_fn, actual_fn)()

        kinetic_terms()

    # The number of position fragments is static, so we iterate over them with a plain Python
    # loop (unrolled in the circuit). This keeps the per-fragment diagonalization keys concrete,
    # so the electronic-diagonalization circuit structure is fixed at compile time.
    for i in range(num_position_fragments):
        position_fragments(i)
    kinetic_fragment()
    for i in range(num_position_fragments - 1, -1, -1):
        position_fragments(i)


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
