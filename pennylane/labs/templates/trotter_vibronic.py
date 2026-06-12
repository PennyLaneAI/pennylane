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
"""Contains the Trotter template for vibronic Hamiltonians."""

import numpy as np

import pennylane as qp


def float_to_binary(x: float, k: int) -> str:
    """Convert a floating-point number x to binary with k-bit precision."""
    # todo: figure out signed float-to-binary encoding that is compatible with phase gradients
    # in terms of scaling, range/period, and sign.
    return None


def get_coefficients(fragment: RealspaceMatrix, n_modes: int, n_degs: int) -> np.ndarray:
    """Get the coefficients of the fragment for the given number of modes and degrees"""
    frag_coeffs = [[] for _ in range(n_degs)]
    poly_vals = [("Q",) * (deg + 1) for deg in range(n_degs)]
    for pval in poly_vals:
        modes_coeffs = [[] for _ in range(n_modes)]
        for vals in fragment.get_coefficients().values():
            if vals:
                for mode in range(n_modes):
                    val = vals.get(pval, {}).get(tuple([mode] * len(pval)), 0.0)
                    modes_coeffs[mode].append(val)
        frag_coeffs[len(pval) - 1] = np.array(modes_coeffs)
    return np.array(frag_coeffs).swapaxes(0, 1)


def get_momentum_coefficients(n_modes: int, n_degs: int) -> np.ndarray:
    """Get the coefficients for the momentum fragment for the given number of modes and degrees."""
    # todo: figure out coefficients for momentum term omega_r


available_hamiltonians = {
    "anthra-c60_ct.pkl",
    "anthracene_6s_66m.pkl",
    "maleimide_5s_24m_bilin.pkl",
    "maleimide_5s_24m_nobilin.pkl",
    "no4a_dimer_6s_21s.pkl",
    "no4a_monomer_5s_19m.pkl",
    "pentacene_20s_102m.pkl",
}


def get_hamiltonian_fragments(**params):
    """Compute fragments of the vibronic Hamiltonian, using PennyLane functionality."""
    hamiltonian = params["hamiltonian"]

    if hamiltonian == "random":
        n_modes = params["n_modes"]
        n_states = params["n_states"]
        seed = params.get("seed", 42)
        rng = np.random.default_rng(seed)
        freqs = rng.random(n_modes)
        taylor_coeffs = [
            rng.random(size=(n_states, n_states)),
            rng.random(size=(n_states, n_states, n_modes)),
        ]
    else:
        if "n_modes" in params or "n_states" in params:
            raise ValueError(
                "The number of modes and states is currently dictated by the "
                f"Hamiltonian if hamiltonian!='random'. Got {hamiltonian=}"
            )
        # Read the pickle file
        if not hamiltonian.endswith(".pkl"):
            hamiltonian = hamiltonian + ".pkl"
        if hamiltonian not in available_hamiltonians:
            availables = "\n".join(available_hamiltonians)
            raise FileNotFoundError(
                f"Did not find file {hamiltonian}. Available hamiltonians: {availables}"
            )
        file_path = Path(__file__).parent / f"vibronic_hamiltonians/{hamiltonian}"
        with file_path.open("rb") as f:
            freqs, couplings = pickle.load(f)

        n_modes = len(freqs)
        n_states = np.shape(couplings[0])[0]
        taylor_coeffs = couplings.values()

    fragments = vibronic_fragments(n_states, n_modes, freqs, taylor_coeffs)
    return fragments, n_modes, n_states


def get_initial_states(**params):
    """Compute initial states for the vibronic workflow."""
    k = params["n_qubits"]
    wires = list(range(k))
    K = 1 << k
    X = np.arange(-K // 2, K // 2)
    alpha = np.exp(-np.pi / K)
    # Compute normalized Gaussian and shift it to match two's complement representation
    gaussian = alpha ** (X**2)
    gaussian /= np.linalg.norm(gaussian)
    gaussian = np.roll(gaussian, K // 2)

    # Todo: Add electronic state. Probably just |0>?
    electronic_initial_state = None
    return gaussian, electronic_initial_state


def _trotter_step_second_order(
    j, time, fragments, registers, params
):  # pylint: disable=unused-argument
    """Second-order Trotter time evolution step implemented via custom arithmetic circuits
    based on a phase gradient resource state.

    Args:
        TODO
        fragments (...): Assume that momentum fragment is the last entry in ``fragments``.

    The ordering of the time evolution fragments within the second-order Trotter step is fixed
    as follows:
    - the (first-order) momentum fragment evolution is put in the middle, wrapped by the
      second-order position fragment evolutions.
    - Within each position fragment evolution, we only have contributions in the same basis,
      so their ordering does not have any impact on the circuit. We first iterate over the
      degrees, including linear, quadratic, and bilinear terms (this iteration is not in form
      of a ``for_loop``). Within each of those three terms, we iterate over the modes (linear,
      quadratic) or pairs of modes (bilinear).

    """
    # pylint: disable=no-value-for-parameter
    n_modes, n_degs, aqft_order, prec_bits = params

    precision = len(registers["coefficients"])
    first_order_time_step = time / 2
    qrom_wires = {
        "control_wires": "electronic",
        "target_wires": "coefficients",
        "work_wires": "work",
    }
    qrom_wires = {new: registers[old] for new, old in qrom_wires.items()}

    def position_fragments(i):
        fragment = fragments[i]
        clifford_diagonalize(fragment, registers["electronic"])
        # Update the coefficient readout
        coeffs = get_coefficients(fragment, n_modes=n_modes, n_degs=n_degs) * first_order_time_step
        coeffs = qp.math.array(coeffs, like="jax")
        # Make linear_coeffs, quadratic_coeffs, bilinear_coeffs. optimally produced directly by
        # get_coefficients

        @qp.for_loop(n_modes)
        def linear_terms(k, prev_bitstrings):
            """Run a single linear time evolution sub-fragment for mode ``k``.
            The currently encoded bitstrings on the coefficients register are provided in
            ``prev_bitstrings``."""
            _coeffs = linear_coeffs[k]
            if np.allclose(_coeffs, 0):
                return prev_bitstrings
            mult_wires = {
                "x_wires": "coefficients",
                "y_wires": f"mode {k}",
                "output_wires": "phase gradient",
                "work_wires": "work",
            }
            mult_wires = {new: registers[old] for new, old in mult_wires.items()}
            new_bitstrings = float_to_binary(_coeffs, precision)
            change_bitstrings = np.bitwise_xor(prev_bitstrings, new_bitstrings)
            qp.QROM(change_bitstrings, **qrom_wires)
            qp.SignedOutMultiplier(**mult_wires)
            return new_bitstrings

        @qp.for_loop(n_modes)
        def quadratic_terms(k, prev_bitstrings):
            """Run a single quadratic time evolution sub-fragment for mode ``k``.
            The currently encoded bitstrings on the coefficients register is provided in
            ``prev_bitstrings``."""
            _coeffs = quadratic_coeffs[k]
            if np.allclose(_coeffs, 0):
                return prev_bitstrings

            square_wires = {"x_wires": f"mode {k}", "y_wires": "cache", "work_wires": "work"}
            square_wires = {new: registers[old] for new, old in square_wires.items()}
            mult_wires = {
                "x_wires": "coefficients",
                "y_wires": "cache",
                "output_wires": "phase gradient",
                "work_wires": "work",
            }
            mult_wires = {new: registers[old] for new, old in mult_wires.items()}

            new_bitstrings = float_to_binary(_coeffs, precision)
            change_bitstrings = np.bitwise_xor(prev_bitstrings, new_bitstrings)
            qp.QROM(change_bitstrings, **qrom_wires)
            qp.SignedOutSquare(**square_wires, output_wires_zeroed=True)
            qp.SignedOutMultiplier(**mult_wires)
            qp.adjoint(qp.SignedOutSquare)(**square_wires, output_wires_zeroed=True)
            return new_bitstrings

        @qp.for_loop(n_modes - 1)
        def bilinear_terms(k, prev_bitstrings):
            """Run a single quadratic time evolution sub-fragment for mode ``k``.
            The currently encoded bitstrings on the coefficients register is provided in
            ``prev_bitstrings``."""

            @qp.for_loop(k + 1, n_modes)
            def _inner_bilinear_terms(ell, prev_bitstrings):
                _coeffs = bilinear_coeffs[k, ell]
                if np.allclose(_coeffs, 0):
                    return prev_bitstrings

                mode_mult_wires = {
                    "x_wires": f"mode {k}",
                    "y_wires": f"mode {ell}",
                    "output_wires": "cache",
                    "work_wires": "work",
                }
                mode_mult_wires = {new: registers[old] for new, old in mode_mult_wires.items()}
                coeff_mult_wires = {
                    "x_wires": "coefficients",
                    "y_wires": "cache",
                    "output_wires": "phase gradient",
                    "work_wires": "work",
                }
                coeff_mult_wires = {new: registers[old] for new, old in coeff_mult_wires.items()}

                new_bitstrings = float_to_binary(_coeffs, precision)
                change_bitstrings = np.bitwise_xor(prev_bitstrings, new_bitstrings)
                qp.QROM(change_bitstrings, **qrom_wires)
                qp.SignedOutMultiplier(**mode_mult_wires, output_wires_zeroed=True)
                qp.SignedOutMultiplier(**coeff_mult_wires)
                qp.adjoint(qp.SignedOutMultiplier)(**mode_mult_wires, output_wires_zeroed=True)
                return new_bitstrings

            return _inner_bilinear_terms(prev_bitstrings)

        prev_bitstrings = np.zeros(precision, dtype=int)
        prev_bitstrings = linear_terms(prev_bitstrings)
        prev_bitstrings = quadratic_terms(prev_bitstrings)
        prev_bitstrings = bilinear_terms(prev_bitstrings)

        # Finish up the coefficients register by unloading the last loaded coefficients
        qp.QROM(prev_bitstrings, **qrom_wires)
        qp.adjoint(clifford_diagonalize)(fragment, wires["electronic"])

    def momentum_fragment():
        # use time, not first_order_time_step because the momentum fragment is the
        # middle one in second-order Trotter.
        # todo: Replace BasisEmbedding + SignedMultiplier by a classical-quantum multiplier?
        coeffs = get_momentum_coefficients(n_modes=n_modes, n_degs=n_degs) * time
        quadratic_momentum_coeffs = qp.math.array(coeffs, like="jax")

        @qp.for_loop(n_modes)
        def quadratic_momentum_terms(k):
            """Run a single quadratic time evolution sub-fragment for mode ``k``.
            The currently encoded bitstring on the coefficients register is provided in
            ``prev_bitstring``."""
            coeff = quadratic_momentum_coeffs[k]
            if np.isclose(coeff, 0):
                return
            square_wires = {"x_wires": f"mode {k}", "y_wires": "cache", "work_wires": "work"}
            square_wires = {new: registers[old] for new, old in square_wires.items()}
            mult_wires = {
                "x_wires": "coefficients",
                "y_wires": "cache",
                "output_wires": "phase gradient",
                "work_wires": "work",
            }
            mult_wires = {new: registers[old] for new, old in mult_wires.items()}
            x = float_to_binary(coeff, precision)
            qp.BasisEmbedding(x, registers["coefficients"])
            qp.AQFT(order=aqft_order, wires=wires[f"mode {k}"])
            qp.SignedOutSquare(**square_wires, output_wires_zeroed=True)
            qp.SignedOutMultiplier(**mult_wires)
            qp.adjoint(qp.SignedOutSquare)(**square_wires, output_wires_zeroed=True)
            qp.adjoint(qp.AQFT)(order=aqft_order, wires=wires[f"mode {k}"])
            qp.BasisEmbedding(x, wires["coefficients"])

        quadratic_momentum_terms()

    qp.for_loop(len(fragments) - 1)(position_fragments)()

    momentum_fragment()

    qp.for_loop(len(fragments) - 2, -1, -1)(position_fragments)()


def clifford_diagonalize(fragment: RealspaceMatrix, wires: WiresLike):
    """Diagonalize a fragment by applying Clifford operations.
    Based on Fig. 2 of arXiv:2411.13669. Correctly takes queuing into account."""
    # TODO: Consider moving to ctrl(BasisState).
    frag_keys = [k for k, v in fragment.get_coefficients().items() if v]
    ham_weight = bin(frag_keys[0][0] ^ frag_keys[0][1])[2:].zfill(len(wires))
    support = [i for i, x in enumerate(ham_weight) if x == "1"]
    if not support:
        return
    elif len(support) == 1:
        qp.H(wires=wires[support[0]])
        return

    control = max(support)
    h_op = qp.H(wires=wires[control])
    u_forward = [qp.CNOT(wires=[wires[control], wires[k]]) for k in support if k != control]
    return


def qs_circuit(num_mode: int, poly_deg: int, wires: dict[str, WiresLike]):
    """Implement phases generated by a monomial of position operators for a given
    polynomial degree."""
    # Todo: Make use of the fact that square cache is an unsigned register
    # todo: optimize between square and square†

    if poly_deg == 2:
        SignedOutSquare(
            wires[f"mode {num_mode}"],
            wires["square cache"],
            wires["work"],
            output_wires_zeroed=True,
        )
        SignedOutMultiplier(
            wires["coefficients"], wires["square cache"], wires["phase gradient"], wires["work"]
        )
        qp.adjoint(SignedOutSquare)(
            wires[f"mode {num_mode}"],
            wires["square cache"],
            wires["work"],
            output_wires_zeroed=True,
        )

    elif poly_deg == 1:
        SignedOutMultiplier(
            wires["coefficients"], wires[f"mode {num_mode}"], wires["phase gradient"], wires["work"]
        )

    else:
        raise NotImplementedError


"""
    fragments, n_modes, n_states = get_hamiltonian_fragments(**params)
    gaussian, electronic_initial_state = get_initial_states(**params)
    log_N = qp.math.ceil_log2(n_states)

    n_degs = params["n_degs"]
    n_qubits_per_mode = params["n_qubits"]
    prec_bits = qp.math.ceil_log2(2 * np.pi / params["delta"])

    phase_grad_precision = params["phase_grad_precision"]
    aqft_order = params["aqft_order"]

    registers = (
        {"electronic": log_N}
        | {f"mode {i}": n_qubits_per_mode for i in range(n_modes)}
        | {
            "square cache": 2 * n_qubits_per_mode,
            "coefficients": prec_bits,
            "phase gradient": prec_bits,
        }
        | {"work": 2 * prec_bits - 1}
    )
    num_wires = sum(registers.values())
    wires = qp.registers(registers)
"""


def trotter_vibronic(evolution_time, num_trotter_steps, fragments, registers):
    """Trotter circuit for vibronic simulation algorithm, using phase gradient arithmetic."""

    assert num_trotter_steps > 0
    trotter_time_step = evolution_time / num_trotter_steps
    _params = (n_modes, n_degs, aqft_order, prec_bits)

    @qp.for_loop(num_trotter_steps)
    def trotter_steps(step_idx):
        _trotter_step_second_order(
            step_idx,
            time=trotter_time_step,
            fragments=fragments,
            wires=wires,
            params=_params,
        )

    trotter_steps()  # pylint: disable=no-value-for-parameter
