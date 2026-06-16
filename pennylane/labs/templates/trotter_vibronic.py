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
from itertools import combinations

import numpy as np

import pennylane as qp
from pennylane.labs.trotter_error.realspace import RealspaceMatrix
from pennylane.wires import WiresLike


def float_to_binary(x: float, k: int) -> str:
    """Convert a floating-point number x to binary with k-bit precision."""
    # todo: figure out signed float-to-binary encoding that is compatible with phase gradients
    # in terms of scaling, range/period, and sign.
    return None


def fragment_to_dense(fragment: RealspaceMatrix, op_type: tuple[str]):
    """Test helper function that converts the coefficients for a specific operator type
    (e.g. ``("P", "P")`` or ``("Q",)``) of a vibronic fragment into a dense matrix. The output
    shape depends on ``op_type``.

    Args:
        fragment (RealspaceMatrix): vibronic fragment from which to extract the coefficients.
        n_states (int): number of electronic states
        n_modes (int): number of vibrational modes
        op_type (tuple[str]): operator type for which to extract the coefficients.

    Returns:
        np.ndarray: dense coefficients tensor for the specific operator type. Has shape
        ``(n_states, n_states) + (n_modes,) * len(op_type)``.
    """
    n_states = fragment.states
    n_modes = fragment.modes
    order = len(op_type)
    dense = np.zeros((n_states, n_states) + (n_modes,) * order)
    for elec_key, val in fragment.get_coefficients().items():
        terms = val.get(op_type, None)
        if terms is None:
            continue
        if order == 0:
            dense[elec_key] = terms.get((), 0.0)
            continue
        for ids_and_modes in combinations(range(n_modes), r=order):
            ids, modes = zip(*ids_and_modes)
            dense[elec_key][ids] = terms.get(modes, 0.0)
    return dense


def get_coefficients(fragment: RealspaceMatrix, fragment_type: str):
    """Get the coefficients for a given position or momentum fragment.
    Also validates that the terms in the fragment are limited to expected terms.

    Args:
        fragment (RealspaceMatrix): fragment of which to read the coefficients
        fragment_type (str): type of the fragment. This is redundant to the operator types within
            ``fragment`` and is used as consistency check. Valid options are ``"position"`` and
            ``"momentum"``.

    Returns:
        tuple[np.ndarray] or np.ndarray: Dense coefficient tensors, arranged for QROM encoding by
        diagonalizing the electronic degrees of freedom. Returns a tuple of four tensors for
        ``fragment_type="position"`` and a single tensor for ``fragment_type="momentum"``.

    """
    if fragment_type == "position":
        n_states = fragment.states
        wires = list(range(qp.math.ceil_log2(n_states)))
        M = qp.matrix(diagonalize_vibronic, wires)(fragment, wires)[:n_states, :n_states]
        constant = M.T @ fragment_to_dense(fragment, ()) @ M
        constant_diag = np.diag(constant)
        assert np.allclose(np.diag(constant_diag), constant)

        linear = np.einsum("ba,bcz,cd->adz", M, fragment_to_dense(fragment, ("Q",)), M)
        linear_diag = np.diagonal(linear).T # np.diagonal puts the diagonal dimension to the end
        assert all(np.allclose(np.diag(sub_diag), sub) for sub_diag, sub in zip(linear_diag.T, np.transpose(linear, (2, 0, 1)), strict=True))

        sec_order = np.einsum("ba,bcwz,cd->adwz", M, fragment_to_dense(fragment, ("Q", "Q")), M)
        # np.diagonal puts the diagonal dimension to the end
        sec_order_diag = np.transpose(np.diagonal(sec_order), (2, 0, 1))
        assert all(np.allclose(np.diag(sub_diag), sub) for sub_diag, sub in zip(np.transpose(sec_order_diag, (1, 2, 0).reshape((n_modes**2, -1)), np.transpose(sec_order, (2, 3, 0, 1)).reshape((n_modes**2, n_states,n_states)), strict=True)))

        quadratic = np.diagonal(sec_order_diag, axis1=1, axis2=2)
        bilinear =




    n_states = fragment.states
    n_modes = fragment.modes
    if fragment_type == "position":
        # Note that per fragment, all of the following arrays only will be populated with one
        # entry per row and per column, so we can improve in terms of memory usage.
        const_coeffs = np.zeros(n_states)
        linear_coeffs = np.zeros((n_states, n_modes))
        quadratic_coeffs = np.zeros((n_states, n_modes))
        bilinear_coeffs = np.zeros((n_states, n_modes, n_modes))

        for (row, _), val in fragment.get_coefficients().items():
            if not val:
                continue
            # Make sure we are not silently dropping any terms
            assert all(op_type in [(), ("Q",), ("Q", "Q")] for op_type in val), f"{val.keys()=}"

            _order_zero = val.get((), None)
            if _order_zero is not None:
                const_coeffs[row] = _order_zero.get((), 0.0)
            _order_one = val.get(("Q",), None)
            if _order_one is not None:
                linear_coeffs[row] = [_order_one.get((mode,), 0.0) for mode in range(n_modes)]
            _order_two = val.get(("Q", "Q"), None)
            if _order_two is not None:
                quadratic_coeffs[row] = [_order_two.get((mode, mode), 0.0) for mode in range(n_modes)]
                bilinear_coeffs[row][np.triu_indices(n_modes, k=1)] = [_order_two.get(mode_pair, 0.0) for mode_pair in combinations(range(n_modes), r=2)]
        return const_coeffs, linear_coeffs, quadratic_coeffs, bilinear_coeffs

    assert fragment_type == "momentum"
    # We will retrieve the redundantly stored frequencies and validate that they are actually
    # redundant, so that no interaction with the electronic system exists.
    quadratic_coeffs = np.zeros((n_states, n_modes))
    for (row, col), val in fragment.get_coefficients().items():
        if not val:
            continue
        # The momentum term is stored on the diagonal of the electronic coupling registry.
        # This means it is stored redundantly n_states times and for off-diagonal entries, the
        # above skipping clause should have triggered
        assert row == col
        # Make sure we are not silently dropping any terms - momentum should only have (P, P) term.
        assert set(val) == {("P", "P")}
        _order_two = val.get(("P", "P"))
        # No mode interaction terms, just quadratic kinetic terms
        assert all(k[0] == k[1] for k in _order_two)
        quadratic_coeffs[row] = [_order_two.get((mode, mode), 0.0) for mode in range(n_modes)]

    assert np.allclose(quadratic_coeffs, quadratic_coeffs[0]) # Validate redundancy
    return quadratic_coeffs[0]

def diagonalize_vibronic(fragment: RealspaceMatrix, wires: WiresLike):
    r"""Diagonalize a vibronic fragment by applying Clifford operations.
    Based on Fig.2 of `Motlagh et al, arXiv:2411.13669 <https://arxiv.org/abs/2411.13669>`__.

    Args:
        fragment (RealspaceMatrix): vibronic fragment to be diagonalized.
        wires (WiresLike): electronic wires on which the fragment acts. These wires are the only
            ones to which we need to apply operations for the diagonalization.

    .. warning::

        Note that this function is tailored to the vibronic fragments computed by
        :func:`~.pennylane.labs.trotter_error.vibronic_fragments`. In particular, only
        one key with non-zero coefficients is read out of ``fragment``, assuming that all other
        populated coefficients follow the pattern imposed by
        ``vibronic_fragments``.

    Note that the diagonalization only is required on the electronic register, which should be
    passed in as the ``wires`` argument, and that it only requires one :class:`~.Hadamard` gate
    and at most :math:`\lceil\log_2(N)\rceil-1` :class:`~.CNOT` gates, where :math:`N` is the
    number of electronic states the Hamiltonian acts on.
    """
    first_coeff_key = next(iter(k for k, v in fragment.get_coefficients().items() if v))
    if first_coeff_key[0] == first_coeff_key[1]:
        return
    diagonalization_key = qp.math.int_to_binary(first_coeff_key[0] ^ first_coeff_key[1], len(wires))
    support = np.where(diagonalization_key)[0][::-1]
    control = support[0]
    qp.H(wires=wires[control])
    [qp.CNOT(wires=[wires[control], wires[k]]) for k in support[1:]]
    return

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
    n_states, n_modes, aqft_order, prec_bits = params

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
        qp.adjoint(diagonalize_vibronic)(fragment, wires["electronic"])
        all_coeffs = get_coefficients(fragment, n_states, n_modes, fragment_type="position")
        all_coeffs = (qp.math.array(c, like="jax") * first_order_time_step for c in all_coeffs)
        const_coeffs, linear_coeffs, quadratic_coeffs, bilinear_coeffs = all_coeffs

        def constant_term(prev_bitstrings):
            if np.allclose(const_coeffs, 0):
                return prev_bitstrings
            new_bitstrings = float_to_binary(const_coeffs, precision)
            change_bitstrings = np.bitwise_xor(prev_bitstrings, new_bitstrings)
            qp.QROM(change_bitstrings, **qrom_wires)
            qp.SemiAdder(registers["coefficients"], registers["phase gradient"], registers["work"])
            return new_bitstrings

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

            # Note that k < ell matches the data structure of bilinear_coeffs
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
        prev_bitstrings = constant_term(prev_bitstrings)
        prev_bitstrings = linear_terms(prev_bitstrings)
        prev_bitstrings = quadratic_terms(prev_bitstrings)
        prev_bitstrings = bilinear_terms(prev_bitstrings)

        # Finish up the coefficients register by unloading the last loaded coefficients
        qp.QROM(prev_bitstrings, **qrom_wires)
        diagonalize_vibronic(fragment, registers["electronic"])

    def kinetic_fragment():
        # use time, not first_order_time_step because the kinetic fragment is the
        # middle one in second-order Trotter, so we immediately merge the two neighbouring
        # fragments with first_order_time_step duration.
        # todo: Replace BasisState + SignedMultiplier by a classical-quantum multiplier?
        kinetic_coeffs = get_coefficients(fragment, n_states, n_modes, fragment_type="momentum")
        kinetic_coeffs = qp.math.array(kinetic_coeffs, like="jax") * time

        @qp.for_loop(n_modes)
        def kinetic_terms(k):
            """Run a single quadratic time evolution sub-fragment for mode ``k`` in momentum space.
            """
            coeff = kinetic_coeffs[k]
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
            qp.BasisState(x, registers["coefficients"])
            qp.AQFT(order=aqft_order, wires=registers[f"mode {k}"])
            qp.SignedOutSquare(**square_wires, output_wires_zeroed=True)
            qp.SignedOutMultiplier(**mult_wires)
            qp.adjoint(qp.SignedOutSquare)(**square_wires, output_wires_zeroed=True)
            qp.adjoint(qp.AQFT)(order=aqft_order, wires=registers[f"mode {k}"])
            qp.BasisState(x, registers["coefficients"])

        kinetic_terms()

    qp.for_loop(len(fragments) - 1)(position_fragments)()

    kinetic_fragment()

    qp.for_loop(len(fragments) - 2, -1, -1)(position_fragments)()



def trotter_vibronic(evolution_time, num_trotter_steps, fragments, registers):
    """Trotter circuit for vibronic simulation algorithm, explicitly using phase gradient
    arithmetic.

    Args:

    """

    assert num_trotter_steps > 0
    trotter_time_step = evolution_time / num_trotter_steps
    _params = (n_modes, aqft_order, prec_bits)

    @qp.for_loop(num_trotter_steps)
    def trotter_steps(step_idx):
        _trotter_step_second_order(
            step_idx,
            time=trotter_time_step,
            fragments=fragments,
            registers=registers,
            params=_params,
        )

    trotter_steps()  # pylint: disable=no-value-for-parameter
