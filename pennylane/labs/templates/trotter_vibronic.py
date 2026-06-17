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


def get_position_coefficients(fragment: RealspaceMatrix):
    """Get the coefficients for a given position fragment.
    Also validates that the terms in the fragment are limited to expected terms.

    Args:
        fragment (RealspaceMatrix): fragment of which to read the coefficients

    Returns:
        tuple[np.ndarray]: Dense coefficient tensors, arranged for QROM encoding by
        diagonalizing the electronic degrees of freedom. Returns a tuple of four tensors,
        corresponding to the constant, linear, quadratic, and bilinear dependencies on
        vibrational modes. The axes corresponding to mode indices are the first axes, followed
        by a single axis for the electronic state index.

    """
    n_states = fragment.states
    n_modes = fragment.modes
    wires = list(range(qp.math.ceil_log2(n_states)))
    M = qp.matrix(diagonalize_vibronic, wires)(fragment, wires)[:n_states, :n_states]
    # constant.shape = (n_states, n_states)
    constant = M.T @ fragment_to_dense(fragment, ()) @ M
    # constant_diag.shape = (n_states,)
    constant_diag = np.diag(constant)
    # Make sure the diagonalization worked
    assert np.allclose(np.diag(constant_diag), constant)

    # linear.shape = (n_modes, n_states, n_states)
    linear = np.einsum("ba,bcz,cd->zad", M, fragment_to_dense(fragment, ("Q",)), M)
    # linear_diag.shape = (n_modes, n_states)
    linear_diag = np.diagonal(linear, axis1=1, axis2=2)
    # Make sure the diagonalization worked
    assert all(
        np.allclose(np.diag(sub_diag), sub)
        for sub_diag, sub in zip(linear_diag, linear, strict=True)
    )

    # sec_order.shape = (n_modes, n_modes, n_states, n_states)
    sec_order = np.einsum("ba,bcyz,cd->yzad", M, fragment_to_dense(fragment, ("Q", "Q")), M)
    # sec_order_diag.shape = (n_modes, n_modes, n_states)
    sec_order_diag = np.diagonal(sec_order, axis1=2, axis2=3)
    # Make sure the diagonalization worked
    assert all(
        np.allclose(np.diag(sub_diag), sub)
        for sub_diag, sub in zip(
            sec_order_diag.reshape((n_modes**2, -1)),
            sec_order.reshape((n_modes**2, n_states, n_states)),
            strict=True,
        )
    )

    # quadratic.shape = (n_modes, n_states)
    quadratic = np.diagonal(sec_order_diag).copy()
    # bilinear.shape = (n_modes, n_modes, n_states), upper triangle w.r.t. axes=(0, 1) populated
    bilinear = sec_order_diag
    bilinear[np.tril_indices(n_modes)] = 0.0
    return constant_diag, linear_diag, quadratic, bilinear


def get_momentum_coefficients(fragment: RealspaceMatrix):
    """Get the coefficients for a given momentum fragment.
    Also validates that the terms in the fragment are limited to expected terms.

    Args:
        fragment (RealspaceMatrix): fragment of which to read the coefficients

    Returns:
        np.ndarray: Array of momentum coefficients with shape ``(fragment.modes,)``

    """
    n_states = fragment.states
    n_modes = fragment.modes
    # coeffs.shape = (n_modes, n_modes, n_states, n_states)
    coeffs = fragment_to_dense(fragment, ("P", "P"))
    # coeffs_diag.shape = (n_modes, n_modes, n_states)
    coeffs_diag = np.diagonal(coeffs, axis1=2, axis2=3)
    # Make sure the kinetic term is diagonal with respect to electronic states
    assert all(
        np.allclose(np.diag(sub_diag), sub)
        for sub_diag, sub in zip(
            coeffs_diag.reshape((n_modes**2, -1)),
            coeffs.reshape((n_modes**2, n_states, n_states)),
            strict=True,
        )
    )
    # coeffs_diag2.shape = (n_modes, n_states)
    coeffs_diag2 = np.diagonal(coeffs_diag)
    # Make sure the kinetic term is diagonal with respect to modes
    assert all(
        np.allclose(np.diag(sub_diag2), sub)
        for sub_diag2, sub in zip(coeffs_diag2.T, coeffs_diag.T, strict=True)
    )
    # coeffs_final.shape = (n_modes,)
    coeffs_final = coeffs_diag2[:, 0]
    # Make sure the kinetic term is encoded redundantly across electronic states, i.e. there is
    # not interaction between electronic state and vibrational kinetic energy
    assert np.allclose(coeffs_diag2, coeffs_final[:, None])
    return coeffs_final


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
        # No diagonalization needed
        return
    diagonalization_key = qp.math.int_to_binary(first_coeff_key[0] ^ first_coeff_key[1], len(wires))
    support = np.where(diagonalization_key)[0][::-1]
    control = support[0]
    qp.H(wires=wires[control])
    _ = [qp.CNOT(wires=[wires[control], wires[k]]) for k in support[1:]]


def _trotter_step_second_order(_, time, fragments, registers, aqft_order):
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
        qp.adjoint(diagonalize_vibronic)(fragment, registers["electronic"])
        all_coeffs = get_position_coefficients(fragment)
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

        @qp.for_loop(fragment.modes)
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

        @qp.for_loop(fragment.modes)
        def quadratic_terms(k, prev_bitstrings):
            """Run a single quadratic time evolution sub-fragment for mode ``k``.
            The currently encoded bitstrings on the coefficients register are provided in
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

        @qp.for_loop(fragment.modes - 1)
        def bilinear_terms(k, prev_bitstrings):
            """Run a single bilinear time evolution sub-fragment for mode ``k``.
            The currently encoded bitstrings on the coefficients register are provided in
            ``prev_bitstrings``."""

            # Note that k < ell matches the data structure of bilinear_coeffs, which is only
            # populated in the upper triangular part, w.r.t. the mode axes.
            @qp.for_loop(k + 1, fragment.modes)
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

    def kinetic_fragment(fragment, aqft_order):
        # use time, not first_order_time_step because the kinetic fragment is the
        # middle one in second-order Trotter, so we immediately merge the two neighbouring
        # fragments with first_order_time_step duration.
        # todo: Replace BasisState + SignedMultiplier by a classical-quantum multiplier?
        kinetic_coeffs = get_momentum_coefficients(fragment)
        kinetic_coeffs = qp.math.array(kinetic_coeffs, like="jax") * time

        @qp.for_loop(fragment.modes)
        def kinetic_terms(k):
            """Run a single quadratic time evolution sub-fragment for mode ``k`` in momentum space."""
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

    kinetic_fragment(fragments[-1], aqft_order)

    qp.for_loop(len(fragments) - 2, -1, -1)(position_fragments)()


def trotter_vibronic(evolution_time, num_trotter_steps, fragments, registers, aqft_order):
    r"""Second-order Trotter circuit for vibronic simulation algorithm, explicitly using phase
    gradient arithmetic.

    Args:
        evolution_time (float): Time for which to evolve under the vibronic Hamiltonian
        num_trotter_steps (int): Number of Trotter steps to use for the time evolution
        fragments (list[RealspaceMatrix]): Fragments of the vibronic Hamiltonian
        registers (dict[str, WiresLike]): Wire registers. See details below.
        aqft_order (int): Approximation order used in :class:`~.AQFT`, which is used to transform
            between position and momentum space.

    .. details::
        :title: Usage details
        :href: usage-details

        The vibronic Hamiltonian acts on :math:`N` electronic states, represented on
        :math:`n=\lceil\log_2(N)\rceil` qubits, as well as :math:`M` vibrational modes,
        each discretized on a grid with :math:`K` grid points, requiring
        :math:`k=\lceil\log_2(K)\rceil` qubits per mode. Phases are implemented directly with
        arithmetic acting on a phase gradient state with :math:`b` qubits.

        According to this setup, the ``registers`` passed to this function should include the
        following wire registers:

        .. list-table::
           :widths: 30 40 30
           :header-rows: 1

           * - ``key``
             - Expected size
             - Comment
           * - ``"electronic"``
             - :math:`n`
             - Encodes electronic states
           * - ``f"mode {i}"``
             - :math:`k`
             - Encodes the :math:`i`\ th vibrational mode
           * - ``"cache"``
             - :math:`???` # TODO: figure out
             - Cache storing computations from one or two vibrational modes
           * - ``"coefficients"``
             - :math:`b`
             - Coefficient encoding register for phase gradient arithmetic
           * - ``"phase gradient"``
             - :math:`b`
             - Stores the phase gradient state
           * - ``"work"``
             - :math:`???` # TODO: figure out
             - Work wires for phase gradient arithmetic # TODO: VERIFY

    """

    assert num_trotter_steps > 0
    trotter_time_step = evolution_time / num_trotter_steps

    @qp.for_loop(num_trotter_steps)
    def trotter_steps(step_idx):
        _trotter_step_second_order(
            step_idx,
            time=trotter_time_step,
            fragments=fragments,
            registers=registers,
            aqft_order=aqft_order,
        )

    trotter_steps()  # pylint: disable=no-value-for-parameter
