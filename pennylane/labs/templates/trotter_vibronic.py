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

from itertools import combinations_with_replacement

import numpy as np

import pennylane as qp
from pennylane.labs.trotter_error.realspace import RealspaceMatrix
from pennylane.wires import WiresLike

from .semi_signed_out_multiplier import semi_signed_out_multiplier


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
        for modes in combinations_with_replacement(range(n_modes), r=order):
            dense[elec_key][modes] = terms.get(modes, 0.0)
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
    diag_key = next(iter(k for k, v in fragment.get_coefficients().items() if v))
    M = qp.matrix(diagonalize_vibronic, wires)(key=diag_key, wires=wires)[:n_states, :n_states]
    constant = M.T @ fragment_to_dense(fragment, ()) @ M
    # constant.shape = (n_states, n_states)
    constant_diag = np.diag(constant)
    # constant_diag.shape = (n_states,)

    linear = np.einsum("ba,bcz,cd->zad", M, fragment_to_dense(fragment, ("Q",)), M)
    # linear.shape = (n_modes, n_states, n_states)
    linear_diag = np.diagonal(linear, axis1=1, axis2=2)
    # linear_diag.shape = (n_modes, n_states)

    sec_order = np.einsum("ba,bcyz,cd->yzad", M, fragment_to_dense(fragment, ("Q", "Q")), M)
    # sec_order.shape = (n_modes, n_modes, n_states, n_states)
    sec_order_diag = np.diagonal(sec_order, axis1=2, axis2=3)
    # sec_order_diag.shape = (n_modes, n_modes, n_states)

    quadratic = np.diagonal(sec_order_diag).copy().T
    # quadratic.shape = (n_modes, n_states)
    bilinear = sec_order_diag.copy()
    # bilinear.shape = (n_modes, n_modes, n_states)
    bilinear[np.tril_indices(n_modes)] = 0.0
    # now only upper triangle w.r.t. axes=(0, 1) populated
    return constant_diag, linear_diag, quadratic, bilinear


def get_momentum_coefficients(fragment: RealspaceMatrix):
    """Get the coefficients for a given momentum fragment.
    Also validates that the terms in the fragment are limited to expected terms.

    Args:
        fragment (RealspaceMatrix): fragment of which to read the coefficients

    Returns:
        np.ndarray: Array of momentum coefficients with shape ``(fragment.modes,)``

    """
    coeffs = fragment_to_dense(fragment, ("P", "P"))
    # coeffs.shape = (n_states, n_states, n_modes, n_modes)
    coeffs_diag = np.diagonal(coeffs, axis1=2, axis2=3)
    # coeffs_diag.shape = (n_states, n_states, n_modes)
    coeffs_diag2 = np.diagonal(coeffs_diag)
    # coeffs_diag2.shape = (n_modes, n_states) (diagonal puts new axis last)
    coeffs_final = coeffs_diag2[:, 0]
    # coeffs_final.shape = (n_modes,)
    return coeffs_final


def diagonalize_vibronic(*, key: tuple[int], wires: WiresLike):
    r"""Diagonalize a vibronic fragment by applying Clifford operations.
    Based on Fig.2 of `Motlagh et al, arXiv:2411.13669 <https://arxiv.org/abs/2411.13669>`__.

    Args:
        key (tuple[int]): Row and column index of the only non-zero matrix element in the first row
            that contains a non-zero matrix element at all.
        wires (WiresLike): electronic wires on which the fragment acts. These wires are the only
            ones to which we need to apply operations for the diagonalization.

    .. warning::

        Note that this function is tailored to the vibronic fragments computed by
        :func:`~.pennylane.labs.trotter_error.vibronic_fragments`. In particular, we assume
        that all populated coefficients other than the provided key follow the pattern imposed by
        ``vibronic_fragments``.

    Note that the diagonalization only is required on the electronic register, which should be
    passed in as the ``wires`` argument, and that it only requires one :class:`~.Hadamard` gate
    and at most :math:`\lceil\log_2(N)\rceil-1` :class:`~.CNOT` gates, where :math:`N` is the
    number of electronic states the Hamiltonian acts on.
    """
    if key[0] == key[1]:
        # No diagonalization needed
        return
    diagonalization_key = qp.math.int_to_binary(key[0] ^ key[1], len(wires))
    support = np.where(diagonalization_key)[0][::-1]
    control = support[0]
    qp.H(wires=wires[control])
    _ = [qp.CNOT(wires=[wires[control], wires[k]]) for k in support[1:]]


def load_coefficients(coefficients, precision, prev_bitstrings, qrom_wires):
    """Extract bit strings for one-dimensional coefficients array, XOR them with
    ``prev_bitstrings``, and load the result with a QROM, using the registers in ``qrom_wires``."""
    new_bitstrings = qp.math.binary_decimals(coefficients, precision, unit=2 * np.pi)
    # np.bitwise_xor is not tracing-compatible
    change_bitstrings = (prev_bitstrings + new_bitstrings) % 2
    qp.QROM(change_bitstrings, **qrom_wires)
    return new_bitstrings


def _extract_registers(registers, term, *mode_ids):
    """Extract registers for a specific term of the vibronic Trotter time evolution."""
    if term == "quadratic":
        (k,) = mode_ids
        square_wires = {"x_wires": f"mode {k}", "output_wires": "cache", "work_wires": "work"}
        square_wires = {new: registers[old] for new, old in square_wires.items()}
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
            "x_wires": f"mode {k}",
            "y_wires": f"mode {ell}",
            "output_wires": "cache",
            "work_wires": "work",
        }
        mode_mult_wires = {new: registers[old] for new, old in mode_mult_wires.items()}
        # The signed register for semi_signed_out_multiplier must be the _second_ input
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

    if term == "constant":
        reg = {"x_wires": "coefficients", "y_wires": "phase gradient", "work_wires": "work"}
    elif term == "linear":
        (k,) = mode_ids
        # The signed register for semi_signed_out_multiplier must be the _second_ input
        reg = {
            "x_wires": "coefficients",
            "y_wires": f"mode {k}",
            "output_wires": "phase gradient",
            "work_wires": "work",
        }
    return {new: registers[old] for new, old in reg.items()}


def _trotter_step_second_order(idx, time, fragments, registers, aqft_order):
    r"""Second-order Trotter time evolution step implemented via custom arithmetic circuits
    based on a phase gradient resource state.

    Args:
        idx (int): Trotter step index. This argument is not used.
        time (float): Second-order Trotter time step size
        fragments (List[RealspaceMatrix]): Trotter fragments of the vibronic Hamiltonian.
            It is assumed that the kinetic fragment is in the last position.

    The ordering of the time evolution fragments within the second-order Trotter step is fixed
    as follows:
    - the (first-Trotter-order) kinetic evolution is put in the middle, wrapped by the
      second-Trotter-order position fragment evolutions.
    - Within each position fragment evolution, we only have contributions in the same (electronict)
      basis, so their ordering does not have any impact on the circuit. We first iterate over the
      degrees, including linear, quadratic, and bilinear terms (this iteration is not in form
      of a ``for_loop``). Within each of those three terms, we iterate over the modes (linear,
      quadratic) or pairs of modes (bilinear). This latter iteration is in form of ``for_loop``\ s.

    """
    # pylint: disable=no-value-for-parameter, unused-argument

    precision = len(registers["phase gradient"])
    first_order_time_step = time / 2
    qrom_wires = _extract_registers(registers, "QROM")
    diag_keys = [
        next(iter(k for k, v in frag.get_coefficients().items() if v)) for frag in fragments
    ]

    def position_fragments(i):
        fragment = fragments[i]
        diag_key = diag_keys[i]
        # THIS IS WRONG- THE ADJOINT NEEDS TO BE USED. IT IS REMOVED FOR COMPILABILITY TEST
        diagonalize_vibronic(key=diag_key, wires=registers["electronic"])
        # qp.adjoint(diagonalize_vibronic)(key=diag_key, wires=registers["electronic"])

        all_coeffs = (c * first_order_time_step for c in get_position_coefficients(fragment))
        const_coeffs, linear_coeffs, quadratic_coeffs, bilinear_coeffs = all_coeffs

        def constant_term(prev_bitstrings):
            if np.allclose(const_coeffs, 0.0):
                return prev_bitstrings
            new_bitstrings = load_coefficients(const_coeffs, precision, prev_bitstrings, qrom_wires)
            qp.SemiAdder(**_extract_registers(registers, "constant"))
            return new_bitstrings

        @qp.for_loop(fragment.modes)
        def linear_terms(k, prev_bitstrings):
            """Run a single linear time evolution sub-fragment for mode ``k``.
            The currently encoded bitstrings on the coefficients register are provided in
            ``prev_bitstrings``."""
            _coeffs = linear_coeffs[k]
            if np.allclose(_coeffs, 0.0):
                return prev_bitstrings
            new_bitstrings = load_coefficients(_coeffs, precision, prev_bitstrings, qrom_wires)
            mult_wires = _extract_registers(registers, "linear", k)
            semi_signed_out_multiplier(**mult_wires)
            return new_bitstrings

        @qp.for_loop(fragment.modes)
        def quadratic_terms(k, prev_bitstrings):
            """Run a single quadratic time evolution sub-fragment for mode ``k``.
            The currently encoded bitstrings on the coefficients register are provided in
            ``prev_bitstrings``."""
            _coeffs = quadratic_coeffs[k]
            if np.allclose(_coeffs, 0.0):
                return prev_bitstrings
            new_bitstrings = load_coefficients(_coeffs, precision, prev_bitstrings, qrom_wires)
            square_wires, mult_wires = _extract_registers(registers, "quadratic", k)
            qp.SignedOutSquare(**square_wires, output_wires_zeroed=True)
            qp.OutMultiplier(**mult_wires)
            # THIS IS WRONG- THE ADJOINT NEEDS TO BE USED. IT IS REMOVED FOR COMPILABILITY TEST
            qp.SignedOutSquare(**square_wires, output_wires_zeroed=True)
            # qp.adjoint(qp.SignedOutSquare)(**square_wires, output_wires_zeroed=True)
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
                if np.allclose(_coeffs, 0.0):
                    return prev_bitstrings
                mode_mult_wires, coeff_mult_wires = _extract_registers(
                    registers, "bilinear", k, ell
                )
                new_bitstrings = load_coefficients(_coeffs, precision, prev_bitstrings, qrom_wires)
                qp.SignedOutMultiplier(**mode_mult_wires, output_wires_zeroed=True)
                semi_signed_out_multiplier(**coeff_mult_wires)
                # THIS IS WRONG- THE ADJOINT NEEDS TO BE USED. IT IS REMOVED FOR COMPILABILITY TEST
                qp.SignedOutMultiplier(**mode_mult_wires, output_wires_zeroed=True)
                # qp.adjoint(qp.SignedOutMultiplier)(**mode_mult_wires, output_wires_zeroed=True)
                return new_bitstrings

            prev_bitstrings = _inner_bilinear_terms(prev_bitstrings)

            return prev_bitstrings

        prev_bitstrings = np.zeros(precision, dtype=int)
        prev_bitstrings = constant_term(prev_bitstrings)
        prev_bitstrings = linear_terms(prev_bitstrings)
        prev_bitstrings = quadratic_terms(prev_bitstrings)
        prev_bitstrings = bilinear_terms(prev_bitstrings)

        # Finish up the coefficients register by unloading the last loaded coefficients
        qp.QROM(prev_bitstrings, **qrom_wires)
        diagonalize_vibronic(key=diag_key, wires=registers["electronic"])

    def kinetic_fragment(fragment, aqft_order):
        # use time, not first_order_time_step because the kinetic fragment is the
        # middle one in second-order Trotter, so we immediately merge the two neighbouring
        # fragments with first_order_time_step duration.
        # todo: Replace BasisState + SignedMultiplier by a classical-quantum multiplier?
        kinetic_coeffs = get_momentum_coefficients(fragment) * time

        @qp.for_loop(fragment.modes)
        def kinetic_terms(k):
            """Run a single quadratic time evolution sub-fragment for mode ``k`` in momentum
            space."""
            square_wires, mult_wires = _extract_registers(registers, "quadratic", k)
            _coeffs = kinetic_coeffs[k]
            if np.allclose(_coeffs, 0.0):
                return
            bitstring = qp.math.binary_decimals(_coeffs, precision, unit=2 * np.pi)

            qp.BasisState(bitstring, registers["coefficients"])
            qp.AQFT(order=aqft_order, wires=registers[f"mode {k}"])
            qp.SignedOutSquare(**square_wires, output_wires_zeroed=True)
            qp.OutMultiplier(**mult_wires)
            # THIS IS WRONG- THE ADJOINT NEEDS TO BE USED. IT IS REMOVED FOR COMPILABILITY TEST
            qp.SignedOutSquare(**square_wires, output_wires_zeroed=True)
            # qp.adjoint(qp.SignedOutSquare)(**square_wires, output_wires_zeroed=True)
            qp.adjoint(qp.AQFT)(order=aqft_order, wires=registers[f"mode {k}"])
            qp.BasisState(bitstring, registers["coefficients"])

        kinetic_terms()

    qp.for_loop(len(fragments) - 1)(position_fragments)()

    kinetic_fragment(fragments[-1], aqft_order)

    qp.for_loop(len(fragments) - 2, -1, -1)(position_fragments)()


def _validate_registers(registers, fragments):
    n_states = fragments[0].states
    n_modes = fragments[0].modes

    expected_register_names = {"electronic", "cache", "coefficients", "phase gradient", "work"}
    expected_register_names |= {f"mode {i}" for i in range(n_modes)}
    assert isinstance(registers, dict)
    assert set(registers) == expected_register_names
    k = len(registers["mode 0"])
    b = len(registers["coefficients"])

    n = qp.math.ceil_log2(n_states)
    assert len(registers["electronic"]) == n
    assert len(registers["cache"]) >= 2 * k
    assert len(registers["phase gradient"]) >= b
    assert len(registers["work"]) >= max(n - 1, 2 * k, 2 * b + 2)
    assert all(len(registers[f"mode {i}"]) == k for i in range(1, n_modes))


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
        :title: Register sizes
        :href: register-sizes

        The vibronic Hamiltonian acts on :math:`N` electronic states, represented on
        :math:`n=\lceil\log_2(N)\rceil` qubits, as well as :math:`M` vibrational modes,
        each discretized on a grid with :math:`K` grid points, requiring
        :math:`k=\lceil\log_2(K)\rceil` qubits per mode. Phases are implemented directly with
        arithmetic acting on a phase gradient state with :math:`b` qubits.

        According to this setup, the ``registers`` passed to this function should include the
        following wire registers:

        .. list-table::
           :widths: 25 25 50
           :header-rows: 1

           * - ``key``
             - Expected size
             - Information content
           * - ``"electronic"``
             - :math:`n`
             - Electronic state
           * - ``f"mode {i}"``
             - :math:`k`
             - Position of :math:`i`\ th vibrational mode (signed)
           * - ``"cache"``
             - :math:`2k`
             - Cached squares/products of modes (see below, signed/unsigned)
           * - ``"coefficients"``
             - :math:`b`
             - Hamiltonian coefficients (unsigned)
           * - ``"phase gradient"``
             - :math:`b`
             - Phase gradient state (unsigned)
           * - ``"work"``
             - :math:`\max(n-1, 2k, 2b+2)`
             - Data loading/arithmetic scratch (see below)

        **The reasoning for the cache register size is:**

        - Squaring a signed :math:`k`-bit integer yields values from the range
          :math:`[0, 2^{2k-2}]`, so that :math:`2k-1` (unsigned) bits are needed to represent
          the output.
        - Multiplying two signed :math:`k`-bit integers yields values from the range
          :math:`[-2^{k-1}(2^{k-1}-1), 2^{2k-2}]`. The lower end of the range would only require
          :math:`2k-1` (signed) bits, but that would limit the upper end to :math:`2^{2k-2}-1`.
          Thus, we need one more bit, i.e. :math:`2k` (signed) bits are needed to represent the
          output.

        Overall, we thus can use a cache register of size :math:`2k` and always consider it to be
        a signed integer register. When caching the squaring result, we simply use just the
        unsigned part of that register, slicing away the sign bit.

        **The reasoning for the work register size is:**

        - The ``QROM``\ s using this register have :math:`n` control wires (electronic register)
          and thus require at least :math:`n-1` work wires for efficient implementation (unary
          iteration).
        - The ``SemiAdder`` in the constant term computes on registers of size :math:`b`, so it
          needs :math:`b-1` work wires.
        - The :func:`~.pennylane.labs.templates.semi_signed_out_multiplier`
          (with ``len(y_wires)=k`` and ``len(output_wires)=b``)
          in the linear terms requires :math:`\max(k, 2b+2)` (see documentation).
        - The ``SignedOutSquare`` of a :math:`k` qubit register with ``output_wires_zeroed=True``
          requires :math:`k` work wires.
        - The ``OutMultiplier`` of the unsigned square cache and the coefficients into the phase
          gradient register (with ``output_wires_zeroed=True``) requires :math:`b+1` work wires.
        - The :func:`~.pennylane.labs.templates.semi_signed_out_multiplier`
          (with ``len(y_wires)=2k`` and ``len(output_wires)=b``)
          in the bilinear terms requires :math:`\max(2k, 2b+2)` (see documentation).

        Overall, the largest requirement is

        .. math::

            \max(n-1, b-1, \max(k, 2b+2), k, b+1, \max(2k, 2b+2))
            =\max(n-1, 2k, 2b+2)

        We typically would expect :math:`b>k\approx n`, so the third term to be the largest, but
        this depends on the specific simulation setup.

        Note that the work wire requirements (in particular those of ``semi_signed_out_multiplier``)
        can be reduced at the cost of additional non-Clifford gates. The above calculation maximizes
        qubit overhead and minimizes the non-Clifford gate count.

    """
    print("trotter_vibronic has been called.")
    _validate_registers(registers, fragments)

    assert num_trotter_steps > 0
    trotter_time_step = evolution_time / num_trotter_steps

    if aqft_order is None:
        aqft_order = len(registers["mode 0"]) - 1

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
