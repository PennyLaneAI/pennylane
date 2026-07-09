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

from .half_signed_out_multiplier import half_signed_out_multiplier


def fragment_to_dense(fragment: RealspaceMatrix, op_type: tuple[str]):
    """Test helper function that converts the coefficients for a specific operator type
    (e.g. ``("P", "P")`` or ``("Q",)``) of a vibronic fragment into a dense matrix. The output
    shape depends on the order of the operator type, i.e., the length of ``op_type``.

    Args:
        fragment (RealspaceMatrix): vibronic fragment from which to extract the coefficients.
        op_type (tuple[str]): operator type for which to extract the coefficients.

    Returns:
        np.ndarray: dense coefficients tensor for the specific operator type. Has shape
        ``(n_states, n_states) + (n_modes,) * len(op_type)``.

    """
    n_states = fragment.states
    n_modes = fragment.modes
    order = len(op_type)
    # Initialize dense coefficients tensor
    dense = np.zeros((n_states, n_states) + (n_modes,) * order)

    # Iterate over all electronic state pairs and corresponding coupling terms in the fragment
    for elec_key, val in fragment.get_coefficients().items():
        # Extract the values for the requested operator type
        terms = val.get(op_type, None)
        if terms is None:
            continue
        if order == 0:
            # For order=0, there is just a single constant term, no mode dependency
            dense[elec_key] = terms.get((), 0.0)
            continue
        # Iterate over all combinations of modes with the given order
        for modes in combinations_with_replacement(range(n_modes), r=order):
            dense[elec_key][modes] = terms.get(modes, 0.0)
    return dense


def get_position_coefficients(fragment: RealspaceMatrix):
    """Get the coefficients for a given position fragment.

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
    M = qp.matrix(diagonalize_vibronic_mat, wires)(key=diag_key, wires=wires)[:n_states, :n_states]
    constant = M.T @ fragment_to_dense(fragment, ()) @ M  # shape = (n_states, n_states)
    constant_diag = np.diag(constant)  # shape = (n_states,)

    # linear will have shape = (n_modes, n_states, n_states)
    linear = np.einsum("ba,bcz,cd->zad", M, fragment_to_dense(fragment, ("Q",)), M)
    linear_diag = np.diagonal(linear, axis1=1, axis2=2)  # shape = (n_modes, n_states)

    # sec_order will have shape = (n_modes, n_modes, n_states, n_states)
    sec_order = np.einsum("ba,bcyz,cd->yzad", M, fragment_to_dense(fragment, ("Q", "Q")), M)
    sec_order_diag = np.diagonal(
        sec_order, axis1=2, axis2=3
    )  # shape = (n_modes, n_modes, n_states)

    quadratic = np.diagonal(sec_order_diag).copy().T  # shape = (n_modes, n_states)
    bilinear = sec_order_diag.copy()  # shape = (n_modes, n_modes, n_states)
    bilinear[np.tril_indices(n_modes)] = 0.0  # only upper triangle w.r.t. axes=(0, 1) populated
    return constant_diag, linear_diag, quadratic, bilinear


def get_momentum_coefficients(fragment: RealspaceMatrix):
    """Get the coefficients for a given momentum fragment.
    Also validates that the terms in the fragment are limited to expected terms.

    Args:
        fragment (RealspaceMatrix): fragment of which to read the coefficients

    Returns:
        np.ndarray: Array of momentum coefficients with shape ``(fragment.modes,)``

    """
    coeffs = fragment_to_dense(
        fragment, ("P", "P")
    )  #  shape=(n_states, n_states, n_modes, n_modes)
    coeffs_diag = np.diagonal(coeffs, axis1=2, axis2=3)  # shape = (n_states, n_states, n_modes)
    coeffs_diag2 = np.diagonal(
        coeffs_diag
    )  # shape = (n_modes, n_states) (np.diagonal: new ax last)
    coeffs_final = coeffs_diag2[:, 0]  # shape = (n_modes,)
    return coeffs_final


def diagonalize_vibronic_mat(*, key: tuple[int], wires: WiresLike):
    r"""Diagonalize a vibronic fragment by applying Clifford operations.
    Based on Fig.2 of `Motlagh et al, arXiv:2411.13669 <https://arxiv.org/abs/2411.13669>`__.

    Args:
        key (tuple[int]): Row and column index of the only non-zero matrix element in the first row
            that contains a non-zero matrix element at all.
        wires (WiresLike): electronic wires on which the fragment acts. These wires are the only
            ones to which we need to apply operations for the diagonalization.

    This function is intended for the computation of the diagonalization *matrix* of a fragment.

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

    key_bitstrings = [qp.math.int_to_binary(k, len(wires)) for k in key]
    diagonalization_key = (key_bitstrings[0] + key_bitstrings[1]) % 2
    support = np.where(diagonalization_key)[0][::-1]
    control = support[0]
    qp.H(wires=wires[control])
    _ = [qp.CNOT(wires=[wires[control], wires[k]]) for k in support[1:]]


def diagonalize_vibronic_qjit(*, key, wires):
    r"""Diagonalize a vibronic fragment by applying Clifford operations.
    Based on Fig.2 of `Motlagh et al, arXiv:2411.13669 <https://arxiv.org/abs/2411.13669>`__.

    This is the qjit-compatible variant of ``diagonalize_vibronic``.
    ``key[0]`` and ``key[1]`` may be Python ints (compile-time)
    or traced integer scalars (runtime). The circuit structure -- whether any
    gates are applied, which wire is the control, and which targets receive a
    CNOT -- is expressed with ``catalyst.cond`` / ``catalyst.for_loop`` so it can
    depend on traced values.

    Args:
        key (tuple[int]): Row and column index of the only non-zero matrix
            element in the first row that contains a non-zero element at all.
        wires (WiresLike): electronic wires on which the fragment acts.

    This function is intended for the computation of the diagonalization *circuit* of a fragment.

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
    n = len(wires)

    # Wire w holds bit (n-1-w) (MSB-first), matching qp.math.int_to_binary.
    # diff[w] is the XOR of the wire-w bit of k0 and k1 -- this is exactly
    # (int_to_binary(k0) + int_to_binary(k1)) % 2 from the matrix function.
    shifts = np.arange(n - 1, -1, -1, dtype=np.int64)
    diff = ((key[0] >> shifts) & 1) ^ ((key[1] >> shifts) & 1)

    any_diff = np.sum(diff) > 0
    # qjit-compatible version of support = where(diff)[::-1]; control = support[0],
    # i.e. control_pos is the largest wire index whose diff bit is set.
    control_pos = qp.math.max(qp.math.where(diff > 0, qp.math.arange(n), -1))
    if qp.compiler.active():
        any_diff = qp.math.array(any_diff, like="jax")
        diff = qp.math.array(diff, like="jax")
        wires = qp.math.array(wires, like="jax")

    control_wire = wires[control_pos]

    @qp.cond(any_diff)  # key[0] != key[1]  <=>  some bit differs
    def apply():
        qp.H(wires=[control_wire])

        # This loop goes over the full range in order to not have a dynamic range that depends
        # on control_pos

        @qp.for_loop(n)
        def loop(w):

            # If the diff bit is set and the wire is not the control wire
            @qp.cond((diff[w] > 0) & (w != control_pos))
            def maybe_cnot():
                qp.CNOT(wires=[control_wire, wires[w]])

            maybe_cnot()

        loop()  # pylint: disable=no-value-for-parameter

    apply()


def load_coefficients(
    coefficients: np.ndarray, precision: int, prev_bitstrings: np.ndarray, qrom_wires: dict
):
    """Extract bit strings for one-dimensional coefficients array, XOR them with
    ``prev_bitstrings``, and load the result with a QROM, using the registers in ``qrom_wires``.

    Args:
        coefficients (np.ndarray): Coefficients to load. Should be one-dimensional.
        precision (int): Bit precision with which to load the coefficients.
        prev_bitstrings (np.ndarray): Bit string array representing the currently loaded reference
            point in the data loading register. Should have shape ``(len(coefficients), precision)``
        qrom_wires (dict[WiresLike]): Wire registers on which the data loading :class:`~.QROM`
            should act

    Returns:
        np.ndarray: Bitstrings corresponding to ``coefficients``. These bitstrings represent the
        newly loaded reference point in the data loading register, so that they can be used in the
        subsequent data loading step.

    """
    new_bitstrings = qp.math.binary_decimals(coefficients, precision, unit=2 * np.pi)
    change_bitstrings = (prev_bitstrings + new_bitstrings) % 2
    qp.QROM(change_bitstrings, **qrom_wires)
    return new_bitstrings


def _extract_registers(registers, mode_registers, term, *mode_ids):
    r"""Extract registers for a specific term of the vibronic Trotter time evolution.

    Args:
        registers (dict): A dictionary of wire registers from which the needed registers for a
            given term are extracted.
        mode_registers (np.ndarray): Array of wire registers for vibrational modes from which to
            extract the registers for a given term.
        term (str): Which term to extract the registers for. Must be one of
            - ``"constant"`` for the adder of the constant term
            - ``"linear"`` for the multiplier of the linear term
            - ``"quadratic"`` for the squaring and the multiplier of the quadratic term
            - ``"bilinear"`` for the two multipliers of the bilinear term
            - ``"QROM"`` for the data loading :class:`~.QROM` used across all terms.
        mode_ids (tuple[int]): Index/indices for the mode(s) involved in the term. Should be a
            sequence of (0 , 1 , 2) integers for ``term`` being (``"constant"``/``"QROM"`` ,
            ``"linear"``/``"quadratic"``, ``"bilinear"``).

    Returns:
        dict or tuple[dict]: One or multiple sets of registers required for the given ``term``,
        extracted from ``registers`` and ``mode_registers``. If ``term`` is one of ``"constant"``,
        ``"linear"``, or ``"QROM"``, a single ``dict`` is returned. For ``"quadratic"`` and
        ``"bilinear"``, two ``dict``\ s are returned.

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
        # The signed register for half_signed_out_multiplier must be the _second_ input
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
        # The signed register for half_signed_out_multiplier must be the _second_ input
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


def _trotter_step_second_order(idx, time, fragments, registers, mode_registers, aqft_order):
    r"""Second-order Trotter time evolution step implemented via custom arithmetic circuits
    based on a phase gradient resource state.

    Args:
        idx (int): Trotter step index. This argument is not used explicitly.
        time (float): Second-order Trotter time step size.
        fragments (list[RealspaceMatrix]): Trotter fragments of the vibronic Hamiltonian.
            It is assumed that the kinetic fragment is in the last position.
        registers (dict[WiresLike]): Wire registers. See main function for details.
        aqft_order (int): Approximation order used in :class:`~.AQFT`, which is used to transform
            between position and momentum space. If set to ``None``, no approximation will be made.

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
    # pylint: disable=no-value-for-parameter, unused-argument, too-many-statements, too-many-arguments

    precision = len(registers["phase gradient"])
    first_order_time_step = time / 2
    qrom_wires = _extract_registers(registers, mode_registers, "QROM")
    diag_keys = [
        next(iter(k for k, v in frag.get_coefficients().items() if v)) for frag in fragments
    ]
    n_modes = fragments[0].modes
    n_states = fragments[0].states
    all_constant = []
    all_linear = []
    all_quadratic = []
    all_bilinear = []

    # dynamic loops
    for fragment in fragments[:-1]:
        _coeffs = [c * first_order_time_step for c in get_position_coefficients(fragment)]
        all_constant.append(_coeffs[0])
        all_linear.append(_coeffs[1])
        all_quadratic.append(_coeffs[2])
        all_bilinear.append(_coeffs[3])

    bilinear_indices = np.array(np.triu_indices(n_modes, 1))
    all_bilinear = [x[*bilinear_indices] for x in all_bilinear]
    if qp.compiler.active():
        all_constant = qp.math.array(all_constant, like="jax")
        all_linear = qp.math.array(all_linear, like="jax")
        all_quadratic = qp.math.array(all_quadratic, like="jax")
        # Reshape the bilinear data into a flattened structure with respect to mode pairs
        # Note that k < ell matches the data structure of bilinear_coeffs, which is only
        # populated in the upper triangular part, w.r.t. the mode axes.
        bilinear_indices = qp.math.array(bilinear_indices, like="jax")
        all_bilinear = qp.math.array(all_bilinear, like="jax")
        diag_keys = qp.math.array(diag_keys, like="jax")

    def position_fragments(i):
        diag_key = diag_keys[i]
        const_coeffs = all_constant[i]
        linear_coeffs = all_linear[i]
        quadratic_coeffs = all_quadratic[i]
        bilinear_coeffs = all_bilinear[i]

        qp.adjoint(diagonalize_vibronic_qjit, lazy=False)(
            key=diag_key, wires=registers["electronic"]
        )

        def constant_term(prev_bitstrings):
            def skip_fn():
                return prev_bitstrings

            def actual_fn():
                new_bitstrings = load_coefficients(
                    const_coeffs, precision, prev_bitstrings, qrom_wires
                )
                qp.SemiAdder(**_extract_registers(registers, mode_registers, "constant"))
                return new_bitstrings

            return qp.cond(qp.math.allclose(const_coeffs, 0.0), skip_fn, actual_fn)()

        @qp.for_loop(n_modes)
        def linear_terms(k, prev_bitstrings):
            """Run a single linear time evolution sub-fragment for mode ``k``.
            The currently encoded bitstrings on the coefficients register are provided in
            ``prev_bitstrings``."""
            _coeffs = linear_coeffs[k]

            def skip_fn():
                return prev_bitstrings

            def actual_fn():
                new_bitstrings = load_coefficients(_coeffs, precision, prev_bitstrings, qrom_wires)
                half_signed_out_multiplier(
                    **_extract_registers(registers, mode_registers, "linear", k)
                )
                return new_bitstrings

            return qp.cond(qp.math.allclose(_coeffs, 0.0), skip_fn, actual_fn)()

        @qp.for_loop(n_modes)
        def quadratic_terms(k, prev_bitstrings):
            """Run a single quadratic time evolution sub-fragment for mode ``k``.
            The currently encoded bitstrings on the coefficients register are provided in
            ``prev_bitstrings``."""
            _coeffs = quadratic_coeffs[k]

            def skip_fn():
                return prev_bitstrings

            def actual_fn():
                new_bitstrings = load_coefficients(_coeffs, precision, prev_bitstrings, qrom_wires)
                square_wires, mult_wires = _extract_registers(
                    registers, mode_registers, "quadratic", k
                )
                qp.SignedOutSquare(**square_wires, output_wires_zeroed=True)
                qp.OutMultiplier(**mult_wires)
                qp.adjoint(qp.SignedOutSquare(**square_wires, output_wires_zeroed=True))
                return new_bitstrings

            return qp.cond(qp.math.allclose(_coeffs, 0.0), skip_fn, actual_fn)()

        @qp.for_loop(len(bilinear_indices))
        def bilinear_terms(k, prev_bitstrings):
            """Run a single bilinear time evolution sub-fragment for mode pair ``bilinear_indices[k]``.
            The currently encoded bitstrings on the coefficients register are provided in
            ``prev_bitstrings``."""

            _coeffs = bilinear_coeffs[k]
            ids = bilinear_indices[:, k]

            def skip_fn():
                return prev_bitstrings

            def actual_fn():
                mode_mult_wires, coeff_mult_wires = _extract_registers(
                    registers, mode_registers, "bilinear", *ids
                )
                new_bitstrings = load_coefficients(_coeffs, precision, prev_bitstrings, qrom_wires)
                qp.SignedOutMultiplier(**mode_mult_wires, output_wires_zeroed=True)
                half_signed_out_multiplier(**coeff_mult_wires)
                qp.adjoint(qp.SignedOutMultiplier(**mode_mult_wires, output_wires_zeroed=True))
                return new_bitstrings

            return qp.cond(qp.math.allclose(_coeffs, 0.0), skip_fn, actual_fn)()

        prev_bitstrings = np.zeros((n_states, precision), dtype=int)
        prev_bitstrings = constant_term(prev_bitstrings)
        prev_bitstrings = linear_terms(prev_bitstrings)
        prev_bitstrings = quadratic_terms(prev_bitstrings)
        prev_bitstrings = bilinear_terms(prev_bitstrings)

        # Finish up the coefficients register by unloading the last loaded coefficients
        qp.QROM(prev_bitstrings, **qrom_wires)
        diagonalize_vibronic_qjit(key=diag_key, wires=registers["electronic"])

    def kinetic_fragment(fragment, aqft_order):
        # use ``time``, not ``first_order_time_step`` because the kinetic fragment is the
        # middle one in second-order Trotter, so we immediately merge the two neighbouring
        # fragments, each with duration ``first_order_time_step``.
        kinetic_coeffs = get_momentum_coefficients(fragment) * time
        if qp.compiler.active():
            kinetic_coeffs = qp.math.array(kinetic_coeffs, like="jax")

        @qp.for_loop(n_modes)
        def kinetic_terms(k):
            """Run a single quadratic time evolution sub-fragment for mode ``k`` in momentum
            space."""
            _coeffs = kinetic_coeffs[k]

            def skip_fn():
                """Do nothing."""

            def actual_fn():
                square_wires, mult_wires = _extract_registers(
                    registers, mode_registers, "quadratic", k
                )
                bitstring = qp.math.binary_decimals(_coeffs, precision, unit=2 * np.pi)
                qp.BasisState(bitstring, registers["coefficients"])
                qp.AQFT(order=aqft_order, wires=mode_registers[k])
                qp.SignedOutSquare(**square_wires, output_wires_zeroed=True)
                qp.OutMultiplier(**mult_wires)
                qp.adjoint(qp.SignedOutSquare(**square_wires, output_wires_zeroed=True))
                qp.adjoint(qp.AQFT)(order=aqft_order, wires=mode_registers[k])
                qp.BasisState(bitstring, registers["coefficients"])

            qp.cond(qp.math.allclose(_coeffs, 0.0), skip_fn, actual_fn)()

        kinetic_terms()

    qp.for_loop(len(fragments) - 1)(position_fragments)()

    kinetic_fragment(fragments[-1], aqft_order)

    qp.for_loop(len(fragments) - 2, -1, -1)(position_fragments)()


def _validate_registers(registers, fragments):
    """Validate wire register sizes for vibronic algorithm. See documentation of
    ``trotter_vibronic`` for details on the expected sizes."""
    n_states = fragments[0].states
    n_modes = fragments[0].modes

    expected_register_names = {"electronic", "cache", "coefficients", "phase gradient", "work"}
    expected_register_names |= {f"mode {i}" for i in range(n_modes)}
    assert isinstance(registers, dict)
    assert set(registers) == expected_register_names

    b = len(registers["coefficients"])
    k = len(registers["mode 0"])
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
            between position and momentum space. If set to ``None``, no approximation will be made.

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
        - The :func:`~.pennylane.labs.templates.half_signed_out_multiplier`
          (with ``len(y_wires)=k`` and ``len(output_wires)=b``)
          in the linear terms requires :math:`\max(k, 2b+2)` (see documentation).
        - The ``SignedOutSquare`` of a :math:`k` qubit register with ``output_wires_zeroed=True``
          into the cache register of size :math:`2k-1` requires :math:`2k-2` work wires for the
          decomposition with minimized non-Clifford cost.
        - The ``OutMultiplier`` of the unsigned square cache and the coefficients into the phase
          gradient register (with ``output_wires_zeroed=True``) requires :math:`b+1` work wires.
        - The :func:`~.pennylane.labs.templates.half_signed_out_multiplier`
          (with ``len(y_wires)=2k`` and ``len(output_wires)=b``)
          in the bilinear terms requires :math:`\max(2k, 2b+2)` (see documentation).

        Overall, the largest requirement is

        .. math::

            \max(n-1, b-1, \max(k, 2b+2), 2k-2, b+1, \max(2k, 2b+2))
            =\max(n-1, 2k, 2b+2)

        We typically would expect :math:`b>k\approx n`, so the third term to be the largest, but
        this depends on the specific simulation setup.

        Note that the work wire requirements (in particular those of ``half_signed_out_multiplier``)
        can be reduced at the cost of additional non-Clifford gates. The above calculation maximizes
        qubit overhead and minimizes the non-Clifford gate count.

    """
    _validate_registers(registers, fragments)

    assert num_trotter_steps > 0
    trotter_time_step = evolution_time / num_trotter_steps

    n_modes = fragments[0].modes
    mode_registers = qp.math.array([registers.pop(f"mode {i}") for i in range(n_modes)], like="jax")

    if aqft_order is None:
        aqft_order = mode_registers.shape[1] - 1

    @qp.for_loop(num_trotter_steps)
    def trotter_steps(step_idx):
        _trotter_step_second_order(
            step_idx,
            time=trotter_time_step,
            fragments=fragments,
            registers=registers,
            mode_registers=mode_registers,
            aqft_order=aqft_order,
        )

    trotter_steps()  # pylint: disable=no-value-for-parameter
