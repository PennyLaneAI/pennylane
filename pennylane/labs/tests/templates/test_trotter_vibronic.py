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
"""Contains tests for the Trotter template for vibronic Hamiltonians."""

import numpy as np
import pytest

import pennylane as qp
from pennylane.labs.templates.trotter_vibronic import (
    _extract_registers,
    _preprocess_data,
    _validate_fragments,
    _validate_registers,
    diagonalize_vibronic_mat,
    diagonalize_vibronic_qjit,
    fragment_to_dense,
    get_momentum_coefficients,
    get_position_coefficients,
    load_coefficients,
    trotter_vibronic,
)
from pennylane.labs.trotter_error.fragments import vibronic_fragments
from pennylane.labs.trotter_error.realspace import (
    RealspaceCoeffs,
    RealspaceMatrix,
    RealspaceOperator,
    RealspaceSum,
)


class TestDiagonalizeVibronicMat:
    """Tests for ``diagonalize_vibronic_mat`` and ``diagonalize_vibronic_qjit``."""

    @pytest.mark.parametrize(
        "n_states, elec_key, expected_support",
        [
            (2, (0, 0), []),
            (2, (0, 1), [0]),
            (3, (0, 0), []),
            (3, (0, 1), [1]),
            (3, (0, 2), [0]),
            (4, (0, 0), []),
            (4, (0, 1), [1]),
            (4, (0, 2), [0]),
            (4, (0, 3), [1, 0]),
            (5, (0, 4), [0]),
            (6, (0, 1), [2]),
            (7, (0, 2), [1]),
            (8, (0, 3), [2, 1]),
            (8, (0, 7), [2, 1, 0]),
            (128, (0, 62), [5, 4, 3, 2, 1]),
            (128, (0, 127), [6, 5, 4, 3, 2, 1, 0]),
        ],
    )
    @pytest.mark.parametrize("fn", [diagonalize_vibronic_mat, diagonalize_vibronic_qjit])
    def test_expected_circuit(self, n_states, elec_key, expected_support, fn, seed):
        """Test that the diagonalization circuit looks as expected for some small-scale examples."""
        # pylint: disable=too-many-arguments
        n_wires = qp.math.ceil_log2(n_states)
        wires = list(range(n_wires))
        rng = np.random.default_rng(seed)
        rng.shuffle(wires)
        with qp.queuing.AnnotatedQueue() as q:
            fn(key=elec_key, wires=wires)

        if expected_support:
            c = wires[expected_support[0]]
            expected_ops = [qp.Hadamard(c)]
            expected_ops += [qp.CNOT([c, wires[idx]]) for idx in expected_support[1:]]
        else:
            expected_ops = []
        assert q.queue == expected_ops, f"{q.queue}\n{expected_ops}"

    @pytest.mark.parametrize(
        "n_states, col", [(3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (4, 3), (17, 9), (34, 19)]
    )
    @pytest.mark.parametrize("row", (0, 1, 2))
    def test_diagonalizes_correctly_mat(self, n_states, col, row, seed):
        """Test that the diagonalization works as expected."""
        n_wires = qp.math.ceil_log2(n_states)
        wires = list(range(n_wires))
        rng = np.random.default_rng(seed)
        rng.shuffle(wires)
        m = row ^ col
        matrix = np.zeros((n_states, n_states))
        for i, val in enumerate(rng.random(n_states)):
            if i < i ^ m < n_states:
                matrix[i, i ^ m] = matrix[i ^ m, i] = val
        key = (row, col)
        diag_mat = qp.matrix(diagonalize_vibronic_mat, wires)(key=key, wires=wires)
        # Just make sure it is an orthogonal matrix
        assert np.all(np.isreal(diag_mat)) and np.allclose(
            diag_mat @ diag_mat.T, np.eye(2**n_wires)
        )
        diag_mat = diag_mat[:n_states, :n_states]

        diagonalized = diag_mat.T @ matrix @ diag_mat
        # Test that the diagonalization worked
        assert np.allclose(np.diag(np.diag(diagonalized)), diagonalized)

    @pytest.mark.parametrize(
        "n_states, elec_key, expected_support",
        [
            (2, (0, 0), []),
            (2, (0, 1), [0]),
            (3, (0, 0), []),
            (3, (0, 1), [1]),
            (3, (0, 2), [0]),
            (4, (0, 0), []),
            (4, (0, 1), [1]),
            (4, (0, 2), [0]),
            (4, (0, 3), [1, 0]),
            (5, (0, 4), [0]),
            (6, (0, 1), [2]),
            (7, (0, 2), [1]),
            (8, (0, 3), [2, 1]),
            (8, (0, 7), [2, 1, 0]),
            (128, (0, 62), [5, 4, 3, 2, 1]),
            (128, (0, 127), [6, 5, 4, 3, 2, 1, 0]),
        ],
    )
    def test_qjit_compatibility(self, n_states, elec_key, expected_support, seed):
        """Test that the diagonalization circuit matrix looks the same when using `qjit` or not."""

        n_wires = qp.math.ceil_log2(n_states)
        wires = list(range(n_wires))
        rng = np.random.default_rng(seed)
        rng.shuffle(wires)
        state = rng.random(size=2**n_wires)
        state /= np.linalg.norm(state)

        @qp.qnode(qp.device("lightning.qubit"))
        def test_fn(key):
            # Use wires in order for state comparability
            qp.StatePrep(state, list(range(n_wires)))
            # Run expected circuit forward
            if expected_support:
                c = wires[expected_support[0]]
                qp.Hadamard(c)
                _ = [qp.CNOT([c, wires[idx]]) for idx in expected_support[1:]]
            # Run qjit-compatible function backward
            qp.adjoint(diagonalize_vibronic_qjit, lazy=False)(key=key, wires=wires)
            return qp.state()

        non_qjit_result = test_fn(elec_key)
        qjit_result = qp.qjit(test_fn)(elec_key)
        assert np.allclose(non_qjit_result, state)
        assert np.allclose(qjit_result, state)


def _random_vibronic_elec_ids(n_states, rng):
    m = rng.integers(0, n_states)
    keys = [(i, int(i ^ m)) for i in range(n_states) if i ^ m < n_states]
    return keys


def random_vibronic_fragment(n_states, n_modes, include_op_types=None, seed=None):
    """Construct a random vibronic fragment.

    Args:
        n_states (int): Number of electronic states
        n_modes (int): Number of modes
        include_op_types (list[tuple[str]]): List of operator types to include
        seed (int): Randomness seed used for numerical data and basis choice of the fragment

    Returns:
        RealspaceMatrix: The random vibronic fragment

    """
    if seed is None:
        seed = np.random.randint(251126)
    rng = np.random.default_rng(seed)

    if include_op_types is None:
        include_op_types = [(), ("Q",), ("Q", "Q")]

    # Can't mix kinetic with potential terms
    kin_op = ("P", "P")
    if kin_op in include_op_types:
        assert include_op_types == [kin_op]
        tensor = np.diag(rng.random(n_modes))
        op = RealspaceOperator(n_modes, kin_op, RealspaceCoeffs(tensor, "label"))
        blocks = {(i, i): RealspaceSum(n_modes, [op]) for i in range(n_states)}
        return RealspaceMatrix(n_states, n_modes, blocks)

    elec_ids = _random_vibronic_elec_ids(n_states, rng)
    blocks = {}
    # Iterate over pairs of electronic states
    for elec_idx in elec_ids:
        ops = []
        # Iterate over operator types such as (), ("Q",) and ("Q", "Q")
        for op_type in include_op_types:
            degree = len(op_type)
            tensor = rng.random((n_modes,) * degree)
            if degree == 2:
                # If the term is of degree two, our convention is that only the upper triangle
                # of the coefficient matrix (incl the diagonal) is populated.
                tensor[np.tril_indices(n_modes, k=-1)] = 0.0
            ops.append(RealspaceOperator(n_modes, op_type, RealspaceCoeffs(tensor, "label")))
        # The coefficients are always symmetric with respect to the electronic state indices,
        # because Hamiltonians are Hermitian, and vibronic Hamiltonians are real-valued
        blocks[elec_idx] = blocks[elec_idx[::-1]] = RealspaceSum(n_modes, ops)

    return RealspaceMatrix(n_states, n_modes, blocks)


def _vibronic_fragment_list(n_states=2, n_modes=2, seed=42):
    """Build a valid position + kinetic fragment list via ``vibronic_fragments``."""
    rng = np.random.default_rng(seed)
    freqs = rng.random(n_modes)
    taylor_coeffs = [
        rng.random((n_states, n_states)),
        rng.random((n_states, n_states, n_modes)),
    ]
    return vibronic_fragments(n_states, n_modes, freqs, taylor_coeffs)


def _make_registers(n_states, n_modes, k=2, b=4, wire_offset=0):
    """Build a register dict that satisfies ``_validate_registers_and_fragments``."""
    n = qp.math.ceil_log2(n_states)
    needed_work = max(n - 1, 2 * k, 2 * b + 2)
    sizes = {
        "_": wire_offset,
        "electronic": n,
        "__": 4,
        "cache": 2 * k,
        "coefficients": b,
        "___": 3,
        "phase gradient": b + 1,
        "work": needed_work,
    }
    sizes |= {f"mode {i}": k for i in range(n_modes)}
    registers = qp.registers(sizes)
    registers.pop("_")
    registers.pop("__")
    registers.pop("___")
    return registers


class TestFragmentReadout:
    """Tests for helper functions that extract information from fragments."""

    @pytest.mark.parametrize("n_states, n_modes", [(3, 2), (5, 10), (14, 2), (19, 7)])
    @pytest.mark.parametrize("op_type", [(), ("Q",), ("Q", "Q"), ("P", "P")])
    def test_fragment_to_dense_roundtrip(self, n_states, n_modes, op_type, seed):
        """Test that extracting coefficients with ``fragment_to_dense`` and converting them
        back to a nested dictionary structure yields the identity mapping."""
        fragment = random_vibronic_fragment(n_states, n_modes, [op_type], seed)
        dense_coeffs = fragment_to_dense(fragment, op_type)
        degree = len(op_type)
        assert isinstance(dense_coeffs, np.ndarray)
        assert dense_coeffs.shape == (n_states, n_states) + (n_modes,) * degree
        assert np.allclose(np.moveaxis(dense_coeffs, 1, 0), dense_coeffs)

        rng = np.random.default_rng(seed)
        if op_type == ("P", "P"):
            # Kinetic term must be diagonal w.r.t. electronic d.o.f.s
            expected_ids = [(i, i) for i in range(n_states)]
        else:
            expected_ids = _random_vibronic_elec_ids(n_states, rng)
        where = np.abs(dense_coeffs) > 1e-12
        for _ in range(degree):
            where = np.any(where, axis=-1)
        ids = list(zip(*np.where(where)))
        assert set(ids) == set(expected_ids)

        # Reconstruct the fragment from the dense matrix
        blocks = {}
        for idx in ids:
            idx = (int(idx[0]), int(idx[1]))
            op = RealspaceOperator(n_modes, op_type, RealspaceCoeffs(dense_coeffs[idx], "label"))
            blocks[idx] = blocks[idx[::-1]] = RealspaceSum(n_modes, [op])
        reconstructed_fragment = RealspaceMatrix(n_states, n_modes, blocks)

        # Compare to the original fragment
        assert reconstructed_fragment == fragment

    @pytest.mark.parametrize("n_states, n_modes", [(3, 2), (5, 10), (14, 2), (19, 7)])
    def test_get_position_coefficients(self, n_states, n_modes, seed):
        """Test that ``get_position_coefficients`` returns the correct terms."""
        fragment = random_vibronic_fragment(n_states, n_modes, seed=seed)

        # Obtain diagonalization matrix
        n_wires = qp.math.ceil_log2(n_states)
        wires = list(range(n_wires))
        diag_key = next(iter(k for k, v in fragment.get_coefficients().items() if v))
        M = qp.matrix(diagonalize_vibronic_mat, wires)(key=diag_key, wires=wires)[
            :n_states, :n_states
        ]

        constant, linear, quadratic, bilinear = get_position_coefficients(fragment)

        # 0th order
        assert np.shape(constant) == (n_states,)
        exp_order_zero = fragment_to_dense(fragment, ())
        # Make sure the diagonalization and extraction worked by inverting the np.diag call
        assert np.allclose(M.T @ exp_order_zero @ M, np.diag(constant))

        # 1st order
        assert np.shape(linear) == (n_modes, n_states)
        exp_order_one = fragment_to_dense(fragment, ("Q",))
        # Make sure the diagonalization and extraction worked by inverting the np.diag call
        exp_order_one = np.einsum("ba,bcz,cd->zad", M, exp_order_one, M)
        reconstructed_order_one = [np.diag(sub_diag) for sub_diag in linear]
        assert np.allclose(reconstructed_order_one, exp_order_one)

        # 2nd order
        assert np.shape(quadratic) == (n_modes, n_states)
        assert np.shape(bilinear) == (n_modes, n_modes, n_states)
        exp_order_two = fragment_to_dense(fragment, ("Q", "Q"))
        # Make sure the diagonalization and extraction worked by inverting the np.diag call
        exp_order_two = np.einsum("ba,bcyz,cd->yzad", M, exp_order_two, M)

        reconstructed_order_two = np.array([np.diag(sub_diag) for sub_diag in quadratic.T]).T
        reconstructed_order_two[np.triu_indices(n_modes, k=1)] = bilinear[
            np.triu_indices(n_modes, k=1)
        ]
        reconstructed_order_two = np.array(
            [[np.diag(_sub) for _sub in sub] for sub in reconstructed_order_two]
        )

        assert np.allclose(reconstructed_order_two, exp_order_two)

    @pytest.mark.parametrize("n_states, n_modes", [(3, 2), (5, 10), (14, 2), (19, 7)])
    def test_get_momentum_coefficients(self, n_states, n_modes, seed):
        """Test that ``get_momentum_coefficients`` returns the correct terms."""
        fragment = random_vibronic_fragment(
            n_states, n_modes, include_op_types=[("P", "P")], seed=seed
        )

        diag_key = next(iter(k for k, v in fragment.get_coefficients().items() if v))
        assert diag_key[0] == diag_key[1]  # Consistency check for fragment sampler

        quadratic = get_momentum_coefficients(fragment)
        assert np.shape(quadratic) == (n_modes,)

        exp_quadratic = fragment_to_dense(fragment, ("P", "P"))
        # Add redundant n_states axes and double n_modes axis
        reconstructed_quadratic = np.einsum("ab,cd->abcd", np.eye(n_states), np.diag(quadratic))
        assert np.allclose(reconstructed_quadratic, exp_quadratic)

    def test_preprocess_data(self):
        """Test time scaling and bilinear index flattening in `_preprocess_data`."""
        fragments = _vibronic_fragment_list(n_states=2, n_modes=3, seed=7)
        time = 0.8
        (constant, linear, quadratic, bilinear), bilinear_indices, diag_keys = _preprocess_data(
            time, fragments
        )

        n_position = len(fragments) - 1
        assert len(constant) == n_position
        assert len(diag_keys) == len(fragments)

        fragment = fragments[0]
        exp_constant, exp_linear, exp_quadratic, exp_bilinear = get_position_coefficients(fragment)
        scale = time / 2
        assert np.allclose(constant[0], exp_constant * scale)
        assert np.allclose(linear[0], exp_linear * scale)
        assert np.allclose(quadratic[0], exp_quadratic * scale)
        assert np.allclose(bilinear[0], exp_bilinear[*bilinear_indices] * scale)

        expected_indices = np.array(np.triu_indices(fragment.modes, 1))
        assert np.allclose(bilinear_indices, expected_indices)
        assert bilinear[0].shape == (expected_indices.shape[1], fragment.states)


class TestExtractRegisters:
    """Tests for ``_extract_registers``."""

    @pytest.fixture
    def register_setup(self):
        """Provide a minimal valid register layout for two modes."""
        n_states, n_modes, k, b = 2, 2, 2, 3
        registers = _make_registers(n_states, n_modes, k=k, b=b)
        mode_registers = [registers[f"mode {i}"] for i in range(n_modes)]
        non_mode_registers = {
            key: wires for key, wires in registers.items() if not key.startswith("mode ")
        }
        return non_mode_registers, mode_registers, k, b

    def test_constant_term(self, register_setup):
        """Test wire extraction for the constant-term adder."""
        registers, mode_registers, _, _ = register_setup
        wires = _extract_registers(registers, mode_registers, "constant")
        assert set(wires) == {"x_wires", "y_wires", "work_wires"}
        assert wires["x_wires"] == registers["coefficients"]
        assert wires["y_wires"] == registers["phase gradient"]
        assert wires["work_wires"] == registers["work"]

    def test_qrom_term(self, register_setup):
        """Test wire extraction for the shared QROM."""
        registers, mode_registers, _, _ = register_setup
        wires = _extract_registers(registers, mode_registers, "QROM")
        assert set(wires) == {"control_wires", "target_wires", "work_wires"}
        assert wires["control_wires"] == registers["electronic"]
        assert wires["target_wires"] == registers["coefficients"]
        assert wires["work_wires"] == registers["work"][: len(registers["electronic"]) - 1]

    def test_linear_term(self, register_setup):
        """Test wire extraction for a linear term."""
        registers, mode_registers, _, _ = register_setup
        wires = _extract_registers(registers, mode_registers, "linear", 1)
        assert set(wires) == {"x_wires", "y_wires", "output_wires", "work_wires"}
        assert wires["x_wires"] == registers["coefficients"]
        assert wires["y_wires"] == mode_registers[1]
        assert wires["output_wires"] == registers["phase gradient"]
        assert wires["work_wires"] == registers["work"]

    def test_quadratic_term_slices_cache(self, register_setup):
        """Test wire extraction for a quadratic term."""
        registers, mode_registers, k, _ = register_setup
        square_wires, mult_wires = _extract_registers(registers, mode_registers, "quadratic", 0)
        assert set(square_wires) == {"x_wires", "output_wires", "work_wires"}
        assert set(mult_wires) == {"x_wires", "y_wires", "output_wires", "work_wires"}
        assert square_wires["x_wires"] == mode_registers[0]
        assert square_wires["output_wires"] == registers["cache"][1:]
        assert len(square_wires["output_wires"]) == 2 * k - 1
        assert square_wires["work_wires"] == registers["work"]

        assert mult_wires["x_wires"] == registers["coefficients"]
        assert mult_wires["y_wires"] == registers["cache"][1:]
        assert mult_wires["output_wires"] == registers["phase gradient"]
        assert mult_wires["work_wires"] == registers["work"]

    def test_bilinear_term(self, register_setup):
        """Test wire extraction for a bilinear term."""
        registers, mode_registers, _, _ = register_setup
        mode_mult_wires, coeff_mult_wires = _extract_registers(
            registers, mode_registers, "bilinear", 0, 1
        )
        assert set(mode_mult_wires) == {"x_wires", "y_wires", "output_wires", "work_wires"}
        assert set(coeff_mult_wires) == {"x_wires", "y_wires", "output_wires", "work_wires"}
        assert mode_mult_wires["x_wires"] == mode_registers[0]
        assert mode_mult_wires["y_wires"] == mode_registers[1]
        assert mode_mult_wires["output_wires"] == registers["cache"]
        assert mode_mult_wires["work_wires"] == registers["work"]

        assert coeff_mult_wires["x_wires"] == registers["coefficients"]
        assert coeff_mult_wires["y_wires"] == registers["cache"]
        assert coeff_mult_wires["output_wires"] == registers["phase gradient"]
        assert coeff_mult_wires["work_wires"] == registers["work"]


class TestValidateFragments:
    """Tests for ``_validate_fragments``."""

    def test_accepts_vibronic_fragments(self):
        """Test that valid fragments from ``vibronic_fragments`` pass validation."""
        _validate_fragments(_vibronic_fragment_list())

    @pytest.mark.parametrize(
        "fragments, match",
        [
            (
                [random_vibronic_fragment(2, 2, include_op_types=[("P", "P")], seed=1)],
                "Expected at least one potential and one kinetic fragment",
            ),
            (
                [
                    random_vibronic_fragment(2, 2, seed=2),
                    random_vibronic_fragment(3, 2, seed=3),
                    random_vibronic_fragment(2, 2, include_op_types=[("P", "P")], seed=4),
                ],
                "Expected all vibronic fragments to have the same number of electronic states",
            ),
            (
                [
                    random_vibronic_fragment(2, 2, seed=5),
                    random_vibronic_fragment(2, 3, seed=6),
                    random_vibronic_fragment(2, 2, include_op_types=[("P", "P")], seed=7),
                ],
                "Expected all vibronic fragments to have the same number of vibrational modes",
            ),
        ],
    )
    def test_rejects_invalid_fragment_lists(self, fragments, match):
        """Test that invalid fragment lists are rejected."""
        with pytest.raises(ValueError, match=match):
            _validate_fragments(fragments)

    def test_rejects_kinetic_fragment_in_position_slot(self):
        """Test that a kinetic fragment before the last slot is rejected."""
        fragments = _vibronic_fragment_list()
        fragments[0] = fragments[-1]
        with pytest.raises(ValueError, match="position terms of at most second order"):
            _validate_fragments(fragments)

    def test_rejects_position_fragment_as_last(self):
        """Test that the last fragment must be kinetic."""
        fragments = _vibronic_fragment_list()
        fragments[-1] = fragments[0]
        with pytest.raises(ValueError, match="kinetic terms only"):
            _validate_fragments(fragments)


class TestValidateRegisters:
    """Tests for ``_validate_registers``."""

    def test_accepts_valid_registers(self):
        """Test that a correctly sized register dict passes validation."""
        fragments = _vibronic_fragment_list()
        registers = _make_registers(fragments[0].states, fragments[0].modes)
        _validate_registers(registers, n_modes=fragments[0].modes, n_states=fragments[0].states)

    @pytest.mark.parametrize(
        "registers, match",
        [
            ("not a dict", "Expected `registers` to be a dictionary"),
            ({}, "Expected the keys in `registers`"),
        ],
    )
    def test_rejects_invalid_register_container(self, registers, match):
        """Test that invalid register containers are rejected."""
        fragments = _vibronic_fragment_list()
        with pytest.raises(ValueError, match=match):
            _validate_registers(registers, n_modes=fragments[0].modes, n_states=fragments[0].states)

    def test_rejects_register_size_mismatches(self):
        """Test that incorrectly sized registers are rejected."""
        fragments = _vibronic_fragment_list(n_states=4, n_modes=2)
        n_states, n_modes = fragments[0].states, fragments[0].modes

        registers = _make_registers(n_states, n_modes)
        registers["electronic"] = registers["electronic"][:-1]
        with pytest.raises(ValueError, match="electronic states"):
            _validate_registers(registers, n_modes=n_modes, n_states=n_states)

        registers = _make_registers(n_states, n_modes)
        registers["cache"] = registers["cache"][:-1]
        with pytest.raises(ValueError, match="cache qubits"):
            _validate_registers(registers, n_modes=n_modes, n_states=n_states)

        registers = _make_registers(n_states, n_modes)
        registers["phase gradient"] = registers["phase gradient"][:-2]
        with pytest.raises(ValueError, match="phase gradient"):
            _validate_registers(registers, n_modes=n_modes, n_states=n_states)

        registers = _make_registers(n_states, n_modes)
        registers["work"] = registers["work"][:-1]
        with pytest.raises(ValueError, match="work qubits"):
            _validate_registers(registers, n_modes=n_modes, n_states=n_states)

        registers = _make_registers(n_states, n_modes)
        registers["mode 1"] = registers["mode 1"] + [99]
        with pytest.raises(ValueError, match="same size"):
            _validate_registers(registers, n_modes=n_modes, n_states=n_states)


class TestTrotterVibronic:
    """Tests for ``trotter_vibronic`` validation and side effects."""

    def test_rejects_invalid_num_trotter_steps(self):
        """Test that non-positive step counts are rejected."""
        fragments = _vibronic_fragment_list()
        registers = _make_registers(fragments[0].states, fragments[0].modes)
        with pytest.raises(ValueError, match="positive integer"):
            trotter_vibronic(1.0, 0, fragments, registers, aqft_order=1)

    def test_load_coefficients_behaviour(self):
        """Test that incremental bitstrings are computed and a QROM is queued."""
        precision = 3
        coefficients = np.array([0.1, 0.2])
        prev_bitstrings = np.zeros((len(coefficients), precision), dtype=int)
        qrom_wires = {
            "control_wires": [0, 1],
            "target_wires": [2, 3, 4],
            "work_wires": [5],
        }

        with qp.queuing.AnnotatedQueue() as q:
            new_bitstrings = load_coefficients(coefficients, precision, prev_bitstrings, qrom_wires)

        expected_new = qp.math.binary_decimals(coefficients, precision, unit=2 * np.pi)
        expected_change = (prev_bitstrings + expected_new) % 2
        assert np.allclose(new_bitstrings, expected_new)
        assert len(q.queue) == 1
        queued_op = q.queue[0]
        assert isinstance(queued_op, qp.QROM)
        assert np.allclose(queued_op.data[0], expected_change)
