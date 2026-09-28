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
"""Tests for the one-body block-encoding."""

import numpy as np
import pytest

import pennylane as qp
from pennylane.fermi import FermiSentence, FermiWord, jordan_wigner
from pennylane.ops.functions.assert_valid import assert_valid
from pennylane.templates.subroutines.alias_sampling import _build_alias_tables
from pennylane.typing import AbstractWires


def _registers(norbs, mu_bits):
    """Return the three registers with the sizes from ``one_body_block_encoding_wires``."""
    req = qp.one_body_block_encoding_wires(norbs, mu_bits)
    n_prep, n_sys, n_work = req["prep_wires"], req["system_wires"], req["work_wires"]
    prep = list(range(n_prep))
    system = list(range(n_prep, n_prep + n_sys))
    work = list(range(n_prep + n_sys, n_prep + n_sys + n_work))
    return prep, system, work


_IDENTITY = ((1.0, 0.0), (0.0, 1.0))


def _static(op_matrix):
    """Return ``op_matrix`` as the nested tuple that ``OneBodyBlockEncoding`` requires.

    The numerical tests build their matrices with numpy, but ``op_matrix`` is a compilable
    argument, so the operator only accepts hashable static data.
    """
    return tuple(tuple(float(entry) for entry in row) for row in np.asarray(op_matrix))


def _discretized_weights(weights, mu_bits):
    r"""Return the probability distribution that coherent alias sampling actually prepares.

    The only approximation of :class:`~.OneBodyBlockEncoding` is that PREP loads a ``mu_bits``-bit
    approximation of :math:`\sqrt{|\mu_p| / \lambda}` rather than the exact value. Because the
    alias tables ``alt`` and ``keep`` are deterministic, that approximation is classically
    predictable:

    .. math::

        \tilde{\rho}_\ell = \frac{1}{L 2^\mu} \Big( \mathrm{keep}_\ell +
        \sum_{k \,:\, \mathrm{alt}_k = \ell} (2^\mu - \mathrm{keep}_k) \Big) ,

    where :math:`L` is the number of weights and :math:`\mu` is ``mu_bits``. The tests can
    therefore compare the block-encoding against :math:`\tilde{\rho}` to machine precision,
    instead of against the exact weights to the loose ``L / 2**mu_bits`` alias bound.

    Args:
        weights (array): the non-negative weights :math:`|\mu_p|` that PREP loads
        mu_bits (int): number of bits of precision used by coherent alias sampling

    Returns:
        array: the normalized probabilities :math:`\tilde{\rho}_\ell`, of length ``len(weights)``

    """
    alt, keep = _build_alias_tables(weights, mu_bits)
    n_states, n_keep = len(alt), 2**mu_bits
    rho = np.zeros(n_states)
    for target in range(n_states):
        rho[target] += keep[target]
        for source in range(n_states):
            if alt[source] == target:
                rho[target] += n_keep - keep[source]
    return rho / (n_keep * n_states)


def _reference_block_matrix(op_matrix, system_wires, mu_bits=None):
    r"""Independent reference for the encoded block.

    Builds

    .. math::

        -\frac{1}{2} \sum_{p,\sigma} \rho_p\, \mathrm{sign}(\mu_p)\,
        \hat V^\dagger \hat Z_{p\sigma} \hat V

    The rotated Pauli comes from Jordan-Wigner mapping
    :math:`\sum_{qs} V_{qp} V_{sp} c_q^\dagger c_s` and using
    :math:`\hat Z = \hat 1 - 2 \hat n`.

    Args:
        op_matrix (array): the real symmetric one-body matrix.
        system_wires (list[int]): wires for representing the ``2 * norbs`` system spin-orbitals
        mu_bits (int or None): alias sampling precision; if None, the exact weights are used.

    Returns:
        array: the Hermitian matrix of shape ``(2**(2 * norbs), 2**(2 * norbs))`` that
        :class:`~.OneBodyBlockEncoding` is expected to encode in its :math:`|\vec 0\rangle` block,
        namely :math:`\hat O / \lambda`. With ``mu_bits=None`` this is the exact
        :math:`\hat O / \lambda`; otherwise it uses the discretized weights that PREP
        really loads.

    """
    norbs = qp.math.shape(op_matrix)[0]
    mu, vmat = np.linalg.eigh(op_matrix)
    weights = np.abs(mu)
    signs = np.where(weights > 0, np.sign(mu), 1.0)
    rho = weights / weights.sum() if mu_bits is None else _discretized_weights(weights, mu_bits)

    # The sum over p is done classically: with V^dag n_{p sigma} V = sum_{qs} V_qp V_sp
    # c^dag_{q sigma} c_{s sigma}, one has sum_p rho_p sign(mu_p) V^dag n_{p sigma} V =
    # sum_{qs} coeffs_qs c^dag_{q sigma} c_{s sigma} with coeffs = V diag(rho * sign(mu)) V^T.
    # Only one Jordan-Wigner mapping per spin sector is left.
    coeffs = vmat @ np.diag(rho * signs) @ vmat.T

    dim = 2 ** (2 * norbs)
    wire_map = {m: system_wires[m] for m in range(2 * norbs)}
    total = np.sum(rho * signs) * 2 * np.eye(dim, dtype=complex)
    for sigma in (0, 1):
        fermi_op = FermiSentence(
            {
                FermiWord({(0, sigma * norbs + q): "+", (1, sigma * norbs + s): "-"}): coeffs[q, s]
                for q in range(norbs)
                for s in range(norbs)
            }
        )
        total -= 2 * qp.matrix(jordan_wigner(fermi_op, wire_map=wire_map), wire_order=system_wires)
    return -0.5 * total


def _apply_block_encoding(op_matrix, mu_bits, state):
    r"""Apply the block-encoding to one system state and project onto :math:`|\vec 0\rangle`.

    Args:
        op_matrix (array): the real symmetric one-body matrix
        mu_bits (int): number of bits of precision used by coherent alias sampling
        state (array): normalized system state of dimension ``2**(2 * norbs)``

    Returns:
        tuple[array, float]: the ``2**(2 * norbs)`` system amplitudes left after projecting both
        the prep and the work register onto :math:`|\vec 0\rangle`, and the norm of the full state
        outside :math:`|\vec 0\rangle` on the work register, which must vanish because the
        block-encoding returns the work wires to :math:`|0\rangle`
    """
    norbs = qp.math.shape(op_matrix)[0]
    prep, system, work = _registers(norbs, mu_bits)
    n_prep, n_sys, n_work = len(prep), len(system), len(work)

    n_wires = n_prep + n_sys + n_work

    dev = qp.device("default.qubit", wires=n_wires)

    @qp.qnode(dev)
    def circuit():
        qp.StatePrep(state, wires=system)
        qp.OneBodyBlockEncoding(_static(op_matrix), mu_bits, prep, system, work)
        return qp.state()

    psi = np.asarray(circuit()).reshape(2**n_prep, 2**n_sys, 2**n_work)
    work_scratch = float(np.linalg.norm(psi[:, :, 1:]))
    return psi[0, :, 0], work_scratch


def _encoded_block(op_matrix, mu_bits):
    r"""Build the full encoded block, one column per system basis state.

    Args:
        op_matrix (array): the real symmetric one-body matrix
        mu_bits (int): number of bits of precision used by coherent alias sampling

    Returns:
        array: the matrix of shape ``(2**(2 * norbs), 2**(2 * norbs))`` encoded in the
        :math:`|\vec 0\rangle` block, directly comparable to :func:`_reference_block_matrix`
    """
    norbs = qp.math.shape(op_matrix)[0]
    _, system, _ = _registers(norbs, mu_bits)
    dim = 2 ** len(system)

    block = np.zeros((dim, dim), dtype=complex)
    for column in range(dim):
        basis = np.zeros(dim)
        basis[column] = 1.0
        block[:, column], _ = _apply_block_encoding(op_matrix, mu_bits, basis)

    return block


@pytest.mark.parametrize("norbs", [2, 3, 4, 16])
@pytest.mark.parametrize("mu_bits", [2, 4, 7])
def test_one_body_block_encoding_wires(norbs, mu_bits):
    """Test that the wire counts match the alias sampling wires."""
    req = qp.one_body_block_encoding_wires(norbs, mu_bits)
    alias = qp.alias_sampling_wires(norbs, mu_bits)

    assert req["prep_wires"] == alias["target_wires"] + 1 + alias["temp_wires"]
    assert req["system_wires"] == 2 * norbs
    assert req["work_wires"] == max(alias["work_wires"], alias["target_wires"])


class TestOneBodyBlockEncoding:
    """Test the validity of the operator, the block it encodes, and its input validation.

    The tests of the encoded block run the real ``AliasSampling`` and compare against
    ``_reference_block_matrix``, which accounts for the ``mu_bits`` discretization, so they hold
    to machine precision instead of only to the ``L / 2**mu_bits`` alias bound.
    """

    @pytest.mark.usefixtures("enable_and_disable_capture")
    @pytest.mark.parametrize(
        "op_matrix",
        [
            ((1.0, 2.0), (2.0, 1.0)),  # one negative eigenvalue
            ((2.0, 0.5), (0.5, 2.0)),  # positive definite, so no sign phase
        ],
    )
    def test_assert_valid(self, op_matrix):
        """Test that OneBodyBlockEncoding is a valid Operator2, with and without capture.

        ``assert_valid`` already covers the decomposition rules of both the operator and its
        adjoint. The two matrices cover the two branches of the decomposition: with and without
        the ``LeftClassicalComparator`` sign phase, which is only applied when some eigenvalue is
        negative.
        """
        prep, system, work = _registers(2, 2)
        op = qp.OneBodyBlockEncoding(op_matrix, 2, prep, system, work)
        assert_valid(op, skip_differentiation=True)

    def test_hyperparameters_and_wires(self):
        """Test that the registers and the static data round-trip through the operator."""
        prep, system, work = _registers(2, 2)
        op = qp.OneBodyBlockEncoding(((1.0, 2.0), (2.0, 1.0)), 2, prep, system, work)
        assert op.op_matrix == ((1.0, 2.0), (2.0, 1.0))
        assert op.alias_sampling_nbits == 2
        assert op.prep_wires == qp.wires.Wires(prep)
        assert op.system_wires == qp.wires.Wires(system)
        assert op.work_wires == qp.wires.Wires(work)
        # ``work_wires`` are auxiliary and excluded from ``wires``, as for AliasSampling
        assert op.wires == qp.wires.Wires(prep + system)

    def test_op_matrix_is_hashable_static_data(self):
        """Test that ``op_matrix`` is stored as hashable static data.

        ``op_matrix`` is a compilable argument, so it travels in the hashable metadata of the
        operator's pytree and in the parameters of the captured jaxpr equation, both of which JAX
        requires to be hashable. Traced inputs are rejected before ``__init__`` by the metaclass,
        covered in ``tests/core/operator/test_operator2_metaclass.py``.
        """
        prep, system, work = _registers(2, 2)
        op = qp.OneBodyBlockEncoding(((1.0, 2.0), (2.0, 1.0)), 2, prep, system, work)
        assert hash(tuple(op.compilable_args.values()))

    @pytest.mark.parametrize(
        "op_matrix",
        [
            [[1.0, 2.0], [2.0, 1.0]],  # list of lists
            ([1.0, 2.0], [2.0, 1.0]),  # tuple of lists
            np.array([[1.0, 2.0], [2.0, 1.0]]),
        ],
    )
    def test_tensor_like_op_matrix_raises(self, op_matrix):
        """Test that tensor-like input is rejected rather than converted.

        A compilable argument must be hashable static data, so accepting an unhashable container
        here would only defer the failure to capture time.
        """
        prep, system, work = _registers(2, 2)
        with pytest.raises(ValueError, match="must be a tuple of tuples of floats"):
            qp.OneBodyBlockEncoding(op_matrix, 2, prep, system, work)

    @pytest.mark.parametrize("norbs, mu_bits", [(2, 1), (2, 2), (2, 3), (3, 2)])
    def test_encodes_operator_on_random_state(self, norbs, mu_bits):
        """Test that the block reproduces the discretized operator on a random state."""

        rng = np.random.default_rng(10 * norbs + mu_bits)
        a = rng.standard_normal((norbs, norbs))
        op_matrix = (a + a.T) / 2

        _, system, _ = _registers(norbs, mu_bits)

        dim = 2 ** (2 * norbs)
        state = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        state /= np.linalg.norm(state)

        block_encoded_state, work_scratch = _apply_block_encoding(op_matrix, mu_bits, state)

        expected = _reference_block_matrix(op_matrix, system, mu_bits) @ state
        assert np.allclose(block_encoded_state, expected, atol=1e-8)
        assert work_scratch < 1e-10

        # the discretized block also stays within the alias bound of the exact operator
        exact = _reference_block_matrix(op_matrix, system) @ state
        assert np.allclose(block_encoded_state, exact, atol=np.linalg.norm(state) / 2**mu_bits)

    def test_encodes_operator_full_block(self):
        """Test that the full block matches column by column, not just on one vector."""

        rng = np.random.default_rng(1000)
        a = rng.standard_normal((2, 2))
        op_matrix = (a + a.T) / 2
        _, system, _ = _registers(2, 2)

        block = _encoded_block(op_matrix, 2)

        assert np.allclose(block, _reference_block_matrix(op_matrix, system, 2), atol=1e-8)

    @pytest.mark.parametrize("norbs", [2, 3])
    def test_negative_definite_spectrum(self, norbs):
        """Test that a negative-definite spectrum is encoded correctly, with the signs of the
        eigenvalues phased onto the index register and only their magnitudes loaded by PREP.
        """
        rng = np.random.default_rng(1000 * norbs)
        a = rng.standard_normal((norbs, norbs))
        op_matrix = (a + a.T) / 2
        op_matrix -= (np.linalg.eigvalsh(op_matrix).max() + 0.5) * np.eye(norbs)
        assert np.all(np.linalg.eigvalsh(op_matrix) < 0)

        _, system, _ = _registers(norbs, 2)
        dim = 2 ** (2 * norbs)
        state = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        state /= np.linalg.norm(state)

        got, _ = _apply_block_encoding(op_matrix, 2, state)

        assert np.allclose(got, _reference_block_matrix(op_matrix, system, 2) @ state, atol=1e-8)

    def test_singular_matrix(self):
        """Test that a zero eigenvalue is encoded correctly: zero weight and no sign phase."""
        op_matrix = np.array([[1.0, 1.0], [1.0, 1.0]])

        _, system, _ = _registers(2, 2)
        dim = 2 ** (2 * 2)
        rng = np.random.default_rng(1000)
        state = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        state /= np.linalg.norm(state)

        got, _ = _apply_block_encoding(op_matrix, 2, state)

        assert np.allclose(got, _reference_block_matrix(op_matrix, system, 2) @ state, atol=1e-8)

    @pytest.mark.parametrize("mu_bits", [1, 2, 3])
    def test_within_precision_bound(self, mu_bits):
        """Test that the gap to the ideal operator stays inside the O(L / 2**mu) alias bound."""
        rng = np.random.default_rng(1000 * mu_bits)
        a = rng.standard_normal((2, 2))
        op_matrix = (a + a.T) / 2

        _, system, _ = _registers(2, mu_bits)

        block = _encoded_block(op_matrix, mu_bits)

        error = np.abs(block - _reference_block_matrix(op_matrix, system_wires=system)).max()
        assert error <= 1 / 2**mu_bits

    def test_non_square_raises(self):
        """Test that a non-square op_matrix is rejected."""
        prep, system, work = _registers(2, 2)
        with pytest.raises(ValueError, match="must be square"):
            qp.OneBodyBlockEncoding(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), 2, prep, system, work)

    def test_single_orbital_raises(self):
        """Test that a single spatial orbital is rejected: the index register would be empty."""
        prep, system, work = _registers(1, 2)
        with pytest.raises(ValueError, match="at least two spatial orbitals"):
            qp.OneBodyBlockEncoding(((2.0,),), 2, prep, system, work)

    def test_complex_raises(self):
        """Test that a complex op_matrix is rejected."""
        prep, system, work = _registers(2, 2)
        with pytest.raises(ValueError, match="must be real"):
            qp.OneBodyBlockEncoding(((1j, 0.0), (0.0, 1j)), 2, prep, system, work)

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_non_finite_raises(self, bad):
        """Test that a non-finite op_matrix is rejected."""
        prep, system, work = _registers(2, 2)
        with pytest.raises(ValueError, match="must be finite"):
            qp.OneBodyBlockEncoding(((1.0, bad), (bad, 1.0)), 2, prep, system, work)

    def test_non_symmetric_raises(self):
        """Test that a non-symmetric op_matrix is rejected."""
        prep, system, work = _registers(2, 2)
        with pytest.raises(ValueError, match="must be symmetric"):
            qp.OneBodyBlockEncoding(((1.0, 2.0), (0.0, 1.0)), 2, prep, system, work)

    @pytest.mark.parametrize("nbits", [True, 0, -1, 2.0])
    def test_invalid_nbits_raises(self, nbits):
        """Test that alias_sampling_nbits must be a positive integer."""
        prep, system, work = _registers(2, 2)
        with pytest.raises(ValueError, match="alias_sampling_nbits must be a positive integer"):
            qp.OneBodyBlockEncoding(_IDENTITY, nbits, prep, system, work)

    @pytest.mark.parametrize("register", ["prep_wires", "system_wires", "work_wires"])
    def test_wrong_register_size_raises(self, register):
        """Test that each register must have the reported size."""
        prep, system, work = _registers(2, 2)
        registers = {"prep_wires": prep, "system_wires": system, "work_wires": work}
        registers[register] = registers[register][:-1]
        with pytest.raises(ValueError, match=f"{register} must have"):
            qp.OneBodyBlockEncoding(
                _IDENTITY,
                2,
                registers["prep_wires"],
                registers["system_wires"],
                registers["work_wires"],
            )

    def test_overlapping_registers_raise(self):
        """Test that the three registers must be disjoint."""
        prep, system, work = _registers(2, 2)
        with pytest.raises(ValueError, match="must not overlap"):
            qp.OneBodyBlockEncoding(_IDENTITY, 2, prep, system, prep[: len(work)])

    def test_abstract_wires_length_is_validated(self):
        """Test that register sizes are checked for AbstractWires, which still expose a length."""
        req = qp.one_body_block_encoding_wires(2, 2)
        with pytest.raises(ValueError, match="prep_wires must have"):
            qp.OneBodyBlockEncoding(
                _IDENTITY,
                2,
                AbstractWires(req["prep_wires"] - 1),
                AbstractWires(req["system_wires"]),
                AbstractWires(req["work_wires"]),
            )
        op = qp.OneBodyBlockEncoding(
            _IDENTITY,
            2,
            AbstractWires(req["prep_wires"]),
            AbstractWires(req["system_wires"]),
            AbstractWires(req["work_wires"]),
        )
        assert isinstance(op.prep_wires, AbstractWires)

    def test_zero_matrix_raises(self):
        """Test that an all-zero op_matrix has lambda = 0 and cannot be normalized."""
        prep, system, work = _registers(2, 2)
        with pytest.raises(ValueError, match="positive value"):
            qp.OneBodyBlockEncoding(((0.0, 0.0), (0.0, 0.0)), 2, prep, system, work)
