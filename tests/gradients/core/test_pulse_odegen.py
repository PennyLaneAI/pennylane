# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Tests for the gradients.pulse_odegen module.
"""
# pylint:disable=import-outside-toplevel, use-implicit-booleaness-not-comparison

import copy

import numpy as np
import pytest

import pennylane as qml
from pennylane import numpy as pnp
from pennylane.gradients.pulse_gradient_odegen import (
    _generate_tapes_and_coeffs,
    _insert_op,
    _nonzero_coeffs_and_words,
    _one_parameter_generators,
    _one_parameter_paulirot_coeffs,
    _parshift_and_contract,
    pulse_odegen,
)
from pennylane.math import expand_matrix
from pennylane.ops.qubit.special_unitary import pauli_basis_matrices, pauli_basis_strings

X, Y, Z = qml.PauliX, qml.PauliY, qml.PauliZ


def grad_fn(tape, dev, fn=pulse_odegen, **kwargs):
    """Utility function to automate execution and processing of gradient tapes"""
    _tapes, fn = fn(tape, **kwargs)
    return fn(dev.execute(_tapes)), _tapes


def integral_of_polyval(params, t):
    """Compute the time integral of polyval."""
    from jax import numpy as jnp

    if qml.math.ndim(t) == 0:
        t = [0, t]
    start, end = t
    new_params = jnp.concatenate([params / jnp.arange(len(params), 0, -1), jnp.array([0])])
    return jnp.polyval(new_params, end) - jnp.polyval(new_params, start)


@pytest.mark.jax
class TestOneParameterGenerators:
    """Test the utility function _one_parameter_generators."""

    @pytest.mark.parametrize("term", [X(0), Y("a") @ X(3)])
    @pytest.mark.parametrize("t", ([0.3, 0.4], [-0.1, 0.1]))
    def test_with_single_const_term_ham(self, term, t):
        """Test that the generators are correct for a single-term constant Hamiltonian."""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)

        H = qml.pulse.constant * term
        params = [jnp.array(0.4)]
        T = t[1] - t[0]

        op = qml.evolve(H)(params, t)
        gens = _one_parameter_generators(op)

        # op has a single parameter in its parameter input list, so the
        # generators should be a length-1 tuple.
        assert isinstance(gens, tuple)
        assert len(gens) == 1
        gen = gens[0]
        # The effective generator should have the shape of the pulse
        assert gen.shape == (2 ** len(term.wires),) * 2
        # A Hamiltonian with a single term and a constant pulse envelope yields
        # a rotation gate about said term, with the effective rotation parameter
        # given by the duration of the constant pulse.
        expected = -1j * T * term.matrix()
        assert qml.math.allclose(gen, expected)

    @pytest.mark.parametrize(
        "terms",
        [
            [X(0), Z(1), X(0) @ Z(1)],
            [X(0), X(0)],
            [X(0), Z(1), Y(3)],
            [
                Y("a") @ X(3),
                X(3) @ Z(0),
                Y("a") @ Z(0),
            ],
        ],
    )
    @pytest.mark.parametrize("t", ([0.3, 0.4], [-0.1, 0.1]))
    def test_with_commuting_const_terms_ham(self, terms, t):
        """Test that the generators are correct for a Hamiltonian with multiple
        constant commuting terms."""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)

        num_terms = len(terms)
        H = qml.dot([qml.pulse.constant for _ in range(num_terms)], terms)
        params = [jnp.array(0.4), jnp.array(0.9), jnp.array(-0.5)][:num_terms]
        T = t[1] - t[0]

        op = qml.evolve(H, dense=True)(params, t)
        gens = _one_parameter_generators(op)

        # op has as many parameters in its parameter input list as there are terms,
        # so generators should be a tuple with as many entries as Hamiltonian terms.
        assert isinstance(gens, tuple)
        assert len(gens) == num_terms
        dim = 2 ** len(H.wires)
        # A Hamiltonian with commuting terms and constant pulse envelopes yields
        # a product of rotation gates about its terms, with the effective rotation parameters
        # given by the duration of the constant pulse.
        for gen, term in zip(gens, terms):
            # The effective generator should have the shape of the pulse
            assert gen.shape == (dim, dim)
            expected = -1j * T * expand_matrix(term.matrix(), term.wires, H.wires)
            assert qml.math.allclose(gen, expected)

    @pytest.mark.parametrize(
        "terms",
        [
            [X(0), Z(0), Y(0)],
            [Y("a") @ X(3), X(3) @ Z("a")],
        ],
    )
    @pytest.mark.parametrize("t", ([0.3, 0.4], [-0.1, 0.1]))
    def test_with_noncommuting_const_terms_ham(self, terms, t):
        """Test that the generators are correct for a Hamiltonian with multiple
        constant non-commuting terms."""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)

        def manual_matrix(params):
            """Compute the matrix of a constant pulse with non-commuting Hamiltonian terms."""
            summands = [
                p * expand_matrix(term.matrix(), term.wires, H.wires)
                for p, term in zip(params, terms)
            ]
            exp = jnp.sum(
                jnp.array(summands),
                axis=0,
            )
            return jax.scipy.linalg.expm(-1j * T * exp)

        num_terms = len(terms)
        H = qml.dot([qml.pulse.constant for _ in range(num_terms)], terms)
        params = [jnp.array(0.4), jnp.array(0.9), jnp.array(-0.5), jnp.array(0.28)][:num_terms]
        T = t[1] - t[0]

        op = qml.evolve(H)(params, t)
        gens = _one_parameter_generators(op)

        # op has as many parameters in its parameter input list as there are terms,
        # so generators should be a tuple with as many entries as Hamiltonian terms.
        assert isinstance(gens, tuple)
        assert len(gens) == num_terms
        dim = 2 ** len(H.wires)
        params = [p.astype(jnp.complex128) for p in params]
        # Manually compute the matrix of the pulse and its derivative. Compose the two into the
        # effective generator. Omega = U* @ dU
        U = manual_matrix(params)
        expected = [U.conj().T @ j for j in jax.jacobian(manual_matrix, holomorphic=True)(params)]
        for gen, expec in zip(gens, expected):
            # The effective generator should have the shape of the pulse
            assert gen.shape == (dim, dim)
            assert qml.math.allclose(gen, expec)

    @pytest.mark.parametrize("term", [X(0), Y("a") @ X(3)])
    @pytest.mark.parametrize("t", ([0.3, 0.4], [-0.1, 0.1]))
    def test_with_single_timedep_term_ham(self, term, t):
        """Test that the generators are correct for a single-term time-dependent Hamiltonian."""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)

        H = jnp.polyval * term
        params = [jnp.array([0.4, 0.2, 0.1])]
        # Jacobian of the effective rotation parameter
        par_fn_jac = jax.grad(integral_of_polyval)(params[0], t)

        op = qml.evolve(H)(params, t)
        gens = _one_parameter_generators(op)

        # op has a single parameter in its parameter input list, so the
        # generators should be a length-1 tuple.
        assert isinstance(gens, tuple)
        assert len(gens) == 1
        gen = gens[0]
        # The effective generator should have the shape of the parameters and the pulse
        # together. Here, the parameters are one-dimensional, so we get (num_params, dim, dim)
        assert gen.shape == (len(params[0]), dim := 2 ** len(term.wires), dim)
        # The outer product of the rotation parameter Jacobian and the matrix of
        # the Hamiltonian term yields the Jacobian of the pulse
        expected = jnp.tensordot(par_fn_jac, -1j * term.matrix(), axes=0)
        assert qml.math.allclose(gen, expected)

    @pytest.mark.parametrize(
        "terms",
        [
            [X(0), Z(1), X(0) @ Z(1)],
            [X(0), X(0)],
            [X(0), Z(1), Y(3)],
            [
                Y("a") @ X(3),
                X(3) @ Z(0),
                Y("a") @ Z(0),
            ],
        ],
    )
    @pytest.mark.parametrize("t", ([0.3, 0.4], [-0.1, 0.1]))
    def test_with_commuting_timedep_terms_ham(self, terms, t):
        """Test that the generators are correct for a Hamiltonian with multiple
        commuting time-dependent terms."""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)

        num_terms = len(terms)
        H = qml.dot([jnp.polyval for _ in range(num_terms)], terms)
        params = [jnp.array([0.4, 0.1, 0.2]), jnp.array([0.9, -0.2, 0.5]), jnp.array([-0.5, 0.2])]
        params = params[:num_terms]
        # Jacobian functions of the effective rotation parameter (all polyval)
        par_fn_jac_fn = jax.grad(integral_of_polyval)
        # Jacobian of the effective rotation parameter for the different terms
        par_fn_jac = [par_fn_jac_fn(p, t) for p in params]

        op = qml.evolve(H, dense=True, atol=1e-9)(params, t)
        gens = _one_parameter_generators(op)

        # op has as many parameters in its parameter input list as there are terms,
        # so generators should be a tuple with as many entries as Hamiltonian terms.
        assert isinstance(gens, tuple)
        assert len(gens) == num_terms
        dim = 2 ** len(H.wires)
        for gen, term, p, jac in zip(gens, terms, params, par_fn_jac):
            # The effective generator should have the shape of the parameters and the pulse
            # together. Here, the parameters are one-dimensional, so we get (num_params, dim, dim)
            assert gen.shape == (len(p), dim, dim)
            expected = jnp.tensordot(
                jac, -1j * expand_matrix(term.matrix(), term.wires, H.wires), axes=0
            )
            assert qml.math.allclose(gen, expected)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "terms",
        [
            [X(0), Z(1), Y(0) @ Y(1)],
            [X(0), Z(0)],
            [X(0), Z(0), Y(0)],
            [
                Y("a") @ X(3),
                Y(3) @ Z(0),
                X("a") @ X(0),
            ],
        ],
    )
    @pytest.mark.parametrize("t", ([0.3, 0.4], [-0.1, -0.05]))
    def test_with_noncommuting_timedep_terms_ham(self, terms, t):
        """Test that the generators are correct for a Hamiltonian with multiple
        commuting time-dependent terms."""
        import jax
        import jax.numpy as jnp
        from jax.experimental.ode import odeint

        jax.config.update("jax_enable_x64", True)

        num_terms = len(terms)
        H = qml.dot([jnp.polyval for _ in range(num_terms)], terms)
        mats = [expand_matrix(term.matrix(), term.wires, H.wires) for term in terms]
        t = jnp.array(t)

        def manual_matrix(params):
            def apply_mat(y, t):
                H = jnp.sum(
                    jnp.array([jnp.polyval(p, t) * mat for p, mat in zip(params, mats)]), axis=0
                )
                return (-1j * H) @ y

            return odeint(apply_mat, jnp.eye(2 ** len(H.wires), dtype=complex), t, atol=1e-7)[-1]

        params = [jnp.array([0.4, 0.1, 0.2]), jnp.array([0.9, -0.2, 0.5]), jnp.array([-0.5, 0.2])]
        params = params[:num_terms]

        op = qml.evolve(H, dense=True, atol=1e-7)(params, t)
        gens = _one_parameter_generators(op)

        # op has as many parameters in its parameter input list as there are terms,
        # so generators should be a tuple with as many entries as Hamiltonian terms.
        assert isinstance(gens, tuple)
        assert len(gens) == num_terms
        dim = 2 ** len(H.wires)

        params = [p.astype(jnp.complex128) for p in params]
        # Manually compute the matrix of the pulse and its derivative. Compose the two into the
        # effective generator. Omega = U* @ dU
        U = manual_matrix(params)
        expected = [
            jnp.transpose(jnp.tensordot(U.conj().T, j, 1), (2, 0, 1))
            for j in jax.jacobian(manual_matrix, holomorphic=True)(params)
        ]
        for gen, expec, p in zip(gens, expected, params):
            # Each effective generator should have the shape of the pulse, and there should be
            # as many as values in the respective parameter
            assert gen.shape == (len(p), dim, dim)
            assert qml.math.allclose(gen, expec, atol=1e-6)


@pytest.mark.jax
class TestOneParameterPauliRotCoeffs:
    """Test the utility function _one_parameter_paulirot_coeffs."""

    @pytest.mark.parametrize("num_wires", [1, 2, 3])
    @pytest.mark.parametrize("pardims", [[tuple()], [(3,)], [(2,), tuple(), (3,)]])
    @pytest.mark.parametrize(
        "r_dtype, c_dtype", [("float64", "complex128"), ("float32", "complex64")]
    )
    def test_output_properties(self, num_wires, pardims, r_dtype, c_dtype):
        """Test that the output for a skew-Hermitian input is real-valued
        and has the correct shape(s)."""
        import jax.numpy as jnp

        r_dtype, c_dtype = getattr(jnp, r_dtype), getattr(jnp, c_dtype)
        dim = 2**num_wires
        gen_shapes = [pardim + (dim, dim) for pardim in pardims]
        gens = tuple(np.random.random(sh) + 1j * np.random.random(sh) for sh in gen_shapes)
        # make skew-Hermitian and cast to right complex dtype
        gens = tuple((g - np.moveaxis(g.conj(), -2, -1)).astype(c_dtype) for g in gens)

        coeffs = _one_parameter_paulirot_coeffs(gens, num_wires)
        assert isinstance(coeffs, tuple)
        assert len(coeffs) == len(pardims)
        assert all(c.shape == (4**num_wires - 1, *pardim) for c, pardim in zip(coeffs, pardims))
        assert all(c.dtype == r_dtype for c in coeffs)

    @pytest.mark.parametrize("num_wires", [1, 2, 3])
    def test_with_pauli_basis(self, num_wires):
        """Test that the generators of all possible ``PauliRot`` gates are correctly decomposed
        into canonical basis vectors as coefficients."""
        # generators of PauliRot
        paulirot_gens = -0.5j * pauli_basis_matrices(num_wires)
        # With many entries, each containing one basis element
        paulirot_coeffs = _one_parameter_paulirot_coeffs(paulirot_gens, num_wires)
        assert qml.math.allclose(paulirot_coeffs, np.eye(4**num_wires - 1))
        # With a "single entry" containing all basis elements
        paulirot_coeffs = _one_parameter_paulirot_coeffs([paulirot_gens], num_wires)
        assert qml.math.allclose(paulirot_coeffs, np.eye(4**num_wires - 1))


@pytest.mark.jax
class TestNonzeroCoeffsAndWords:
    """Test the utility function _nonzero_coeffs_and_words."""

    @pytest.mark.parametrize("num_wires", [1, 2, 3])
    def test_all_zero(self, num_wires):
        """Test that no coefficients and words are returned when all coefficients vanish."""
        dim = 4**num_wires - 1
        shapes = [(dim, 3), (dim,), (dim, 2, 5)]
        # Generate zero coefficients only
        coeffs = tuple(np.zeros(shape) for shape in shapes)
        new_coeffs, words = _nonzero_coeffs_and_words(coeffs, num_wires)
        # There should be as many coefficient sets as before, namely three, and they should
        # be empty.
        assert new_coeffs == [[], [], []]
        # There should be an empty list of Pauli words left.
        assert words == []

    @pytest.mark.parametrize("num_wires", [1, 2, 3])
    def test_separate_nonzero(self, num_wires, seed):
        """Test that a single coefficient in any of the coefficients is sufficient
        to keep the Pauli word in the filter."""
        # Create many coefficients, each greater or equal ``1`` at distinct places.
        rng = np.random.default_rng(seed)
        coeffs = tuple(rng.uniform(1, 2, size=(4**num_wires - 1, 4**num_wires - 1)))
        new_coeffs, words = _nonzero_coeffs_and_words(coeffs, num_wires)

        # The coefficients should not have changed and all words should be returned
        assert all(qml.math.allclose(nc, c) for nc, c in zip(new_coeffs, coeffs))
        assert len(words) == 4**num_wires - 1
        # Also check that the order of the words is consistent.
        assert all(w == exp for w, exp in zip(words, pauli_basis_strings(num_wires)))

    @pytest.mark.parametrize(
        "num_wires, remove_ids", [(1, [0]), (1, [2]), (2, [0, 3]), (2, [0, 1, 2, 3]), (2, [10])]
    )
    def test_single_zeros(self, num_wires, remove_ids):
        """Test that a removing single entries/Pauli words leads to the corresponding
        rows in the coefficients being skipped."""
        dim = 4**num_wires - 1
        # Set a few coefficients to zero
        coeffs = tuple(np.zeros(dim) if i in remove_ids else e for i, e in enumerate(np.eye(dim)))
        new_coeffs, words = _nonzero_coeffs_and_words(coeffs, num_wires)

        mask = np.ones(dim, bool)
        mask[remove_ids] = 0
        # Filter the coefficients and Pauli words for non-zero components
        exp_coeffs = tuple(c[mask] for c in coeffs)
        exp_words = [w for i, w in zip(mask, pauli_basis_strings(num_wires)) if i]
        assert all(qml.math.allclose(nc, c) for nc, c in zip(new_coeffs, exp_coeffs))
        assert len(words) == 4**num_wires - 1 - len(remove_ids)
        assert all(w == exp for w, exp in zip(words, exp_words))

        # Remove entries in np.eye(...) coefficients tuple altogether, effectively doing
        # the same as setting rows of coefficients to zero.
        coeffs = tuple(e for i, e in enumerate(np.eye(dim)) if i not in remove_ids)
        new_coeffs, words = _nonzero_coeffs_and_words(coeffs, num_wires)
        exp_coeffs = tuple(c[mask] for c in coeffs)
        assert all(qml.math.allclose(nc, c) for nc, c in zip(new_coeffs, exp_coeffs))
        assert len(words) == 4**num_wires - 1 - len(remove_ids)
        assert all(w == exp for w, exp in zip(words, exp_words))

    def test_atol(self):
        """Test that the precision keyword argument atol is used correctly."""
        atols = [1e-8, 1e-4, 1e-1]
        # With the following coefficients, different rows will be considered to vanish
        # for the different tolerances set above.
        coeffs = (np.array(atols), np.array(atols) * 2)
        for i, atol in enumerate(atols):
            new_coeffs, words = _nonzero_coeffs_and_words(coeffs, 1, atol=atol / 10)
            assert len(new_coeffs) == 2
            assert all(len(c) == 3 - i for c in new_coeffs)
            assert words == ["X", "Y", "Z"][i:]
        new_coeffs, words = _nonzero_coeffs_and_words(coeffs, 1, atol=1.0)
        assert new_coeffs == [[], []]
        assert words == []


all_ops_and_meas = [
    ([X(0)], []),
    (["evolve_op"], [qml.expval(Z(0))]),
    (["evolve_op", qml.RZ(0.3, "b"), "evolve_op", X(0)], []),
]

ops_meas_and_op_ids = [
    (all_ops_and_meas[0], 0),
    (all_ops_and_meas[0], 1),
    (all_ops_and_meas[1], 0),
    (all_ops_and_meas[1], 1),
    (all_ops_and_meas[2], 0),
    (all_ops_and_meas[2], 3),
    (all_ops_and_meas[2], 4),
]
all_ops = [[rot := qml.PauliRot(0.3, "IXZ", [0, 1, "a"])], [rot, qml.RY(0.3, 5), X(1)]]


@pytest.mark.jax
class TestInsertOp:
    """Test the utility _insert_op."""

    # pylint: disable=too-few-public-methods

    @pytest.mark.parametrize("ops_and_meas, op_idx", ops_meas_and_op_ids)
    @pytest.mark.parametrize("ops", all_ops)
    def test_output_properties(self, ops_and_meas, op_idx, ops):
        """Test that the input tape and inserted ops are taken into account correctly."""
        evolve_op = qml.evolve(qml.pulse.constant * Z("a"))([np.array(0.2)], 0.2)
        operations, measurements = ops_and_meas
        operations = [evolve_op if isinstance(op, str) else op for op in operations]
        tape = qml.tape.QuantumScript(operations, measurements)
        new_tapes = _insert_op(tape, ops, op_idx)
        assert isinstance(new_tapes, list) and len(new_tapes) == len(ops)
        for t, op in zip(new_tapes, ops):
            for o0, o1 in zip(t[:op_idx], tape[:op_idx]):
                qml.assert_equal(o0, o1)
            qml.assert_equal(t[op_idx], op)
            for o0, o1 in zip(t[op_idx + 1 :], tape[op_idx:]):
                qml.assert_equal(o0, o1)


@pytest.mark.jax
class TestGenerateTapesAndCoeffs:
    """Test the utility function _generate_tapes_and_coeffs."""

    atol = 1e-6
    T = 0.4

    def make_tape(self, all_H, all_params):
        """Make a tape with parametrized evolutions."""
        ops = [qml.evolve(H)(p, self.T) for H, p in zip(all_H, all_params)]
        return qml.tape.QuantumScript(ops, [qml.expval(Z(0))])

    def check_cache_equality(self, cache, expected):
        """Check that a cache equals an expected cache."""
        assert list(cache.keys()) == list(expected.keys())
        # "total_num_tapes" always is a key in the caches
        assert cache["total_num_tapes"] == expected["total_num_tapes"]
        for k, v in cache.items():
            if k == "total_num_tapes":
                # We already checked "total_num_tapes", so skip this entry
                continue
            expected_value = expected[k]
            # values of the cache should be three-tuples with format (int, int, tuple[tensor_like])
            assert isinstance(v, tuple) and len(v) == 3
            # first two entries (integers) are start and end pointers into the list of
            # tapes (or equivalently into the list of results later on in the gradient transfom)
            assert v[:2] == expected_value[:2]
            # last entry is a tuple of coefficients, so we iterate over the outer-most axis
            # (the tuple axis) and compare the tensors one after the other
            for _v, e in zip(v[2], expected_value[2]):
                assert qml.math.allclose(_v, e, atol=self.atol)

    def check_tapes_and_coeffs_equality(self, grad_tapes, tup, expected):
        """Check that generated tapes and coefficients equal the expectation."""
        import jax.numpy as jnp

        start, end, num_tapes, words, wires, old_tape, insert_idx, exp_coeffs = expected
        assert len(grad_tapes) == num_tapes
        for t_idx, word in enumerate(words):
            for sign, t in zip([1, -1], grad_tapes[2 * t_idx : 2 * (t_idx + 1)]):
                assert len(t.operations) == len(old_tape.operations) + 1
                expected_ops = copy.copy(old_tape.operations)
                expected_ops.insert(insert_idx, qml.PauliRot(sign * np.pi / 2, word, wires))
                for op, old_op in zip(t.operations, expected_ops):
                    qml.assert_equal(op, old_op)
        assert tup[:2] == (start, end)

        # Check coefficients
        coeffs = tup[2]
        assert isinstance(coeffs, list) and len(coeffs) == len(exp_coeffs)
        assert isinstance(coeffs[0], jnp.ndarray) and coeffs[0].shape == exp_coeffs[0].shape
        assert qml.math.allclose(coeffs[0], exp_coeffs[0], atol=self.atol)

    def test_raises_non_pulse_op(self):
        """Test that an error is raised for an operation that is not a pulse."""
        tape = qml.tape.QuantumScript([qml.RX(0.4, 0)], [qml.expval(qml.PauliZ(0))])
        cache = {"total_num_tapes": 0}
        with pytest.raises(ValueError, match="pulse_odegen does not support differentiating"):
            _generate_tapes_and_coeffs(tape, 0, 1e-6, cache)

    @pytest.mark.parametrize("add_constant", [False, True])
    def test_single_op_single_term(self, add_constant):
        """Test the tape generation for a single parameter and Hamiltonian term in a tape."""
        import jax.numpy as jnp

        H = qml.pulse.constant * X(0)
        if add_constant:
            H = 0.4 * Z(1) + H
        params = [jnp.array(0.4)]
        tape = self.make_tape([H], [params])
        cache = {"total_num_tapes": -5}
        grad_tapes, (start, end, coeffs), cache = _generate_tapes_and_coeffs(
            tape, 0, self.atol, cache
        )

        # The Hamiltonian above has a single parametrized term, generated by PauliX(0).
        # We therefore expect 2 tapes both with an inserted PauliRot about PauliX(0), but
        # with differing angles. This will yield a single partial derivative with respect to
        # PauliX(0). According to the single scalar parameter of H and the single
        # derivative, we expect a single coefficient which is a scalar and has value 2 * T.
        words = ["IX"] if add_constant else ["X"]
        wires = [1, 0] if add_constant else 0
        # There is a factor of 2 for the exp(i*(pauliword)) to PauliRot conversion
        exp_coeffs = [jnp.array(2 * self.T)]
        expected = (-5, -3, 2, words, wires, tape, 0, exp_coeffs)
        exp_cache = {"total_num_tapes": -3, 0: (-5, -3, [coeffs])}

        self.check_tapes_and_coeffs_equality(grad_tapes, (start, end, coeffs), expected)
        self.check_cache_equality(cache, exp_cache)

    @pytest.mark.parametrize("same_terms", [False, True])
    def test_single_op_multi_term(self, same_terms):
        """Test the tape generation for multiple parameters of a single Hamiltonian in a tape."""
        import jax
        import jax.numpy as jnp

        H = qml.pulse.constant * X(0) + jnp.polyval * (X(0) if same_terms else Z(1))
        params = [jnp.array(0.4), jnp.array([0.3, 0.2, 0.1])]
        tape = self.make_tape([H], [params])
        cache = {"total_num_tapes": 10}
        # The Hamiltonian above has two parametrized terms, generated by PauliX(0) and
        # by PauliX(0) or PauliZ(1), depending on whether same_terms is True.
        # For same_terms=True, we expect 2 tapes both with an inserted PauliRot about PauliX(0),
        # but with differing angles. For same_terms=False, there are two more tapes, with
        # inserted rotations about PauliZ(1). This will yield one (two) partial derivative(s)
        # with respect to PauliX(0) (and PauliZ(1)). According to the parameter shapes
        # () and (3,) of H and the one-entry (two-entry) derivative, we expect coefficients
        # with shapes () and (3,) (with shapes (2,) and (2, 3)).
        # First trainable parameter - everything is computed
        grad_tapes, (start, end, coeffs), cache = _generate_tapes_and_coeffs(
            tape, 0, self.atol, cache
        )

        num_tapes = 2 if same_terms else 4
        words = ["X"] if same_terms else ["IZ", "XI"]
        wires = 0 if same_terms else [0, 1]
        if same_terms:
            exp_coeffs = [
                [jnp.array(2 * self.T)],
                [2 * jax.grad(integral_of_polyval)(params[1], self.T)],
            ]
        else:
            exp_coeffs = [
                [jnp.array(0), jnp.array(2 * self.T)],
                [2 * jax.grad(integral_of_polyval)(params[1], self.T), jnp.zeros(3)],
            ]
        expected = (10, 10 + num_tapes, num_tapes, words, wires, tape, 0, exp_coeffs[0])
        exp_cache = {"total_num_tapes": 10 + num_tapes, 0: (10, 10 + num_tapes, exp_coeffs)}

        self.check_tapes_and_coeffs_equality(grad_tapes, (start, end, coeffs), expected)
        self.check_cache_equality(cache, exp_cache)

        # Second trainable parameter - everything is cached
        grad_tapes, (start, end, coeffs), cache = _generate_tapes_and_coeffs(
            tape, 1, self.atol, cache
        )
        expected = (10, 10 + num_tapes, 0, [], None, tape, None, exp_coeffs[1])

        self.check_tapes_and_coeffs_equality(grad_tapes, (start, end, coeffs), expected)
        self.check_cache_equality(cache, exp_cache)

    def test_multi_op_multi_term(self):
        """Test the tape generation for multiple parameters of multiple Hamiltonians in a tape."""
        import jax
        import jax.numpy as jnp

        def _sin(p, t):
            """Compute the function a * sin(b * t + c)."""
            return p[0] * jnp.sin(p[1] * t + p[2])

        def _int_sin(p, t):
            """Compute the (time) integral of the function a * sin(b * t + c)."""
            if qml.math.ndim(t) == 0:
                t = [0, t]
            args = (p[1] * t[1] + p[2], p[1] * t[0] + p[2])
            return -p[0] / p[1] * jnp.cos(args[0]) + p[0] / p[1] * jnp.cos(args[1])

        H0 = qml.pulse.constant * X(0) + jnp.polyval * Z(1)
        H1 = jnp.polyval * (Y(1) @ Y(0)) + _sin * (Z(0) @ Z(1))
        params0 = [jnp.array(0.4), jnp.array([0.3, 0.2, 0.1])]
        params1 = [jnp.array([0.3, 0.2, 0.1, 1.2]), jnp.array([0.4, 1.2, -0.9])]

        exp_coeffs0 = [
            [jnp.array(0), jnp.array(2 * self.T)],
            [2 * jax.grad(integral_of_polyval)(params0[1], self.T), jnp.zeros(3)],
        ]
        exp_coeffs1 = [
            [2 * jax.grad(integral_of_polyval)(params1[0], self.T), jnp.zeros(4)],
            [jnp.zeros(3), 2 * jax.grad(_int_sin)(params1[1], self.T)],
        ]

        tape = self.make_tape([H0, H1], [params0, params1])
        cache = {"total_num_tapes": 0}
        # First trainable parameter - everything for H0 is computed
        grad_tapes, (start, end, coeffs), cache = _generate_tapes_and_coeffs(
            tape, 0, self.atol, cache
        )

        words = ["IZ", "XI"]
        wires = [0, 1]
        expected = (0, 4, 4, words, wires, tape, 0, exp_coeffs0[0])
        exp_cache = {"total_num_tapes": 4, 0: (0, 4, exp_coeffs0)}

        self.check_tapes_and_coeffs_equality(grad_tapes, (start, end, coeffs), expected)
        self.check_cache_equality(cache, exp_cache)

        # Second trainable parameter - everything for H0 is cached
        grad_tapes, (start, end, coeffs), cache = _generate_tapes_and_coeffs(
            tape, 1, self.atol, cache
        )

        expected = (0, 4, 0, [], None, tape, None, exp_coeffs0[1])

        self.check_tapes_and_coeffs_equality(grad_tapes, (start, end, coeffs), expected)
        self.check_cache_equality(cache, exp_cache)

        # Third trainable parameter - everything for H1 is computed
        grad_tapes, (start, end, coeffs), cache = _generate_tapes_and_coeffs(
            tape, 2, self.atol, cache
        )

        words = ["YY", "ZZ"]
        wires = [1, 0]
        expected = (4, 8, 4, words, wires, tape, 1, exp_coeffs1[0])
        exp_cache["total_num_tapes"] = 8
        exp_cache[1] = (4, 8, exp_coeffs1)

        self.check_tapes_and_coeffs_equality(grad_tapes, (start, end, coeffs), expected)
        self.check_cache_equality(cache, exp_cache)

        # Fourth trainable parameter - everything for H1 is cached
        grad_tapes, (start, end, coeffs), cache = _generate_tapes_and_coeffs(
            tape, 3, self.atol, cache
        )

        expected = (4, 8, 0, [], None, tape, 1, exp_coeffs1[1])

        self.check_tapes_and_coeffs_equality(grad_tapes, (start, end, coeffs), expected)
        self.check_cache_equality(cache, exp_cache)


class TestParshiftAndContract:
    """Test the utility function _parshift_and_contract."""

    @pytest.mark.parametrize("num_params", [1, 3])
    @pytest.mark.parametrize("num_out", [1, 4])
    def test_single_measure_no_shot_vector(self, num_params, num_out):
        """Test _parshift_and_contract with a single measurement without shot vector."""
        values = np.arange(1, num_params + 1)
        # Emulate results to be [1.2*value, -0.42*value], leading to the expected PSR
        # derivative 0.81*value
        results = [np.array(pref * v) for v in values for pref in [1.2, -0.42]]
        coeffs = [np.random.random(num_out) for _ in range(num_params)]
        output = _parshift_and_contract(
            results, coeffs, single_measure=True, single_shot_entry=True
        )
        assert isinstance(output, np.ndarray)
        assert output.shape == (num_out,)
        expected = 0.81 * values @ np.stack(coeffs)
        assert qml.math.allclose(output, expected)

    @pytest.mark.parametrize("num_params", [1, 3])
    @pytest.mark.parametrize("num_out", [1, 4])
    @pytest.mark.parametrize("len_shot_vector", [1, 2, 3])
    def test_single_measure_with_shot_vector(self, num_params, num_out, len_shot_vector):
        """Test _parshift_and_contract with a single measurement with shot vector."""
        values = np.arange(1, num_params + 1)
        shot_factors = np.random.random(len_shot_vector)
        # Emulate results to be [1.2*value, -0.42*value], leading to the expected PSR
        # derivative 0.81*value
        results = [shot_factors * (pref * v) for v in values for pref in [1.2, -0.42]]
        coeffs = [np.random.random(num_out) for _ in range(num_params)]
        output = _parshift_and_contract(
            results, coeffs, single_measure=True, single_shot_entry=False
        )
        assert isinstance(output, tuple)
        assert len(output) == len_shot_vector
        assert all(isinstance(x, np.ndarray) for x in output)
        assert all(x.shape == (num_out,) for x in output)
        expected = np.outer(shot_factors, 0.81 * values @ np.stack(coeffs))
        assert qml.math.allclose(output, expected)

    @pytest.mark.parametrize("num_params", [1, 3])
    @pytest.mark.parametrize("num_out", [1, 4])
    @pytest.mark.parametrize("num_measurements", [2, 3])
    def test_multi_measure_no_shot_vector(self, num_params, num_out, num_measurements):
        """Test _parshift_and_contract with multiple measurements without shot vector."""
        values = np.arange(1, num_params + 1)
        meas_factors = np.random.random(num_measurements)
        # Emulate results to be [1.2*value, -0.42*value], leading to the expected PSR
        # derivative 0.81*value
        results = [meas_factors * (pref * v) for v in values for pref in [1.2, -0.42]]
        coeffs = [np.random.random(num_out) for _ in range(num_params)]
        output = _parshift_and_contract(
            results, coeffs, single_measure=False, single_shot_entry=True
        )
        # Note that these checks are equal to those for single_measure and single_shot_entry
        assert isinstance(output, tuple)
        assert len(output) == num_measurements
        assert all(isinstance(x, np.ndarray) for x in output)
        assert all(x.shape == (num_out,) for x in output)
        expected = np.outer(meas_factors, 0.81 * values @ np.stack(coeffs))
        assert qml.math.allclose(output, expected)

    @pytest.mark.parametrize("num_params", [1, 3])
    @pytest.mark.parametrize("num_out", [1, 4])
    @pytest.mark.parametrize("num_measurements", [2, 3])
    @pytest.mark.parametrize("len_shot_vector", [1, 2, 3])
    def test_multi_measure_with_shot_vector(
        self, num_params, num_out, num_measurements, len_shot_vector
    ):
        """Test _parshift_and_contract with multiple measurements without shot vector."""
        values = np.arange(1, num_params + 1)
        meas_factors = np.random.random(num_measurements)
        shot_factors = np.random.random(len_shot_vector)
        # Emulate results to be [1.2*value, -0.42*value], leading to the expected PSR
        # derivative 0.81*value
        factors = np.outer(shot_factors, meas_factors)
        results = [factors * (pref * v) for v in values for pref in [1.2, -0.42]]
        coeffs = [np.random.random(num_out) for _ in range(num_params)]
        output = _parshift_and_contract(
            results, coeffs, single_measure=False, single_shot_entry=False
        )
        # Note that these checks are equal to those for single_measure and single_shot_entry
        assert isinstance(output, tuple)
        assert len(output) == len_shot_vector
        for x in output:
            assert isinstance(x, tuple)
            assert len(x) == num_measurements
            assert all(isinstance(y, np.ndarray) for y in x)
            assert all(y.shape == (num_out,) for y in x)
            expected = np.tensordot(factors, 0.81 * values @ np.stack(coeffs), axes=0)
            assert qml.math.allclose(output, expected)


@pytest.mark.jax
class TestPulseOdegenEdgeCases:
    """Test that differentiating edge case tapes with ``pulse_odegen`` works."""

    def test_raises_with_state_return(self):
        """Make sure an error is raised for a tape that returns a state."""
        tape = qml.tape.QuantumScript(measurements=[qml.state()])
        _match = "circuits that return the state with the pulse generator parameter-shift gradient"
        with pytest.raises(ValueError, match=_match):
            pulse_odegen(tape)

    def test_raises_with_variance_return(self):
        """Make sure an error is raised for a tape that returns a variance."""
        tape = qml.tape.QuantumScript(measurements=[qml.var(X(0))])
        _match = "gradient of variances with the pulse generator parameter-shift gradient"
        with pytest.raises(ValueError, match=_match):
            pulse_odegen(tape)

    def test_raises_with_invalid_op(self):
        """Test that an error is raised when calling ``pulse_odegen`` on a non-pulse op."""
        tape = qml.tape.QuantumScript([qml.RX(0.4, 0)], [qml.expval(Z(0))])
        _match = "pulse_odegen does not support differentiating parameters of other"
        with pytest.raises(ValueError, match=_match):
            pulse_odegen(tape)

    def test_trainable_batched_tape_raises(self):
        """Test that an error is raised for a broadcasted/batched tape if the broadcasted
        parameter is differentiated."""
        tape = qml.tape.QuantumScript([qml.RX([0.4, 0.2], 0)], [qml.expval(qml.PauliZ(0))])
        _match = r"Computing the gradient of broadcasted tapes .* using the pulse generator"
        with pytest.raises(NotImplementedError, match=_match):
            pulse_odegen(tape)

    def test_nontrainable_batched_tape(self):
        """Test that no error is raised for a broadcasted/batched tape if the broadcasted
        parameter is not differentiated, and that the results correspond to the stacked
        results of the single-tape derivatives."""
        import jax.numpy as jnp

        dev = qml.device("default.qubit")
        x = [0.4, 0.2]
        params = [jnp.array(0.14)]
        ham_single_q_const = qml.pulse.constant * qml.PauliY(0)
        op = qml.evolve(ham_single_q_const)(params, 0.1)
        tape = qml.tape.QuantumScript(
            [qml.RX(x, 0), op], [qml.expval(qml.PauliZ(0))], trainable_params=[1]
        )
        batched_tapes, batched_fn = pulse_odegen(tape, argnum=0)
        batched_grad = batched_fn(dev.execute(batched_tapes))
        separate_tapes = [
            qml.tape.QuantumScript(
                [qml.RX(_x, 0), op], [qml.expval(qml.PauliZ(0))], trainable_params=[1]
            )
            for _x in x
        ]
        separate_tapes_and_fns = [pulse_odegen(t, argnum=0) for t in separate_tapes]
        separate_grad = [_fn(dev.execute(_tapes)) for _tapes, _fn in separate_tapes_and_fns]
        assert np.allclose(batched_grad, separate_grad)

    def test_no_trainable_params_tape(self):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters"""
        dev = qml.device("default.qubit", wires=2)

        weights = [0.1, 0.2]
        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            qml.expval(Z(0) @ Z(1))

        tape = qml.tape.QuantumScript.from_queue(q)
        # TODO: remove once #2155 is resolved
        tape.trainable_params = []

        with pytest.warns(UserWarning, match="gradient of a tape with no trainable parameters"):
            g_tapes, post_processing = pulse_odegen(tape)
        res = post_processing(qml.execute(g_tapes, dev, None))

        assert g_tapes == []
        assert isinstance(res, np.ndarray)
        assert res.shape == (0,)

    def test_no_trainable_params_multiple_return_tape(self):
        """Test that the correct ouput and warning is generated in the absence of any trainable
        parameters with multiple returns."""
        dev = qml.device("default.qubit", wires=2)

        weights = [0.1, 0.2]
        with qml.queuing.AnnotatedQueue() as q:
            qml.RX(weights[0], wires=0)
            qml.RY(weights[1], wires=0)
            qml.expval(Z(0) @ Z(1))
            qml.probs(wires=[0, 1])

        tape = qml.tape.QuantumScript.from_queue(q)
        tape.trainable_params = []

        with pytest.warns(UserWarning, match="gradient of a tape with no trainable parameters"):
            _tapes, fn = pulse_odegen(tape)
        res = fn(dev.execute(_tapes))

        assert _tapes == []
        assert isinstance(res, tuple)
        for r in res:
            assert isinstance(r, np.ndarray)
            assert r.shape == (0,)

    def test_all_zero_diff_methods_tape(self):
        """Test that the transform works correctly when the diff method for every parameter is
        identified to be 0, and that no tapes were generated."""
        dev = qml.device("default.qubit", wires=3)

        params = pnp.array([0.5, 0.5, 0.5], requires_grad=True)

        with qml.queuing.AnnotatedQueue() as q:
            qml.Rot(*params, wires=0)  # does not impact the measurement -> zero grad
            qml.probs([2, 3])

        tape = qml.tape.QuantumScript.from_queue(q)

        res_pulse_gen, _tapes = grad_fn(tape, dev)

        assert _tapes == []

        assert isinstance(res_pulse_gen, tuple)

        assert len(res_pulse_gen) == 3

        assert isinstance(res_pulse_gen[0], np.ndarray)
        assert res_pulse_gen[0].shape == (4,)
        assert np.allclose(res_pulse_gen[0], 0)

        assert isinstance(res_pulse_gen[1], np.ndarray)
        assert res_pulse_gen[1].shape == (4,)
        assert np.allclose(res_pulse_gen[1], 0)

        assert isinstance(res_pulse_gen[2], np.ndarray)
        assert res_pulse_gen[2].shape == (4,)
        assert np.allclose(res_pulse_gen[2], 0)

    def test_all_zero_diff_methods_multiple_returns_tape(self):
        """Test that the transform works correctly when the diff method for every parameter is
        identified to be 0, and that no tapes were generated."""

        dev = qml.device("default.qubit", wires=3)

        params = pnp.array([0.5, 0.5, 0.5], requires_grad=True)

        with qml.queuing.AnnotatedQueue() as q:
            qml.Rot(*params, wires=0)  # does not impact the measurements -> zero grad
            qml.expval(Z(wires=2))
            qml.probs([2, 3])

        tape = qml.tape.QuantumScript.from_queue(q)

        res_pulse_gen, _tapes = grad_fn(tape, dev)

        assert _tapes == []

        assert isinstance(res_pulse_gen, tuple)

        assert len(res_pulse_gen) == 2

        # First elem
        assert len(res_pulse_gen[0]) == 3

        assert isinstance(res_pulse_gen[0][0], np.ndarray)
        assert res_pulse_gen[0][0].shape == ()
        assert np.allclose(res_pulse_gen[0][0], 0)

        assert isinstance(res_pulse_gen[0][1], np.ndarray)
        assert res_pulse_gen[0][1].shape == ()
        assert np.allclose(res_pulse_gen[0][1], 0)

        assert isinstance(res_pulse_gen[0][2], np.ndarray)
        assert res_pulse_gen[0][2].shape == ()
        assert np.allclose(res_pulse_gen[0][2], 0)

        # Second elem
        assert len(res_pulse_gen[0]) == 3

        assert isinstance(res_pulse_gen[1][0], np.ndarray)
        assert res_pulse_gen[1][0].shape == (4,)
        assert np.allclose(res_pulse_gen[1][0], 0)

        assert isinstance(res_pulse_gen[1][1], np.ndarray)
        assert res_pulse_gen[1][1].shape == (4,)
        assert np.allclose(res_pulse_gen[1][1], 0)

        assert isinstance(res_pulse_gen[1][2], np.ndarray)
        assert res_pulse_gen[1][2].shape == (4,)
        assert np.allclose(res_pulse_gen[1][2], 0)


@pytest.mark.jax
class TestPulseOdegenTape:
    """Test that differentiating tapes with ``pulse_odegen`` works."""

    @pytest.mark.parametrize("shots, tol", [(None, 1e-7), (1000, 0.05), ([1000, 100], 0.05)])
    def test_single_pulse_single_term(self, shots, tol, seed):
        """Test that a single pulse with a single Hamiltonian term is
        differentiated correctly."""
        import jax
        import jax.numpy as jnp

        prng_key = jax.random.PRNGKey(seed)
        dev = qml.device("default.qubit", wires=1, seed=prng_key)

        H = jnp.polyval * X(0)
        x = jnp.array([0.4, 0.2, 0.1])
        t = [0.2, 0.3]
        op = qml.evolve(H)([x], t=t)
        tape = qml.tape.QuantumScript([op], [qml.expval(Z(0))], shots=shots)

        theta = integral_of_polyval(x, t)

        _tapes, fn = pulse_odegen(tape)
        assert len(_tapes) == 2

        val = qml.execute([tape], dev)
        assert qml.math.allclose(val, jnp.cos(2 * theta), atol=tol)

        grad = fn(qml.execute(_tapes, dev))
        par_jac = jax.jacobian(integral_of_polyval)(x, t)
        exp_grad = -2 * par_jac * jnp.sin(2 * theta)
        if isinstance(shots, list):
            assert isinstance(grad, tuple) and len(grad) == len(shots)
            assert all(isinstance(g, jnp.ndarray) and g.shape == x.shape for g in grad)
            assert qml.math.allclose(grad, exp_grad, atol=tol)
        else:
            assert isinstance(grad, jnp.ndarray) and grad.shape == x.shape
            assert qml.math.allclose(grad, exp_grad, atol=tol)

    @pytest.mark.slow
    @pytest.mark.parametrize("shots, tol", [(None, 1e-7), ([1000, 100], 0.05)])
    def test_single_pulse_multi_term(self, shots, tol, seed):
        """Test that a single pulse with multiple Hamiltonian terms is
        differentiated correctly."""
        import jax
        import jax.numpy as jnp

        prng_key = jax.random.PRNGKey(seed)
        dev = qml.device("default.qubit", wires=1, seed=prng_key)

        H = 0.1 * Z(0) + jnp.polyval * X(0) + qml.pulse.constant * Y(0)
        x = jnp.array([0.4, 0.2, 0.1])
        y = jnp.array(-0.5)
        t = [0.2, 0.3]

        @qml.qnode(dev, interface="jax")
        def circuit(par):
            qml.evolve(H)(par, t=t)
            return qml.expval(Z(0))

        # TODO: remove once #2155 is resolved
        tape_with_shots = qml.workflow.construct_tape(circuit)([x, y])
        tape_with_shots.trainable_params = [0, 1]
        tape_with_shots._shots = qml.measurements.Shots(shots)  # pylint:disable=protected-access
        _tapes, fn = pulse_odegen(tape_with_shots, argnum=[0, 1])
        assert len(_tapes) == 6  # dim(DLA)=3, two shifts per basis element

        grad = fn(qml.execute(_tapes, dev))
        exp_grad = jax.jacobian(circuit)([x, y])
        if isinstance(shots, list):
            assert isinstance(grad, tuple) and len(grad) == len(shots)
            for _grad in grad:
                assert isinstance(_grad, tuple) and len(_grad) == 2
                assert all(qml.math.allclose(g, e, atol=tol) for g, e in zip(_grad, exp_grad))
        else:
            assert isinstance(grad, tuple) and len(grad) == 2
            assert all(qml.math.allclose(g, e, atol=tol) for g, e in zip(grad, exp_grad))

    @pytest.mark.parametrize("argnum", (0, [0], 1, [1]))
    def test_single_pulse_multi_term_argnum(self, argnum):
        """Test that a single pulse with multiple Hamiltonian terms is
        differentiated correctly when setting ``argnum``."""
        import jax
        import jax.numpy as jnp

        dev = qml.device("default.qubit", wires=1)

        H = jnp.polyval * X(0) + qml.pulse.constant * X(0)
        x = jnp.array([0.4, 0.2, 0.1])
        y = jnp.array(0.6)
        t = [0.2, 0.3]
        op = qml.evolve(H)([x, y], t=t)
        tape = qml.tape.QuantumScript([op], [qml.expval(Z(0))])

        theta = integral_of_polyval(x, t) + y * (t[1] - t[0])

        # Argnum=[0] or 0
        par_jac_0 = jax.jacobian(integral_of_polyval)(x, t)
        exp_grad_0 = -2 * par_jac_0 * jnp.sin(2 * theta)
        # Argnum=[1] or 1
        par_jac_1 = t[1] - t[0]
        exp_grad_1 = -2 * par_jac_1 * jnp.sin(2 * theta)
        exp_grads = [exp_grad_0, exp_grad_1]

        _tapes, fn = pulse_odegen(tape, argnum=argnum)
        assert len(_tapes) == 2

        val = qml.execute([tape], dev)
        assert qml.math.allclose(val, jnp.cos(2 * theta))

        grad = fn(qml.execute(_tapes, dev))
        assert isinstance(grad, tuple) and len(grad) == 2
        assert isinstance(grad[0], jnp.ndarray) and grad[0].shape == x.shape
        assert isinstance(grad[1], jnp.ndarray) and grad[1].shape == y.shape
        argnum_int = argnum if isinstance(argnum, int) else argnum[0]
        assert qml.math.allclose(grad[argnum_int], exp_grads[argnum_int])
        assert qml.math.allclose(grad[1 - argnum_int], 0.0)

    @pytest.mark.slow
    @pytest.mark.parametrize("shots, tol", [(None, 1e-7), ([1000, 100], 0.05)])
    def test_multi_pulse(self, shots, tol, seed):
        """Test that a single pulse with multiple Hamiltonian terms is
        differentiated correctly."""
        import jax
        import jax.numpy as jnp

        prng_key = jax.random.PRNGKey(seed)
        dev = qml.device("default.qubit", wires=1, seed=prng_key)

        H0 = 0.1 * Z(0) + jnp.polyval * X(0)
        H1 = 0.2 * Y(0) + qml.pulse.constant * Y(0) + jnp.polyval * Z(0)
        x = jnp.array([0.4, 0.2, 0.1])
        y = jnp.array(0.5)
        z = jnp.array([-0.3, 0.6])
        t = [0.2, 0.3]

        @qml.qnode(dev, interface="jax")
        def circuit(par):
            qml.evolve(H0)(par[:1], t=t)
            qml.evolve(H1)(par[1:], t=t)
            return qml.expval(Z(0))

        # TODO: remove once #2155 is resolved
        tape_with_shots = qml.workflow.construct_tape(circuit)([x, y, z])
        tape_with_shots.trainable_params = [0, 1, 2]
        tape_with_shots._shots = qml.measurements.Shots(shots)  # pylint:disable=protected-access
        _tapes, fn = pulse_odegen(tape_with_shots, argnum=[0, 1, 2])
        assert len(_tapes) == 12  # two pulses, dim(DLA)=3, two shifts per basis element

        grad = fn(qml.execute(_tapes, dev))
        exp_grad = jax.jacobian(circuit)([x, y, z])
        if isinstance(shots, list):
            assert isinstance(grad, tuple) and len(grad) == len(shots)
            for _grad in grad:
                assert isinstance(_grad, tuple) and len(_grad) == 3
                assert all(qml.math.allclose(g, e, atol=tol) for g, e in zip(_grad, exp_grad))
        else:
            assert isinstance(grad, tuple) and len(grad) == 3
            assert all(qml.math.allclose(g, e, atol=tol) for g, e in zip(grad, exp_grad))


@pytest.mark.jax
class TestPulseOdegenQNode:
    """Test that pulse_odegen integrates correctly with QNodes."""

    def test_raises_for_application_to_qnodes(self):
        """Test that an error is raised when applying ``stoch_pulse_grad``
        to a QNode directly."""

        dev = qml.device("default.qubit", wires=1)
        ham_single_q_const = qml.pulse.constant * qml.PauliY(0)

        @qml.qnode(dev, interface="jax")
        def circuit(params):
            qml.evolve(ham_single_q_const)([params], 0.2)
            return qml.expval(qml.PauliZ(0))

        _match = "pulse generator parameter-shift gradient transform to a QNode directly"
        with pytest.raises(NotImplementedError, match=_match):
            pulse_odegen(circuit)

    # TODO: include the following tests when #4225 is resolved.
    @pytest.mark.skip("Applying this gradient transform to QNodes directly is not supported.")
    def test_qnode_expval_single_par(self):
        """Test that a simple qnode that returns an expectation value
        can be differentiated with pulse_odegen."""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        dev = qml.device("default.qubit", wires=1)
        T = 0.2
        ham_single_q_const = qml.pulse.constant * Y(0)

        @qml.qnode(dev, interface="jax")
        def circuit(params):
            qml.evolve(ham_single_q_const)([params], T)
            return qml.expval(Z(0))

        params = jnp.array(0.4)
        with qml.Tracker(dev) as tracker:
            grad = pulse_odegen(circuit)(params)

        p = params * T
        exp_grad = -2 * jnp.sin(2 * p) * T
        assert jnp.allclose(grad, exp_grad)
        assert tracker.totals["executions"] == 2  # two shifted tapes

    @pytest.mark.skip("Applying this gradient transform to QNodes directly is not supported.")
    def test_qnode_expval_probs_single_par(self):
        """Test that a simple qnode that returns an expectation value
        can be differentiated with pulse_odegen."""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        dev = qml.device("default.qubit", wires=1)
        T = 0.2
        ham_single_q_const = jnp.polyval * Y(0)

        @qml.qnode(dev, interface="jax")
        def circuit(params):
            qml.evolve(ham_single_q_const)([params], T)
            return qml.probs(wires=0), qml.expval(Z(0))

        params = jnp.array([0.4, 0.9, 1.2])
        with qml.Tracker(dev) as tracker:
            jac = pulse_odegen(circuit)(params)

        assert tracker.totals["executions"] == 2  # two shifted tapes
        circuit(params)
        p = integral_of_polyval(params, T)
        p_jac = jax.jacobian(integral_of_polyval)(params, T)
        exp_jac = (
            jnp.outer(jnp.array([-1, 1]), jnp.sin(2 * p) * p_jac),
            -2 * jnp.sin(2 * p) * p_jac,
        )
        for j, e in zip(jac, exp_jac):
            assert qml.math.allclose(j, e)

    @pytest.mark.skip("Applying this gradient transform to QNodes directly is not supported.")
    def test_qnode_probs_expval_multi_par(self):
        """Test that a simple qnode that returns probabilities
        can be differentiated with pulse_odegen."""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        dev = qml.device("default.qubit", wires=1)
        T = 0.2
        ham_single_q_const = jnp.polyval * Y(0) + qml.pulse.constant * Y(0)

        @qml.qnode(dev, interface="jax")
        def circuit(params, c):
            qml.evolve(ham_single_q_const)([params, c], T)
            return qml.probs(wires=0), qml.expval(Z(0))

        params = jnp.array([0.4, 0.2, 0.1])
        c = jnp.array(0.9)
        with qml.Tracker(dev) as tracker:
            jac = pulse_odegen(circuit, argnums=[0, 1])(params, c)

        assert tracker.totals["executions"] == 2  # two shifted tapes
        p0 = integral_of_polyval(params, T)
        p0_jac = jax.jacobian(integral_of_polyval)(params, T)
        p1 = c * T
        p1_jac = T
        p = p0 + p1
        exp_jac = (
            (
                jnp.outer(jnp.array([-1, 1]), jnp.sin(2 * p) * p0_jac),
                jnp.outer(jnp.array([-1, 1]), jnp.sin(2 * p) * p1_jac),
            ),
            (-2 * jnp.sin(2 * p) * p0_jac, -2 * jnp.sin(2 * p) * p1_jac),
        )
        for j, e in zip(jac, exp_jac):
            for _j, _e in zip(j, e):
                assert qml.math.allclose(_j, _e)


@pytest.mark.jax
class TestPulseOdegenIntegration:
    """Test that pulse_odegen integrates correctly with QNodes."""

    def test_simple_qnode_expval(self):
        """Test that a simple qnode that returns an expectation value
        can be differentiated with pulse_odegen."""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        dev = qml.device("default.qubit", wires=1)
        T = 0.2
        ham_single_q_const = qml.pulse.constant * Y(0)

        @qml.qnode(dev, interface="jax", diff_method=pulse_odegen)
        def circuit(params):
            qml.evolve(ham_single_q_const)(params, T)
            return qml.expval(Z(0))

        params = [jnp.array(0.4)]
        with qml.Tracker(dev) as tracker:
            grad = jax.jacobian(circuit)(params)
        p = params[0] * T
        exp_grad = -2 * jnp.sin(2 * p) * T
        assert qml.math.allclose(grad, exp_grad)
        assert tracker.totals["executions"] == 1 + 2  # one forward pass, two shifted tapes

    def test_simple_qnode_expval_two_evolves(self):
        """Test that a simple qnode that returns an expectation value
        can be differentiated with pulse_odegen."""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        dev = qml.device("default.qubit", wires=1)
        T_x = 0.1
        T_y = 0.2
        ham_x = qml.pulse.constant * X(0)
        ham_y = qml.pulse.constant * X(0)

        @qml.qnode(dev, interface="jax", diff_method=pulse_odegen)
        def circuit(params):
            qml.evolve(ham_x)(params[0], T_x)
            qml.evolve(ham_y)(params[1], T_y)
            return qml.expval(Z(0))

        params = [[jnp.array(0.4)], [jnp.array(-0.1)]]
        grad = jax.jacobian(circuit)(params)
        p_x = params[0][0] * T_x
        p_y = params[1][0] * T_y
        exp_grad = [[-2 * jnp.sin(2 * (p_x + p_y)) * T_x], [-2 * jnp.sin(2 * (p_x + p_y)) * T_y]]
        assert qml.math.allclose(grad, exp_grad)

    def test_simple_qnode_probs(self):
        """Test that a simple qnode that returns probabilities
        can be differentiated with pulse_odegen."""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        dev = qml.device("default.qubit", wires=1)
        T = 0.2
        ham_single_q_const = qml.pulse.constant * Y(0)

        @qml.qnode(dev, interface="jax", diff_method=pulse_odegen)
        def circuit(params):
            qml.evolve(ham_single_q_const)(params, T)
            return qml.probs(wires=0)

        params = [jnp.array(0.4)]
        jac = jax.jacobian(circuit)(params)
        p = params[0] * T
        exp_jac = jnp.array([-1, 1]) * jnp.sin(2 * p) * T
        assert qml.math.allclose(jac, exp_jac)

    def test_simple_qnode_probs_expval(self):
        """Test that a simple qnode that returns probabilities
        can be differentiated with pulse_odegen."""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        dev = qml.device("default.qubit", wires=1)
        T = 0.2
        ham_single_q_const = jnp.polyval * Y(0)

        @qml.qnode(dev, interface="jax", diff_method=pulse_odegen)
        def circuit(params):
            qml.evolve(ham_single_q_const)(params, T)
            return qml.probs(wires=0), qml.expval(Z(0))

        params = [jnp.array([0.4, 0.2, 0.1])]
        jac = jax.jacobian(circuit)(params)
        p = integral_of_polyval(params[0], T)
        p_jac = jax.jacobian(integral_of_polyval)(params[0], T)
        exp_jac = (
            jnp.outer(jnp.array([-1, 1]), jnp.sin(2 * p) * p_jac),
            -2 * jnp.sin(2 * p) * p_jac,
        )
        for j, e in zip(jac, exp_jac):
            assert qml.math.allclose(j[0], e)

    @pytest.mark.xfail
    @pytest.mark.parametrize("time_interface", ["python", "numpy", "jax"])
    def test_simple_qnode_jit(self, time_interface):
        """Test that a simple qnode can be differentiated with pulse_odegen."""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        dev = qml.device("default.qubit", wires=1)
        T = {"python": 0.2, "numpy": np.array(0.2), "jax": jnp.array(0.2)}[time_interface]
        ham_single_q_const = qml.pulse.constant * Y(0)

        @qml.qnode(dev, interface="jax", diff_method=pulse_odegen)
        def circuit(params, T=None):
            qml.evolve(ham_single_q_const)(params, T)
            return qml.expval(Z(0))

        params = [jnp.array(0.4)]
        p = params[0] * T
        exp_grad = -2 * jnp.sin(2 * p) * T
        jit_grad = jax.jit(jax.grad(circuit))(params, T=T)
        assert qml.math.isclose(jit_grad, exp_grad)

    @pytest.mark.slow
    def test_advanced_qnode(self):
        """Test that an advanced qnode can be differentiated with pulse_odegen."""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)

        params = [jnp.array(0.21), jnp.array(-0.171), jnp.array([0.05, 0.03, -0.1])]
        dev = qml.device("default.qubit", wires=2)
        ham = (
            qml.pulse.constant * X(0)
            + (lambda p, t: jnp.sin(p * t)) * Z(0)
            + jnp.polyval * (Y(0) @ Y(1))
        )

        def ansatz(params):
            qml.evolve(ham)(params, 0.1)
            return qml.expval(Y(0) @ X(1))

        qnode_pulse_grad = qml.QNode(
            ansatz,
            dev,
            interface="jax",
            diff_method=pulse_odegen,
        )
        qnode_backprop = qml.QNode(ansatz, dev, interface="jax")

        with qml.Tracker(dev) as tracker:
            grad_pulse_grad = jax.grad(qnode_pulse_grad)(params)
        assert tracker.totals["executions"] == 1 + 12  # one forward execution, dim(DLA)=6
        grad_backprop = jax.grad(qnode_backprop)(params)

        assert all(
            qml.math.allclose(r, e, atol=1e-7) for r, e in zip(grad_pulse_grad, grad_backprop)
        )

    @pytest.mark.parametrize("argnums", [[0, 1], 0, 1])
    def test_simple_qnode_expval_multiple_params(self, argnums):
        """Test that a simple qnode with two parameters
        can be differentiated with pulse_odegen and `argnums` works as expected."""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        dev = qml.device("default.qubit", wires=1)
        T = 0.2
        ham1 = qml.pulse.constant * Y(0)
        ham2 = qml.pulse.constant * Y(0)

        @qml.qnode(dev, interface="jax", diff_method=pulse_odegen)
        def circuit(param1, param2):
            qml.evolve(ham1)(param1, T)
            qml.evolve(ham2)(param2, T)
            return qml.expval(Z(0))

        param1, param2 = [jnp.array(0.4)], [jnp.array(0.5)]
        with qml.Tracker(dev) as tracker:
            grad = jax.jacobian(circuit, argnums=argnums)(param1, param2)
        p = (param1[0] + param2[0]) * T
        exp_grad = -2 * jnp.sin(2 * p) * T
        if argnums == [0, 1]:
            for g in grad:
                assert qml.math.allclose(g, exp_grad)
                assert tracker.totals["executions"] == 1 + 4  # one forward pass, four shifted tapes
        else:
            assert qml.math.allclose(grad, exp_grad)
            assert tracker.totals["executions"] == 1 + 2  # one forward pass, two shifted tapes


@pytest.mark.jax
class TestPulseOdegenDiff:
    """Test that pulse_odegen is differentiable, i.e. that computing
    the derivative with pulse_odegen is differentiable a second time,
    yielding the Hessian."""

    # pylint: disable=too-few-public-methods
    @pytest.mark.slow
    def test_jax(self):
        """Test that pulse_odegen is differentiable,
        allowing to compute the Hessian, with JAX.."""
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        dev = qml.device("default.qubit", wires=1)
        T = 0.5
        ham_single_q_const = qml.pulse.constant * Y(0)

        def fun(params):
            op = qml.evolve(ham_single_q_const)(params, T)
            tape = qml.tape.QuantumScript([op], [qml.expval(Z(0))])
            tape.trainable_params = [0]
            _tapes, fn = pulse_odegen(tape)
            return fn(qml.execute(_tapes, dev, "backprop"))

        params = [jnp.array(0.4)]
        p = params[0] * T
        exp_grad = -2 * jnp.sin(2 * p) * T
        grad = fun(params)
        assert qml.math.isclose(grad, exp_grad)

        exp_diff_of_grad = -4 * jnp.cos(2 * p) * T**2
        diff_of_grad = jax.grad(fun)(params)
        assert qml.math.isclose(diff_of_grad, exp_diff_of_grad)
