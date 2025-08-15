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
Unit tests for functions needed for performing givens decomposition of a unitary.
"""

import pytest
from scipy.stats import ortho_group, unitary_group

import pennylane as qml
from pennylane import numpy as np
from pennylane.math.decomposition import _givens_matrix, _set_unitary_matrix, givens_decomposition


@pytest.mark.parametrize("left", [True, False])
@pytest.mark.parametrize(
    ("a", "b"),
    [
        (1.2, 2.3),
        (1.2j, 2.3j),
        (1.5 + 2.3j, 2.1 - 3.7j),
        (1.0, 0.0),
        (0.0, 1.0),
        (-1.0, 0.0),
        (0.0, -1.0),
        (0.0, 1j),
        (1.2, 2.3j),
        (0.0, 0.0),
    ],
)
def test_givens_matrix(a, b, left):
    r"""Test that `_givens_matrix` builds the correct Givens rotation matrices."""

    grot_mat = _givens_matrix(a, b, left)
    assert np.isreal(grot_mat[0, 1]) and np.isreal(grot_mat[1, 1])

    rotated_vector = grot_mat @ np.array([a, b])
    if np.abs(a) < 1e-8 or np.abs(b) < 1e-8:
        phase = 1.0
    else:
        phase = b * np.conj(a) / np.abs(a) / np.abs(b)
    hypot = np.hypot(np.abs(a), np.abs(b)) + 1e-15
    result_element = (phase * a * np.abs(a) + b * np.abs(b)) / hypot
    rvec = np.array([0.0, result_element]) if left else np.array([result_element, 0.0])
    assert np.allclose([a, b], 0.0) or np.allclose(rotated_vector, rvec)

    res1 = np.round(grot_mat @ grot_mat.conj().T, 5)
    res2 = np.round(grot_mat.conj().T @ grot_mat, 5)
    assert np.all(res1 == res2) and np.all(res1 == np.eye(2))


@pytest.mark.parametrize("left", [True, False])
@pytest.mark.parametrize(
    ("a", "b"),
    [
        (1.2, 2.3),
        (1.0, 1.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (-1.0, 0.0),
        (0.0, -1.0),
        (-1.0, -1.0),
        (0.1, 20.3),
        (-0.1, 20.3),
        (0.0, 0.0),
    ],
)
def test_givens_matrix_real(a, b, left):
    r"""Test that `_givens_matrix` builds the correct real-valued Givens rotation matrices."""

    grot_mat = _givens_matrix(a, b, left, real_valued=True)
    assert grot_mat.dtype == np.float64

    rotated_vector = grot_mat @ np.array([a, b])
    sign = np.sign(a * b)
    if np.abs(a) < 1e-8 or np.abs(b) < 1e-8:
        sign = 1.0
    hypot = np.hypot(np.abs(a), np.abs(b)) + 1e-15
    result_element = (sign * a * np.abs(a) + b * np.abs(b)) / hypot
    if not left:
        result_element *= sign
    rvec = np.array([0.0, result_element]) if left else np.array([result_element, 0.0])
    assert np.allclose([a, b], 0.0) or np.allclose(rotated_vector, rvec)
    assert np.allclose(grot_mat @ grot_mat.T, np.eye(2))
    assert np.allclose(grot_mat.T @ grot_mat, np.eye(2))


@pytest.mark.parametrize("left", [True, False])
@pytest.mark.parametrize("row", [True, False])
@pytest.mark.parametrize("indices", [[0, 1], [2, 3], [1, 4], [0, 3]])
@pytest.mark.parametrize("shape", [(5, 5), (6, 6)])
def test_givens_rotate(shape, indices, row, left):
    r"""Test that Givens rotation is performed correctly for matrices built via `_givens_matrix`."""
    matrix = np.random.rand(*shape) * 1j + np.random.rand(*shape)
    unitary, (i, j) = matrix.copy(), indices
    if row:
        a, b = matrix[indices, j - 1]
        grot_mat = _givens_matrix(a, b, left)
        unitary[indices] = grot_mat @ unitary[indices]
        res = b / np.abs(b) * np.hypot(np.abs(a), np.abs(b)) if b else 1.0
        if left:
            assert np.isclose(unitary[i, j - 1], 0.0) and np.isclose(unitary[j, j - 1], res)
        else:
            assert np.isclose(unitary[i, j - 1], res) and np.isclose(unitary[j, j - 1], 0.0)
    else:
        a, b = matrix[j - 1, indices].T
        grot_mat = _givens_matrix(a, b, left)
        unitary[:, indices] = unitary[:, indices] @ grot_mat.T
        res = b / np.abs(b) * np.hypot(np.abs(a), np.abs(b)) if b else 1.0
        if left:
            assert np.isclose(unitary[j - 1, i], 0.0) and np.isclose(unitary[j - 1, j], res)
        else:
            assert np.isclose(unitary[j - 1, indices[0]], res) and np.isclose(
                unitary[j - 1, indices[1]], 0.0
            )


@pytest.mark.parametrize("shape", [2, 3, 7, 8, 15, 16])
def test_givens_decomposition(shape, seed):
    r"""Test that `givens_decomposition` performs a correct Givens decomposition."""

    matrix = unitary_group.rvs(shape, random_state=seed)

    phase_mat, ordered_rotations = givens_decomposition(matrix)
    assert all(j == i + 1 for _, (i, j) in ordered_rotations)
    decomposed_matrix = np.diag(phase_mat)
    for grot_mat, (i, _) in ordered_rotations:
        rotation_matrix = np.eye(shape, dtype=complex)
        rotation_matrix[i : i + 2, i : i + 2] = grot_mat
        decomposed_matrix = decomposed_matrix @ rotation_matrix

    # check if U = D x Π T_{m, n}
    assert np.allclose(matrix, decomposed_matrix), f"\n{matrix}\n{decomposed_matrix}"


@pytest.mark.jax
@pytest.mark.parametrize("shape", [2, 3, 7])
@pytest.mark.parametrize("jit", [False, True])
def test_givens_decomposition_jax(shape, jit, seed):
    r"""Test that `givens_decomposition` performs a correct Givens decomposition."""
    import jax
    from jax import numpy as jnp

    matrix = jnp.array(unitary_group.rvs(shape, random_state=seed))
    func = jax.jit(givens_decomposition) if jit else givens_decomposition

    phase_mat, ordered_rotations = func(matrix)
    assert all(j == i + 1 for _, (i, j) in ordered_rotations)
    decomposed_matrix = np.diag(phase_mat)
    for grot_mat, (i, _) in ordered_rotations:
        rotation_matrix = np.eye(shape, dtype=complex)
        rotation_matrix[i : i + 2, i : i + 2] = grot_mat
        decomposed_matrix = decomposed_matrix @ rotation_matrix

    # check if U = D x Π T_{m, n}
    assert np.allclose(matrix, decomposed_matrix), f"\n{matrix}\n{decomposed_matrix}"


@pytest.mark.parametrize("shape", [2, 3, 4, 5, 6, 7, 8, 14, 15, 16])
@pytest.mark.parametrize("dtype", [np.complex128, np.float64])
def test_givens_decomposition_real_valued(shape, dtype, seed):
    r"""Test that `givens_decomposition` performs a correct Givens decomposition of
    real-valued matrices, both for real and complex data type."""

    matrix = ortho_group.rvs(shape, random_state=seed).astype(dtype)
    matrix[0] *= np.linalg.det(matrix)  # Make unit determinant

    phase_mat, ordered_rotations = givens_decomposition(matrix)
    assert all(j == i + 1 for _, (i, j) in ordered_rotations)
    decomposed_matrix = np.diag(phase_mat)
    if dtype is np.float64:
        assert np.allclose(phase_mat, 1.0)

    for grot_mat, (i, _) in ordered_rotations:
        rotation_matrix = np.eye(shape, dtype=dtype)
        rotation_matrix[i : i + 2, i : i + 2] = grot_mat
        decomposed_matrix = decomposed_matrix @ rotation_matrix

    # check data type
    assert decomposed_matrix.dtype == dtype
    # check if U = D x Π T_{m, n}
    assert np.allclose(matrix, decomposed_matrix), f"\n{matrix}\n{decomposed_matrix}"


@pytest.mark.jax
@pytest.mark.parametrize("shape", [2, 3, 4, 5, 6])
@pytest.mark.parametrize("dtype", [np.complex128, np.float64])
@pytest.mark.parametrize("jit", [False, True])
def test_givens_decomposition_real_valued_jax(shape, dtype, jit, seed):
    r"""Test that `givens_decomposition` performs a correct Givens decomposition of
    real-valued matrices, both for real and complex data type, using JAX."""
    import jax
    from jax import numpy as jnp

    matrix = ortho_group.rvs(shape, random_state=seed).astype(dtype)
    matrix[0] *= np.linalg.det(matrix)  # Make unit determinant
    matrix = jnp.array(matrix)
    func = jax.jit(givens_decomposition) if jit else givens_decomposition

    phase_mat, ordered_rotations = func(matrix)
    assert all(j == i + 1 for _, (i, j) in ordered_rotations)
    decomposed_matrix = np.diag(phase_mat)
    if dtype is np.float64:
        assert np.allclose(phase_mat, 1.0)

    for grot_mat, (i, _) in ordered_rotations:
        rotation_matrix = np.eye(shape, dtype=dtype)
        rotation_matrix[i : i + 2, i : i + 2] = grot_mat
        decomposed_matrix = decomposed_matrix @ rotation_matrix

    # check data type
    assert decomposed_matrix.dtype == dtype
    # check if U = D x Π T_{m, n}
    assert np.allclose(matrix, decomposed_matrix), f"\n{matrix}\n{decomposed_matrix}"


@pytest.mark.parametrize(
    ("unitary_matrix", "msg_match"),
    [
        (
            np.array(
                [
                    [0.51378719 + 0.0j, 0.0546265 + 0.79145487j, -0.2051466 + 0.2540723j],
                    [0.62651582 + 0.0j, -0.00828925 - 0.60570321j, -0.36704948 + 0.32528067j],
                ]
            ),
            "The unitary matrix should be of shape NxN",
        ),
        (
            np.array(
                [
                    [0.51378719 + 0.0j, 0.0546265 + 0.79145487j, -0.2051466 + 0.2540723j],
                    [0.62651582 + 0.0j, -0.00828925 - 0.60570321j, -0.36704948 + 0.32528067j],
                ]
            ).T,
            "The unitary matrix should be of shape NxN",
        ),
    ],
)
def test_givens_decomposition_exceptions(unitary_matrix, msg_match):
    """Test that givens_decomposition throws an exception if the parameters have illegal shapes."""

    with pytest.raises(ValueError, match=msg_match):
        givens_decomposition(unitary_matrix)


@pytest.mark.jax
def test_givens_matrix_exceptions():
    """Test that _givens_matrix throws an exception if the parameters have different interface."""
    import jax.numpy as jnp

    a = np.array(1.2)
    b = jnp.array(2.3)

    with pytest.raises(TypeError, match="The interfaces of 'a' and 'b' do not match."):
        _givens_matrix(a, b)


@pytest.mark.jax
def test_givens_matrix_jaxpr():
    """Verify the JAXPR representation includes a function"""
    import jax.numpy as jnp
    from jax import make_jaxpr

    a = jnp.array(1.2)
    b = jnp.array(2.3)

    assert "givens_matrix_jax" in str(make_jaxpr(_givens_matrix)(a, b))


# pylint:disable = too-many-arguments
@pytest.mark.parametrize(
    ("jax", "unitary_matrix", "index", "value", "like", "expected_matrix"),
    [
        (False, np.array([[1, 0], [0, 1]]), (0, 0), 5, None, np.array([[5, 0], [0, 1]])),
        (False, np.array([[1, 0], [0, 1]]), (0, 0), 5, "numpy", np.array([[5, 0], [0, 1]])),
        (
            False,
            np.array([[1, 0], [0, 1]]),
            (0, Ellipsis),
            [1, 2],
            None,
            np.array([[1, 2], [0, 1]]),
        ),
        (
            False,
            np.array([[1, 0], [0, 1]]),
            (0, Ellipsis),
            [1, 2],
            "numpy",
            np.array([[1, 2], [0, 1]]),
        ),
        (False, np.array([[1, 0], [0, 1]]), (1, [0, 1]), [1, 2], None, np.array([[1, 0], [1, 2]])),
        (
            False,
            np.array([[1, 0], [0, 1]]),
            (1, [0, 1]),
            [1, 2],
            "numpy",
            np.array([[1, 0], [1, 2]]),
        ),
        (True, [[1, 0], [0, 1]], (0, 0), 5, None, [[5, 0], [0, 1]]),
        (True, [[1, 0], [0, 1]], (0, 0), 5, "jax", [[5, 0], [0, 1]]),
        (True, [[1, 0], [0, 1]], (0, Ellipsis), [1, 2], None, [[1, 2], [0, 1]]),
        (True, [[1, 0], [0, 1]], (0, Ellipsis), [1, 2], "jax", [[1, 2], [0, 1]]),
        (True, [[1, 0], [0, 1]], (1, [0, 1]), [1, 2], None, [[1, 0], [1, 2]]),
        (True, [[1, 0], [0, 1]], (1, [0, 1]), [1, 2], "jax", [[1, 0], [1, 2]]),
    ],
)
@pytest.mark.jax
def test_set_unitary_matrix(jax, unitary_matrix, index, value, like, expected_matrix):
    """Test the _set_unitary function on different interfaces."""
    import jax.numpy as jnp

    if jax:
        unitary_matrix = jnp.array(unitary_matrix)
        expected_matrix = jnp.array(expected_matrix)

    new_unitary_matrix = _set_unitary_matrix(unitary_matrix, index, value, like)
    assert qml.math.allclose(new_unitary_matrix, expected_matrix)
