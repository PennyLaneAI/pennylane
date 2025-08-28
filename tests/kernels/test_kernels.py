# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
Unit tests for the :mod:`pennylane` kernels module.
"""
# pylint: disable=import-outside-toplevel
import math
import sys
from functools import partial

import numpy as np
import pytest

import pennylane as qml
import pennylane.kernels as kern
from pennylane import numpy as pnp


@pytest.fixture(name="cvxpy_support")
def fixture_cvxpy_support():
    """Fixture to determine whether cvxpy and cvxopt are installed."""
    # pylint: disable=unused-import
    try:
        import cvxopt
        import cvxpy

        cvxpy_support = True
    except ModuleNotFoundError:
        cvxpy_support = False

    return cvxpy_support


@pytest.fixture()
def skip_if_no_cvxpy_support(cvxpy_support):
    """Fixture to skip a test if cvxpy or cvxopt are not installed."""
    if not cvxpy_support:
        pytest.skip("Skipped, no cvxpy support")


def _mock_kernel(x1, x2, history, batch=None):
    """A kernel that memorizes its calls and encodes a fixed values for equal/unequal
    datapoint pairs."""
    history.append((x1, x2))

    mult = 1.0 if batch is None else np.linspace(1, 5, batch)

    if x1 == x2:
        return 1 * mult
    return 0.2 * mult


def _laplace_kernel(x1, x2, batch=None):
    """The laplace kernel on scalar data."""
    mult = 1.0 if batch is None else np.linspace(1, 5, batch)

    return mult * np.exp(-math.fabs(x1 - x2))


def _diffable_kernel(x1, x2):
    """A differentiable kernel"""
    return qml.math.exp(-((x1 - x2) ** 2))


def _jacobian_of_diffable_kernel(x1, x2):
    """Analytic Jacobian of diffable_kernel"""
    return -2 * (x1 - x2) * qml.math.exp(-((x1 - x2) ** 2))


class TestKernelMatrix:
    """Tests kernel matrix computations."""

    @pytest.mark.parametrize("batch", [None, 1, 4])
    def test_simple_kernel(self, batch):
        """Test square_kernel_matrix and kernel_matrix of the _mock_kernel above."""
        X1 = [0.1, 0.4]
        X2 = [0.1, 0.3, 0.5]

        K1_expected = pnp.array([[1, 0.2], [0.2, 1]])
        K2_expected = pnp.array([[1, 0.2, 0.2], [0.2, 0.2, 0.2]])
        if batch is not None:
            K1_expected = pnp.tensordot(np.linspace(1, 5, batch), K1_expected, axes=0)
            K2_expected = pnp.tensordot(np.linspace(1, 5, batch), K2_expected, axes=0)

        K1 = kern.square_kernel_matrix(X1, partial(_mock_kernel, history=[], batch=batch))
        K2 = kern.kernel_matrix(X1, X2, partial(_mock_kernel, history=[], batch=batch))

        assert np.array_equal(K1, K1_expected)
        assert np.array_equal(K2, K2_expected)

    def test_square_kernel_single_datapoint(self):
        """Test that the square kernel matrix of a data set containing a single
        data point is computed correctly for the _mock_kernel above."""
        X = [0.1]

        hist = []

        K1 = kern.square_kernel_matrix(X, partial(_mock_kernel, history=hist), True)
        assert not hist
        assert qml.math.array_equal(K1, np.eye(1))
        K2 = kern.square_kernel_matrix(X, partial(_mock_kernel, history=hist), False)
        assert hist == [(0.1, 0.1)]
        assert qml.math.array_equal(K2, np.eye(1))
        # When not using assume_normalized_kernel, also test batching
        hist = []
        K3 = kern.square_kernel_matrix(X, partial(_mock_kernel, history=hist, batch=4), False)
        assert hist == [(0.1, 0.1)]
        assert qml.math.array_equal(K3, np.array([[[val]] for val in np.linspace(1, 5, 4)]))

    @pytest.mark.parametrize("batch", [None, 1, 4])
    def test_laplace_kernel(self, batch):
        """Test square_kernel_matrix and kernel_matrix of the _laplace_kernel above."""
        X1 = [0.1, 0.4, 0.2]
        X2 = [0.0, 0.1, 0.3, 0.2]

        K1_expected = pnp.exp(-np.array([[0.0, 0.3, 0.1], [0.3, 0.0, 0.2], [0.1, 0.2, 0.0]]))
        K2_expected = pnp.exp(
            -np.array([[0.1, 0.0, 0.2, 0.1], [0.4, 0.3, 0.1, 0.2], [0.2, 0.1, 0.1, 0.0]])
        )
        if batch is not None:
            K1_expected = pnp.tensordot(np.linspace(1, 5, batch), K1_expected, axes=0)
            K2_expected = pnp.tensordot(np.linspace(1, 5, batch), K2_expected, axes=0)

        K1 = kern.square_kernel_matrix(X1, partial(_laplace_kernel, batch=batch), False)
        K2 = kern.kernel_matrix(X1, X2, partial(_laplace_kernel, batch=batch))

        assert np.allclose(K1, K1_expected)
        assert np.allclose(K2, K2_expected)

    X1 = [0.1, 0.4, 0.2]
    X2 = [0.0, 0.1, 0.3, 0.2]
    expected_K1 = np.exp(-((np.outer(X1, np.ones(3)) - np.outer(np.ones(3), X1)) ** 2))
    expected_K2 = np.exp(-((np.outer(X1, np.ones(4)) - np.outer(np.ones(3), X2)) ** 2))
    expected_K3 = expected_K1.copy()
    expected_K3 = expected_K3 - np.diag(np.diag(expected_K3)) + np.eye(3)
    expected_dK1 = np.zeros((3, 3, 3))
    expected_dK3 = np.zeros((3, 3, 3))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X1):
            expected_dK1[i, j, i] = _jacobian_of_diffable_kernel(x1, x2)
            expected_dK1[i, j, j] = -_jacobian_of_diffable_kernel(x1, x2)
            if i != j:
                expected_dK3[i, j, i] = _jacobian_of_diffable_kernel(x1, x2)
                expected_dK3[i, j, j] = -_jacobian_of_diffable_kernel(x1, x2)

    expected_dK2 = (np.zeros((3, 4, 3)), np.zeros((3, 4, 4)))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            expected_dK2[0][i, j, i] = _jacobian_of_diffable_kernel(x1, x2)
            expected_dK2[1][i, j, j] = -_jacobian_of_diffable_kernel(x1, x2)

    @pytest.mark.autograd
    def test_autograd(self):
        """Test differentiability of the kernel matrix methods with Autograd."""
        X1 = pnp.array(self.X1, requires_grad=True)
        X2 = pnp.array(self.X2, requires_grad=True)
        K1 = kern.square_kernel_matrix(X1, _diffable_kernel, False)
        K2 = kern.kernel_matrix(X1, X2, _diffable_kernel)
        K3 = kern.square_kernel_matrix(X1, _diffable_kernel, True)

        assert qml.math.allclose(K1, self.expected_K1)
        assert qml.math.allclose(K2, self.expected_K2)
        assert qml.math.allclose(K3, self.expected_K3)

        dK1 = qml.jacobian(kern.square_kernel_matrix, argnum=0)(X1, _diffable_kernel, False)
        assert qml.math.allclose(dK1, self.expected_dK1)
        dK2 = qml.jacobian(kern.kernel_matrix, argnum=(0, 1))(X1, X2, _diffable_kernel)
        assert qml.math.allclose(dK2[0], self.expected_dK2[0])
        assert qml.math.allclose(dK2[1], self.expected_dK2[1])
        dK3 = qml.jacobian(kern.square_kernel_matrix, argnum=0)(X1, _diffable_kernel, True)
        assert qml.math.allclose(dK3, self.expected_dK3)

    @pytest.mark.jax
    def test_jax(self):
        """Test differentiability of the kernel matrix methods with JAX."""
        import jax

        jax.config.update("jax_enable_x64", True)
        jnp = jax.numpy

        X1 = jnp.array(self.X1)
        X2 = jnp.array(self.X2)
        K1 = kern.square_kernel_matrix(X1, _diffable_kernel, False)
        K2 = kern.kernel_matrix(X1, X2, _diffable_kernel)
        K3 = kern.square_kernel_matrix(X1, _diffable_kernel, True)

        assert qml.math.allclose(K1, self.expected_K1)
        assert qml.math.allclose(K2, self.expected_K2)
        assert qml.math.allclose(K3, self.expected_K3)

        dK1 = jax.jacobian(kern.square_kernel_matrix, argnums=0)(X1, _diffable_kernel, False)
        assert qml.math.allclose(dK1, self.expected_dK1)
        dK2 = jax.jacobian(kern.kernel_matrix, argnums=(0, 1))(X1, X2, _diffable_kernel)
        assert qml.math.allclose(dK2[0], self.expected_dK2[0])
        assert qml.math.allclose(dK2[1], self.expected_dK2[1])
        dK3 = jax.jacobian(kern.square_kernel_matrix, argnums=0)(X1, _diffable_kernel, True)
        assert qml.math.allclose(dK3, self.expected_dK3)

    @pytest.mark.torch
    def test_torch(self):
        """Test differentiability of the kernel matrix methods with PyTorch."""
        import torch

        X1 = torch.tensor(self.X1, requires_grad=True)
        X2 = torch.tensor(self.X2, requires_grad=True)
        K1 = kern.square_kernel_matrix(X1, _diffable_kernel, False)
        K2 = kern.kernel_matrix(X1, X2, _diffable_kernel)
        K3 = kern.square_kernel_matrix(X1, _diffable_kernel, True)

        assert qml.math.allclose(K1, self.expected_K1)
        assert qml.math.allclose(K2, self.expected_K2)
        assert qml.math.allclose(K3, self.expected_K3)

        jac = torch.autograd.functional.jacobian
        dK1 = jac(
            partial(
                kern.square_kernel_matrix,
                kernel=_diffable_kernel,
                assume_normalized_kernel=False,
            ),
            X1,
        )
        assert qml.math.allclose(dK1, self.expected_dK1)
        dK2 = jac(partial(kern.kernel_matrix, kernel=_diffable_kernel), (X1, X2))
        assert qml.math.allclose(dK2[0], self.expected_dK2[0])
        assert qml.math.allclose(dK2[1], self.expected_dK2[1])
        dK3 = jac(
            partial(
                kern.square_kernel_matrix,
                kernel=_diffable_kernel,
                assume_normalized_kernel=True,
            ),
            X1,
        )
        assert qml.math.allclose(dK3, self.expected_dK3)

    @pytest.mark.tf
    def test_tf(self):
        """Test differentiability of the kernel matrix methods with Tensorflow."""
        import tensorflow as tf

        X1 = tf.Variable(self.X1)
        X2 = tf.Variable(self.X2)
        with tf.GradientTape(persistent=True) as tape:
            K1 = kern.square_kernel_matrix(X1, _diffable_kernel, False)
            K2 = kern.kernel_matrix(X1, X2, _diffable_kernel)
            K3 = kern.square_kernel_matrix(X1, _diffable_kernel, True)

        assert qml.math.allclose(K1, self.expected_K1)
        assert qml.math.allclose(K2, self.expected_K2)
        assert qml.math.allclose(K3, self.expected_K3)

        dK1 = tape.jacobian(K1, X1)
        assert qml.math.allclose(dK1, self.expected_dK1)
        dK2 = tape.jacobian(K2, (X1, X2))
        assert qml.math.allclose(dK2[0], self.expected_dK2[0])
        assert qml.math.allclose(dK2[1], self.expected_dK2[1])
        dK3 = tape.jacobian(K3, X1)
        assert qml.math.allclose(dK3, self.expected_dK3)


class TestKernelPolarity:
    """Tests kernel methods to compute polarity."""

    def test_correct_calls(self):
        """Test number and order of calls of the kernel function when computing the
        polarity, including computation of the diagonal kernel matrix entries."""
        X = [0.1, 0.4]
        Y = [1, -1]

        hist = []

        kern.polarity(X, Y, lambda x1, x2: _mock_kernel(x1, x2, hist))

        assert hist == [(0.1, 0.4), (0.1, 0.1), (0.4, 0.4)]

    def test_correct_calls_normalized(self):
        """Test number and order of calls of the kernel function when computing the
        polarity, assuming normalized diagonal kernel matrix entries."""
        X = [0.1, 0.4]
        Y = [1, -1]

        hist = []

        kern.polarity(
            X, Y, lambda x1, x2: _mock_kernel(x1, x2, hist), assume_normalized_kernel=True
        )

        assert hist == [(0.1, 0.4)]

    def test_polarity_value(self):
        """Test value of polarity without class label rescaling (1/2)."""
        X = [0.1, 0.4]
        Y = [1, -1]
        pol = kern.polarity(
            X, Y, lambda x1, x2: _mock_kernel(x1, x2, []), rescale_class_labels=False
        )
        pol_assume = kern.polarity(
            X,
            Y,
            lambda x1, x2: _mock_kernel(x1, x2, []),
            assume_normalized_kernel=True,
            rescale_class_labels=False,
        )

        assert pol == 1.6
        assert pol == pol_assume

    def test_polarity_value_other_labels(self):
        """Test value of polarity without class label rescaling (2/2)."""
        X = [0.1, 0.4]
        Y = [1, 1]
        pol = kern.polarity(
            X, Y, lambda x1, x2: _mock_kernel(x1, x2, []), rescale_class_labels=False
        )
        pol_assume = kern.polarity(
            X,
            Y,
            lambda x1, x2: _mock_kernel(x1, x2, []),
            assume_normalized_kernel=True,
            rescale_class_labels=False,
        )

        assert pol == 2.4
        assert pol == pol_assume


class TestKernelTargetAlignment:
    """Tests computation of kernel target alignment."""

    def test_correct_calls(self):
        """Test number and order of calls of the kernel function when computing the
        kernel target alignment, including computation of the diagonal kernel matrix entries."""
        X = [0.1, 0.4]
        Y = [1, -1]

        hist = []

        kern.target_alignment(X, Y, lambda x1, x2: _mock_kernel(x1, x2, hist))

        assert hist == [(0.1, 0.4), (0.1, 0.1), (0.4, 0.4)]

    def test_correct_calls_normalized(self):
        """Test number and order of calls of the kernel function when computing the
        kernel target alignment, assuming normalized diagonal kernel matrix entries."""
        X = [0.1, 0.4]
        Y = [1, -1]

        hist = []

        kern.target_alignment(
            X, Y, lambda x1, x2: _mock_kernel(x1, x2, hist), assume_normalized_kernel=True
        )

        assert hist == [(0.1, 0.4)]

    def test_alignment_value(self):
        """Test value of kernel target alignment without class label rescaling (1/3)."""
        X = [0.1, 0.4]
        Y = [1, -1]

        alignment = kern.target_alignment(
            X, Y, lambda x1, x2: _mock_kernel(x1, x2, []), rescale_class_labels=False
        )
        alignment_assume = kern.target_alignment(
            X,
            Y,
            lambda x1, x2: _mock_kernel(x1, x2, []),
            assume_normalized_kernel=True,
            rescale_class_labels=False,
        )

        assert alignment == 1.6 / (2 * math.sqrt(2 + 2 * 0.2**2))
        assert alignment == alignment_assume

    def test_alignment_value_other_labels(self):
        """Test value of kernel target alignment without class label rescaling (2/3)."""
        X = [0.1, 0.4]
        Y = [1, 1]
        alignment = kern.target_alignment(
            X, Y, lambda x1, x2: _mock_kernel(x1, x2, []), rescale_class_labels=False
        )
        alignment_assume = kern.target_alignment(
            X,
            Y,
            lambda x1, x2: _mock_kernel(x1, x2, []),
            assume_normalized_kernel=True,
            rescale_class_labels=False,
        )

        assert alignment == 2.4 / (2 * math.sqrt(2 + 2 * 0.2**2))
        assert alignment == alignment_assume

    def test_alignment_value_three(self):
        """Test value of kernel target alignment without class label rescaling
        on more data (3/3)."""
        X = [0.1, 0.4, 0.0]
        Y = [1, -1, 1]

        alignment = kern.target_alignment(
            X, Y, lambda x1, x2: _mock_kernel(x1, x2, []), rescale_class_labels=False
        )
        alignment_assume = kern.target_alignment(
            X,
            Y,
            lambda x1, x2: _mock_kernel(x1, x2, []),
            assume_normalized_kernel=True,
            rescale_class_labels=False,
        )

        K1 = np.array(
            [
                [
                    1,
                    0.2,
                    0.2,
                ],
                [0.2, 1, 0.2],
                [0.2, 0.2, 1],
            ]
        )
        K2 = np.array([[1, -1, 1], [-1, 1, -1], [1, -1, 1]])
        expected_alignment = np.trace(np.dot(K1, K2)) / math.sqrt(
            np.trace(np.dot(K1, K1)) * np.trace(np.dot(K2, K2))
        )

        assert alignment == expected_alignment
        assert alignment == alignment_assume

    def test_alignment_value_with_normalization(self):
        """Test value of kernel target alignment with class label rescaling."""
        X = [0.1, 0.4, 0.0]
        Y = [1, -1, 1]

        alignment = kern.target_alignment(
            X, Y, lambda x1, x2: _mock_kernel(x1, x2, []), rescale_class_labels=True
        )
        alignment_assume = kern.target_alignment(
            X,
            Y,
            lambda x1, x2: _mock_kernel(x1, x2, []),
            assume_normalized_kernel=True,
            rescale_class_labels=True,
        )

        K1 = np.array(
            [
                [
                    1,
                    0.2,
                    0.2,
                ],
                [0.2, 1, 0.2],
                [0.2, 0.2, 1],
            ]
        )
        _Y = np.array([1 / 2, -1, 1 / 2])
        K2 = np.outer(_Y, _Y)
        expected_alignment = np.trace(np.dot(K1, K2)) / math.sqrt(
            np.trace(np.dot(K1, K1)) * np.trace(np.dot(K2, K2))
        )

        assert alignment == expected_alignment
        assert alignment == alignment_assume


class TestRegularization:
    """Tests regularization/postprocessing methods."""

    @pytest.mark.parametrize(
        "input",
        [
            (np.diag([1, 0.4])),
            (np.diag([1, 0.0])),
            (np.array([[1, -0.5], [-0.5, 1]])),
        ],
    )
    def test_do_nothing_on_non_negative(self, input):
        """Test thresholding, displacing and flipping matrix to do nothing on PSD matrices."""
        assert np.allclose(kern.threshold_matrix(input), input)
        assert np.allclose(kern.displace_matrix(input), input)
        assert np.allclose(kern.flip_matrix(input), input)

    @pytest.mark.parametrize(
        "input,expected_output",
        [
            (np.diag([1, -1]), np.diag([1, 0])),
            (np.array([[1, 1], [1, -1]]), np.array([[1.20710678, 0.5], [0.5, 0.20710678]])),
            (np.array([[0, 1], [1, 0]]), np.array([[1, 1], [1, 1]]) / 2.0),
        ],
    )
    def test_threshold_matrix(self, input, expected_output):
        """Test thresholding of eigenvalues of a matrix."""
        assert np.allclose(kern.threshold_matrix(input), expected_output)

    @pytest.mark.parametrize(
        "input,expected_output",
        [
            (np.diag([1, -1]), np.diag([2, 0])),
            (
                np.array([[1, 1], [1, -1]]),
                np.array([[1 + math.sqrt(2), 1], [1, -1 + math.sqrt(2)]]),
            ),
            (np.array([[0, 1], [1, 0]]), np.array([[1, 1], [1, 1]])),
        ],
    )
    def test_displace_matrix(self, input, expected_output):
        """Test displacing/shifting of eigenvalues of a matrix."""
        assert np.allclose(kern.displace_matrix(input), expected_output)

    @pytest.mark.parametrize(
        "input,expected_output",
        [
            (np.diag([1, -1]), np.diag([1, 1])),
            (
                np.array([[1, 1], [1, -1]]),
                np.array([[math.sqrt(2), 0], [0, math.sqrt(2)]]),
            ),
            (np.array([[0, 1], [1, 0]]), np.array([[1, 0], [0, 1]])),
        ],
    )
    def test_flip_matrix(self, input, expected_output):
        """Test taking the absolute values of eigenvalues of a matrix."""
        assert np.allclose(kern.flip_matrix(input), expected_output)

    @pytest.mark.parametrize(
        "input,fix_diagonal,expected_output",
        [
            (np.diag([1, -1]), False, np.diag([1, 0])),
            (
                np.array([[1, 1], [1, -1]]),
                False,
                np.array([[math.sqrt(2) + 1, 1], [1, math.sqrt(2) - 1]]) / 2,
            ),
            (np.array([[0, 1], [1, 0]]), False, np.array([[1, 1], [1, 1]]) / 2),
            (np.diag([1, -1]), True, np.diag([1, 1])),
            (np.array([[1, 1], [1, -1]]), True, np.array([[1.0, 1.0], [1.0, 1.0]])),
        ],
    )
    @pytest.mark.usefixtures("skip_if_no_cvxpy_support")
    def test_closest_psd_matrix(self, input, fix_diagonal, expected_output):
        """Test obtaining the closest positive semi-definite matrix using a semi-definite program."""
        try:
            import cvxpy as cp

            output = kern.closest_psd_matrix(input, fix_diagonal=fix_diagonal, feastol=1e-10)
        except cp.error.SolverError:
            pytest.skip(
                "The cvxopt solver seems to not be installed on the system."
                "It is the default solver for qml.kernels.closest_psd_matrix"
                " and can be installed via `pip install cvxopt`."
            )

        assert np.allclose(output, expected_output, atol=1e-5)

    @pytest.mark.usefixtures("skip_if_no_cvxpy_support")
    def test_closest_psd_matrix_small_perturb(self):
        """Test obtaining the closest positive semi-definite matrix using a
        semi-definite program with a small perturbation input.

        The small perturbation ensures that the solver does not get stuck.
        """
        if sys.version_info.minor > 11:
            pytest.xfail("Test does not converge with Python 3.12")
        input, fix_diagonal, expected_output = (
            np.array([[0, 1.000001], [1, 0]]),
            True,
            np.array([[1, 1], [1, 1]]),
        )
        try:
            import cvxpy as cp

            output = kern.closest_psd_matrix(input, fix_diagonal=fix_diagonal, feastol=1e-10)
        except cp.error.SolverError:
            pytest.skip(
                "The cvxopt solver seems to not be installed on the system."
                "It is the default solver for qml.kernels.closest_psd_matrix"
                " and can be installed via `pip install cvxopt`."
            )

        assert np.allclose(output, expected_output, atol=1e-5)

    @pytest.mark.parametrize(
        "input",
        [
            (np.diag([1, -1])),
        ],
    )
    def test_closest_psd_matrix_import_error(self, input, mocker):
        """Test import error raising if cvxpy is not installed."""
        with pytest.raises(ImportError) as import_error:
            mocker.patch.dict(sys.modules, {"cvxpy": None})
            _ = kern.closest_psd_matrix(input, fix_diagonal=True, feastol=1e-10)

        assert "CVXPY is required" in str(import_error.value)

    @pytest.mark.parametrize(
        "input,solver",
        [
            (np.diag([1, -1]), "I am not a solver"),
        ],
    )
    @pytest.mark.usefixtures("skip_if_no_cvxpy_support")
    def test_closest_psd_matrix_solve_error(self, input, solver):
        """Test verbose error raising if problem.solve crashes."""
        with pytest.raises(Exception) as solve_error:
            _ = kern.closest_psd_matrix(input, solver=solver, fix_diagonal=True, feastol=1e-10)

        assert "CVXPY solver did not converge." in str(solve_error.value)


def depolarize(mat, rates, num_wires, level):
    """Apply effect of depolarizing noise in circuit to kernel matrix."""
    if level == "per_circuit":
        noisy_mat = (1 - rates) * mat + rates / (2**num_wires) * np.ones_like(mat)
    elif level == "per_embedding":
        noisy_mat = np.copy(mat)
        for i in range(len(mat)):
            for j in range(i, len(mat)):
                rate = rates[i] + rates[j] - rates[i] * rates[j]
                noisy_mat[i, j] *= 1 - rate
                noisy_mat[i, j] += rate / (2**num_wires)

    return noisy_mat


class TestMitigation:
    """Tests depolarizing noise mitigation techniques."""

    num_wires = 1

    @pytest.mark.parametrize(
        "input, use_entries, expected_output",
        [
            (np.diag([0.9, 0.9]), (0,), np.array([[1, -1 / 8], [-1 / 8, 1]])),
            (np.diag([0.9, 0.9]), (1,), np.array([[1, -1 / 8], [-1 / 8, 1]])),
            (np.diag([1.0, 0.9]), None, np.diag([1, 0.9])),
            (np.diag([1.0, 0.9]), (1,), np.array([[9 / 8, -1 / 8], [-1 / 8, 1.0]])),
            (
                depolarize(np.array([[1.0, 0.5], [0.5, 1.0]]), 0.1, num_wires, "per_circuit"),
                (0,),
                np.array([[1.0, 0.5], [0.5, 1.0]]),
            ),
            (
                depolarize(np.array([[1.0, 0.5], [0.5, 1.0]]), 0.1, num_wires, "per_circuit"),
                (1,),
                np.array([[1.0, 0.5], [0.5, 1.0]]),
            ),
        ],
    )
    def test_mitigate_depolarizing_noise_single(self, input, use_entries, expected_output):
        """Test mitigation of depolarizing noise in kernel matrix measuring a single noise rate."""
        output = kern.mitigate_depolarizing_noise(input, self.num_wires, "single", use_entries)
        assert np.allclose(output, expected_output)

    @pytest.mark.parametrize(
        "input, use_entries, expected_output",
        [
            (np.diag([0.9, 0.9]), None, np.array([[1, -1 / 8], [-1 / 8, 1]])),
            (np.diag([1.0, 0.9]), None, np.array([[19 / 18, -1 / 18], [-1 / 18, 17 / 18]])),
            (np.diag([1.0, 0.9]), (0, 1), np.array([[19 / 18, -1 / 18], [-1 / 18, 17 / 18]])),
            (
                depolarize(np.array([[1.0, 0.5], [0.5, 1.0]]), 0.1, num_wires, "per_circuit"),
                None,
                np.array([[1.0, 0.5], [0.5, 1.0]]),
            ),
        ],
    )
    def test_mitigate_depolarizing_noise_average(self, input, use_entries, expected_output):
        """Test mitigation of depolarizing noise in kernel matrix averaging a single noise rate."""
        output = kern.mitigate_depolarizing_noise(input, self.num_wires, "average", use_entries)
        assert np.allclose(output, expected_output)

    @pytest.mark.parametrize(
        "input, expected_output",
        [
            (np.diag([0.9, 0.9]), np.array([[1, -1 / 8], [-1 / 8, 1]])),
            (np.diag([1.0, 0.9]), np.array([[1, -0.059017], [-0.059017, 1.0]])),
            (
                depolarize(np.array([[1.0, 0.5], [0.5, 1.0]]), 0.1, num_wires, "per_circuit"),
                np.array([[1.0, 0.5], [0.5, 1.0]]),
            ),
            (
                depolarize(
                    np.array([[1.0, 0.5], [0.5, 1.0]]), [0.2, 0.1], num_wires, "per_embedding"
                ),
                np.array([[1.0, 0.5], [0.5, 1.0]]),
            ),
        ],
    )
    def test_mitigate_depolarizing_noise_split_channel(self, input, expected_output):
        """Test mitigation of depolarizing noise in kernel matrix estimating individual
        noise rates per datapoint."""
        output = kern.mitigate_depolarizing_noise(input, self.num_wires, "split_channel")
        assert np.allclose(output, expected_output)


class TestErrorForNonRealistic:
    """Tests that the noise mitigation techniques raise an error whenever the
    used quantities are too small."""

    def test_mitigate_depolarizing_noise_wrong_method(self):
        """Test that an error is raised when specifying an incorrect method."""
        with pytest.raises(
            ValueError, match="Incorrect noise depolarization mitigation method specified"
        ):
            qml.kernels.mitigate_depolarizing_noise(np.array([0]), 4, method="some_dummy_strat")

    def test_mitigate_depolarizing_noise_average_method_error(self):
        """Test that an error is raised when using the average method for the
        mitigation of depolarizing noise with a matrix that has too small diagonal
        entries."""
        num_wires = 6
        wires = range(num_wires)

        dev = qml.device("default.qubit", wires=num_wires)

        @qml.qnode(dev)
        def kernel_circuit(x1, x2):
            qml.templates.AngleEmbedding(x1, wires=wires)
            qml.adjoint(qml.templates.AngleEmbedding)(x2, wires=wires)
            return qml.probs(wires)

        def kernel(x1, x2):
            return kernel_circuit(x1, x2)[0]

        # "Training feature vectors"
        X_train = qml.numpy.tensor(
            [[0.73096199, 0.19012506, 0.57223395], [0.78126872, 0.53535039, 0.31160784]],
            requires_grad=True,
        )

        # Create symmetric square kernel matrix (for training)
        K = qml.kernels.square_kernel_matrix(X_train, kernel)

        # Add some (symmetric) Gaussian noise to the kernel matrix.
        N = qml.numpy.tensor(
            [[-2.33010045, -2.22195441], [-0.40680862, 0.21785961]], requires_grad=True
        )
        K += (N + N.T) / 2
        with pytest.raises(
            ValueError, match="The average noise mitigation method cannot be applied"
        ):
            qml.kernels.mitigate_depolarizing_noise(K, num_wires, method="average")

    @pytest.mark.parametrize(
        "msg, method", [("single", "single"), ("split channel", "split_channel")]
    )
    def test_mitigate_depolarizing_noise_error(self, msg, method):
        """Test that an error is raised for the mitigation of depolarizing
        noise with a matrix that has too small specified entries for the
        single and the split channel strategies."""
        num_wires = 6
        wires = range(num_wires)

        dev = qml.device("default.qubit", wires=num_wires)

        @qml.qnode(dev)
        def kernel_circuit(x1, x2):
            qml.templates.AngleEmbedding(x1, wires=wires)
            qml.adjoint(qml.templates.AngleEmbedding)(x2, wires=wires)
            return qml.probs(wires)

        def kernel(x1, x2):
            return kernel_circuit(x1, x2)[0]

        # "Training feature vectors"
        X_train = qml.numpy.tensor(
            [[0.39375865, 0.50895605, 0.30720779], [0.34389837, 0.7043728, 0.40067889]],
            requires_grad=True,
        )

        # Create symmetric square kernel matrix (for training)
        K = qml.kernels.square_kernel_matrix(X_train, kernel)

        # Add some (symmetric) Gaussian noise to the kernel matrix.
        N = qml.numpy.tensor(
            [[-1.15035284, 0.36726945], [0.26436627, -0.59287149]], requires_grad=True
        )
        K += (N + N.T) / 2

        with pytest.raises(
            ValueError, match=f"The {msg} noise mitigation method cannot be applied"
        ):
            qml.kernels.mitigate_depolarizing_noise(K, num_wires, method=method)
