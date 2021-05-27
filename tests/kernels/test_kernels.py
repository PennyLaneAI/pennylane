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
import pennylane as qml
import pennylane.kernels as kern
import pytest
import numpy as np
from pennylane import numpy as pnp
import math
import sys


@qml.template
def _simple_ansatz(x, params):
    """A simple quantum circuit ansatz."""
    qml.RX(params[0], wires=[0])
    qml.RZ(x, wires=[0])
    qml.RX(params[1], wires=[0])


def _mock_kernel(x1, x2, history):
    """A kernel that memorizes its calls and encodes a fixed values for equal/unequal
    datapoint pairs."""
    history.append((x1, x2))

    if x1 == x2:
        return 1
    else:
        return 0.2


def _laplace_kernel(x1, x2):
    """The laplace kernel on scalar data."""
    return np.exp(-math.fabs(x1 - x2))


class TestKernelMatrix:
    """Tests kernel matrix computations."""

    def test_simple_kernel(self):
        """Test square_kernel_matrix and kernel_matrix of the _mock_kernel above."""
        X1 = [0.1, 0.4]
        X2 = [0.1, 0.3, 0.5]

        K1_expected = pnp.array([[1, 0.2], [0.2, 1]])
        K2_expected = pnp.array([[1, 0.2, 0.2], [0.2, 0.2, 0.2]])

        K1 = kern.square_kernel_matrix(X1, lambda x1, x2: _mock_kernel(x1, x2, []))
        K2 = kern.kernel_matrix(X1, X2, lambda x1, x2: _mock_kernel(x1, x2, []))

        assert np.array_equal(K1, K1_expected)
        assert np.array_equal(K2, K2_expected)

    def test_laplace_kernel(self):
        """Test square_kernel_matrix and kernel_matrix of the _laplace_kernel above."""
        X1 = [0.1, 0.4, 0.2]
        X2 = [0.0, 0.1, 0.3, 0.2]

        K1_expected = pnp.exp(-np.array([[0.0, 0.3, 0.1], [0.3, 0.0, 0.2], [0.1, 0.2, 0.0]]))
        K2_expected = pnp.exp(
            -np.array([[0.1, 0.0, 0.2, 0.1], [0.4, 0.3, 0.1, 0.2], [0.2, 0.1, 0.1, 0.0]])
        )

        K1 = kern.square_kernel_matrix(X1, _laplace_kernel, assume_normalized_kernel=False)
        K2 = kern.kernel_matrix(X1, X2, _laplace_kernel)

        assert np.allclose(K1, K1_expected)
        assert np.allclose(K2, K2_expected)


class TestKernelPolarity:
    """Tests kernel methods to compute polarity."""

    def test_correct_calls(self):
        """Test number and order of calls of the kernel function when computing the
        polarity, including computation of the diagonal kernel matrix entries."""
        X = [0.1, 0.4]
        Y = [1, -1]

        hist = []

        kern.polarity(X, Y, lambda x1, x2: _mock_kernel(x1, x2, hist))

        assert hist == [(0.1, 0.1), (0.1, 0.4), (0.4, 0.4)]

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

        assert hist == [(0.1, 0.1), (0.1, 0.4), (0.4, 0.4)]

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

        assert alignment == 1.6 / (2 * math.sqrt(2 + 2 * 0.2 ** 2))
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

        assert alignment == 2.4 / (2 * math.sqrt(2 + 2 * 0.2 ** 2))
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
            # the small perturbation ensures that the solver does not get stuck
            (np.array([[0, 1.000001], [1, 0]]), True, np.array([[1, 1], [1, 1]])),
        ],
    )
    def test_closest_psd_matrix(self, input, fix_diagonal, expected_output):
        """Test obtaining the closest positive semi-definite matrix using a semi-definite program."""
        try:
            import cvxpy as cp

            output = kern.closest_psd_matrix(input, fix_diagonal=fix_diagonal, feastol=1e-10)
        except ModuleNotFoundError:
            pytest.skip(
                "cvxpy seems to not be installed on the system."
                "It is required for qml.kernels.closest_psd_matrix"
                " and can be installed via `pip install cvxpy`."
            )
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
            output = kern.closest_psd_matrix(input, fix_diagonal=True, feastol=1e-10)

        assert "CVXPY is required" in str(import_error.value)

    @pytest.mark.parametrize(
        "input,solver",
        [
            (np.diag([1, -1]), "I am not a solver"),
        ],
    )
    def test_closest_psd_matrix_solve_error(self, input, solver):
        """Test verbose error raising if problem.solve crashes."""
        with pytest.raises(Exception) as solve_error:
            output = kern.closest_psd_matrix(input, solver=solver, fix_diagonal=True, feastol=1e-10)

        assert "CVXPY solver did not converge." in str(solve_error.value)


def depolarize(mat, rates, num_wires, level):
    """Apply effect of depolarizing noise in circuit to kernel matrix."""
    if level == "per_circuit":
        noisy_mat = (1 - rates) * mat + rates / (2 ** num_wires) * np.ones_like(mat)
    elif level == "per_embedding":
        noisy_mat = np.copy(mat)
        for i in range(len(mat)):
            for j in range(i, len(mat)):
                rate = rates[i] + rates[j] - rates[i] * rates[j]
                noisy_mat[i, j] *= 1 - rate
                noisy_mat[i, j] += rate / (2 ** num_wires)

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
