# Copyright

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
import math


@qml.template
def _simple_ansatz(x, params):
    qml.RX(params[0], wires=[0])
    qml.RZ(x, wires=[0])
    qml.RX(params[1], wires=[0])


class TestEmbeddingKernel:
    def test_construction(self):
        dev = qml.device("default.qubit", wires=1)
        k = kern.EmbeddingKernel(_simple_ansatz, dev)

        assert k.probs_qnode is not None

    @pytest.mark.parametrize("x1", np.linspace(0, 2 * np.pi, 5))
    @pytest.mark.parametrize("x2", np.linspace(0, 2 * np.pi, 5))
    def test_value_range(self, x1, x2):
        dev = qml.device("default.qubit", wires=1)
        k = kern.EmbeddingKernel(_simple_ansatz, dev)
        params = np.array([0.5, 0.9])

        val = k(x1, x2, params)

        assert 0 <= val
        assert val <= 1

    def test_known_values(self):
        dev = qml.device("default.qubit", wires=1)
        k = kern.EmbeddingKernel(_simple_ansatz, dev)
        params = np.array([0.5, 0.9])

        val = k(0.1, 0.1, params)

        assert val == pytest.approx(1.0)

    def test_kernel_matrix(self):
        dev = qml.device("default.qubit", wires=1)
        k = kern.EmbeddingKernel(_simple_ansatz, dev)
        params = np.array([0.5, 0.9])

        K = k.kernel_matrix([0.1, 0.2, 0.4], params)

        # TODO: Add value tests

        assert np.allclose(K, np.transpose(K))
        assert np.allclose(np.diag(K), np.array([1, 1, 1]))

    def test_kernel_target_alignment(self):
        dev = qml.device("default.qubit", wires=1)
        k = kern.EmbeddingKernel(_simple_ansatz, dev)
        params = np.array([0.5, 0.9])

        alignment = k.target_alignment([0.1, 0.2, 0.4], [1, -1, 1], params)

        # TODO: Add value tests

        assert 0 <= alignment
        assert alignment <= 1

    def test_kernel_polarization(self):
        dev = qml.device("default.qubit", wires=1)
        k = kern.EmbeddingKernel(_simple_ansatz, dev)
        params = np.array([0.5, 0.9])

        polarization = k.polarization([0.1, 0.2, 0.4], [1, -1, 1], params)

        # TODO: Add value tests

        assert 0 <= polarization


def _mock_kernel(x1, x2, history):
    history.append((x1, x2))

    if x1 == x2:
        return 1
    else:
        return 0.2


def _laplace_kernel(x1, x2):
    return np.exp(-math.fabs(x1 - x2))


class TestHelperFunctions:
    @pytest.mark.parametrize(
        "A,B,expected",
        [
            (np.eye(2), np.eye(2), 2.0),
            (np.eye(2), np.zeros((2, 2)), 0.0),
            (np.array([[1.0, 2.3], [-1.3, 2.4]]), np.array([[0.7, -7.3], [-1.0, -2.9]]), -21.75),
        ],
    )
    def test_matrix_inner_product(self, A, B, expected):
        assert expected == pytest.approx(kern.cost_functions._matrix_inner_product(A, B))


class TestKernelMatrix:
    def test_simple_kernel(self):
        X = [0.1, 0.4]

        K_expected = np.array([[1, 0.2], [0.2, 1]])

        K = kern.kernel_matrix(X, lambda x1, x2: _mock_kernel(x1, x2, []))

        assert np.array_equal(K, K_expected)

    def test_laplace_kernel(self):
        X = [0.1, 0.4, 0.2]

        K_expected = np.exp(-np.array([[0.0, 0.3, 0.1], [0.3, 0.0, 0.2], [0.1, 0.2, 0.0]]))

        K = kern.kernel_matrix(X, _laplace_kernel, assume_normalized_kernel=False)

        assert np.array_equal(K, K_expected)


class TestKernelPolarization:
    def test_correct_calls(self):
        X = [0.1, 0.4]
        Y = [1, -1]

        hist = []

        kern.kernel_polarization(X, Y, lambda x1, x2: _mock_kernel(x1, x2, hist))

        assert len(hist) == 3

        assert (0.1, 0.4) in hist
        assert (0.1, 0.1) in hist
        assert (0.4, 0.4) in hist

    def test_correct_calls_normalized(self):
        X = [0.1, 0.4]
        Y = [1, -1]

        hist = []

        kern.kernel_polarization(
            X, Y, lambda x1, x2: _mock_kernel(x1, x2, hist), assume_normalized_kernel=True
        )

        assert len(hist) == 1

        assert (0.1, 0.4) in hist
        assert (0.1, 0.1) not in hist
        assert (0.4, 0.4) not in hist

    def test_polarization_value(self):
        X = [0.1, 0.4]
        Y = [1, -1]
        pol = kern.kernel_polarization(
            X, Y, lambda x1, x2: _mock_kernel(x1, x2, []), rescale_class_labels=False
        )
        pol_assume = kern.kernel_polarization(
            X,
            Y,
            lambda x1, x2: _mock_kernel(x1, x2, []),
            assume_normalized_kernel=True,
            rescale_class_labels=False,
        )

        assert pol == 1.6
        assert pol == pol_assume

    def test_polarization_value_other_labels(self):
        X = [0.1, 0.4]
        Y = [1, 1]
        pol = kern.kernel_polarization(
            X, Y, lambda x1, x2: _mock_kernel(x1, x2, []), rescale_class_labels=False
        )
        pol_assume = kern.kernel_polarization(
            X,
            Y,
            lambda x1, x2: _mock_kernel(x1, x2, []),
            assume_normalized_kernel=True,
            rescale_class_labels=False,
        )

        assert pol == 2.4
        assert pol == pol_assume


class TestKernelTargetAlignment:
    def test_correct_calls(self):
        X = [0.1, 0.4]
        Y = [1, -1]

        hist = []

        kern.kernel_target_alignment(X, Y, lambda x1, x2: _mock_kernel(x1, x2, hist))

        assert len(hist) == 3

        assert (0.1, 0.4) in hist
        assert (0.1, 0.1) in hist
        assert (0.4, 0.4) in hist

    def test_correct_calls_normalized(self):
        X = [0.1, 0.4]
        Y = [1, -1]

        hist = []

        kern.kernel_target_alignment(
            X, Y, lambda x1, x2: _mock_kernel(x1, x2, hist), assume_normalized_kernel=True
        )

        assert len(hist) == 1

        assert (0.1, 0.4) in hist
        assert (0.1, 0.1) not in hist
        assert (0.4, 0.4) not in hist

    def test_alignment_value(self):
        X = [0.1, 0.4]
        Y = [1, -1]

        alignment = kern.kernel_target_alignment(
            X, Y, lambda x1, x2: _mock_kernel(x1, x2, []), rescale_class_labels=False
        )
        alignment_assume = kern.kernel_target_alignment(
            X,
            Y,
            lambda x1, x2: _mock_kernel(x1, x2, []),
            assume_normalized_kernel=True,
            rescale_class_labels=False,
        )

        assert alignment == 1.6 / (2 * math.sqrt(2 + 2 * 0.2 ** 2))
        assert alignment == alignment_assume

    def test_alignment_value_other_labels(self):
        X = [0.1, 0.4]
        Y = [1, 1]
        alignment = kern.kernel_target_alignment(
            X, Y, lambda x1, x2: _mock_kernel(x1, x2, []), rescale_class_labels=False
        )
        alignment_assume = kern.kernel_target_alignment(
            X,
            Y,
            lambda x1, x2: _mock_kernel(x1, x2, []),
            assume_normalized_kernel=True,
            rescale_class_labels=False,
        )

        assert alignment == 2.4 / (2 * math.sqrt(2 + 2 * 0.2 ** 2))
        assert alignment == alignment_assume

    def test_alignment_value_three(self):
        X = [0.1, 0.4, 0.0]
        Y = [1, -1, 1]

        alignment = kern.kernel_target_alignment(
            X, Y, lambda x1, x2: _mock_kernel(x1, x2, []), rescale_class_labels=False
        )
        alignment_assume = kern.kernel_target_alignment(
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
        X = [0.1, 0.4, 0.0]
        Y = [1, -1, 1]

        alignment = kern.kernel_target_alignment(
            X, Y, lambda x1, x2: _mock_kernel(x1, x2, []), rescale_class_labels=True
        )
        alignment_assume = kern.kernel_target_alignment(
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


class TestPostprocessing:
    
    @pytest.mark.parametrize(
        "input,expected_output",
        [
            (np.diag([1, -1]), np.diag([1, 0])),
            (np.array([[1, 1], [1, -1]]), np.array([[1.20710678, 0.5], [0.5, 0.20710678]])),
            (np.array([[0, 1], [1, 0]]), np.array([[1, 1], [1, 1]]) / 2.0),
        ],
    )
    def test_threshold(self, input, expected_output):
        assert np.allclose(kern.threshold_matrix(input), expected_output)
    
    @pytest.mark.parametrize(
        "input,expected_output",
        [
            (np.diag([1, -1]), np.diag([2, 0])),
            (np.array([[1, 1], [1, -1]]), np.array([[1 + math.sqrt(2), 1], [1, -1 + math.sqrt(2)]])),
            (np.array([[0, 1], [1, 0]]), np.array([[1, 1], [1, 1]])),
        ],
    )
    def test_displacement(self, input, expected_output):
        assert np.allclose(kern.displace_matrix(input), expected_output)
