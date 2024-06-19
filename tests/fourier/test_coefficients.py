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
Unit tests for :mod:`fourier` coefficient and spectra calculations.
"""
# pylint: disable=import-outside-toplevel
from functools import partial

import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane.fourier import coefficients


def fourier_function(freq_dict, x):
    r"""Function of the form :math:`\sum c_n e^{inx}` to construct
    and evaluate a Fourier series given
    a dictionary of frequencies and their coefficients.

    Args:
        freq_dict (dict[int, float]): Pairs of positive integer frequencies and corresponding
            Fourier coefficients.

    Returns:
        float: output of the function
    """
    result = 0

    # Handle 0 coefficient separately
    if 0 in freq_dict.keys():
        result = freq_dict[0]

    _sum = sum(coeff * np.exp(1j * freq * x) for freq, coeff in freq_dict.items() if freq != 0)
    result += _sum + np.conj(_sum)
    return result.real


class TestExceptions:
    """Test that exceptions for ill-formatted inputs are raised."""

    # pylint: disable=unused-argument
    def dummy_fn(self, *args, **kwargs):
        """Dummy function that returns None."""

    @pytest.mark.parametrize("degree, n_inputs", [((2,), 2), ([3, 1, 2, 1], 2), ((1, 1), 1)])
    def test_wrong_number_of_degrees(self, degree, n_inputs):
        """Test that a ValueError is raised if the number of degrees does
        not match the number of inputs."""
        with pytest.raises(ValueError, match="If multiple degrees are provided, their number"):
            coefficients(self.dummy_fn, n_inputs, degree)

    @pytest.mark.parametrize("threshold, n_inputs", [((2,), 2), ([3, 1, 2, 1], 2), ((1, 1), 1)])
    def test_wrong_number_of_filter_thresholds(self, threshold, n_inputs):
        """Test that a ValueError is raised if the number of filtering thresholds does
        not match the number of inputs."""
        with pytest.raises(ValueError, match="If multiple filter_thresholds are provided"):
            coefficients(self.dummy_fn, n_inputs, 2, True, filter_threshold=threshold)


# pylint: disable=too-few-public-methods
class TestFourierCoefficientSingleVariable:
    """Test that the Fourier coefficients of a single-variable function are computed correctly"""

    @pytest.mark.parametrize(
        "freq_dict,expected_coeffs",
        [
            ({0: 0.5, 1: 2.3, 3: 0.4}, np.array([0.5, 2.3, 0, 0.4, 0.4, 0, 2.3])),
            (
                {3: 0.4, 4: 0.2 + 0.8j, 5: 1j},
                np.array([0, 0, 0, 0.4, 0.2 + 0.8j, 1j, -1j, 0.2 - 0.8j, 0.4, 0.0, 0]),
            ),
        ],
    )
    def test_single_variable_fourier_coeffs(self, freq_dict, expected_coeffs):
        """Test that the Fourier coefficients of a single-variable
        function are computed correctly"""
        degree = max(freq_dict.keys())
        partial_func = partial(fourier_function, freq_dict)
        # Testing with a single degree provided as integer
        coeffs = coefficients(partial_func, 1, degree)

        assert np.allclose(coeffs, expected_coeffs)
        # Testing with a single-entry sequence of degrees
        coeffs = coefficients(partial_func, 1, (degree,))

        assert np.allclose(coeffs, expected_coeffs)


dev_1 = qml.device("default.qubit", wires=1)
dev_2 = qml.device("default.qubit", wires=2)


@qml.qnode(dev_1)
def circuit_one_qubit_one_param_rx(inpt):
    r"""Circuit with a single-qubit, single-param, output function <Z>.

    By-hand calculation of :math:`f(x)` gives :math:`<Z> = cos^2(x/2) - sin^2(x/2) = cos(x)`.
    Fourier coeffs are :math:`c_1 = c_-1 = 0.5`.
    """
    qml.RX(inpt[0], wires=0)
    return qml.expval(qml.PauliZ(0))


circuit_one_qubit_one_param_rx.n_inputs = 1


@qml.qnode(dev_1)
def circuit_one_qubit_one_param_h_ry(inpt):
    r"""Circuit with a single-qubit, single-param, output function <Z>.

    By-hand calculation of :math:`f(x)` gives :math:`<Z> = -sin(x)`.
    Fourier coeffs are :math:`c_1 = 0.5i, c_-1 = -0.5i`.
    """
    qml.Hadamard(wires=0)
    qml.RY(inpt[0], wires=0)
    return qml.expval(qml.PauliZ(0))


circuit_one_qubit_one_param_h_ry.n_inputs = 1


@qml.qnode(dev_1)
def circuit_one_qubit_one_param_rx_ry(inpt):
    r"""Circuit with a single-qubit, single-param, output function <Z>.

    By-hand calculation of :math:`f(x)` gives :math:`<Z> = 1/2 + 1/2 cos(2x)`.
    Fourier coeffs are :math:`c_0 = 0.5, c_1 = c_-1 = 0, c_2 = c_-2 = 0.5`.
    """
    qml.RX(inpt[0], wires=0)
    qml.RY(inpt[0], wires=0)
    return qml.expval(qml.PauliZ(0))


circuit_one_qubit_one_param_rx_ry.n_inputs = 1


@qml.qnode(dev_1)
def circuit_one_qubit_two_params(inpt):
    r"""Circuit with a single-qubit, single-param, output function <Z>.

    By-hand calculation of :math:`f(x)` gives :math:`<Z> = cos(x_1) cos(x_2)`
    Fourier coeffs are 0.25 for all :math:`+/-1` combinations, 0 elsewhere.
    """
    qml.RY(inpt[0], wires=0)
    qml.RX(inpt[1], wires=0)
    return qml.expval(qml.PauliZ(0))


circuit_one_qubit_two_params.n_inputs = 2


@qml.qnode(dev_2)
def circuit_two_qubits_repeated_param(inpt):
    r"""Circuit with two qubits, repeated single-param output function :math:`<Z>`

    By-hand calculation of :math:`f(x)` gives :math:`<Z> = 1/2 + 1/2 cos(2x)`
    Fourier coeffs are :math:`c_0 = 0.5, c_1 = c_-1 = 0, c_2 = c_-2 = 0.25`
    (same as above circuit_one_qubit_one_param_rx_ry, just different qubits).
    """
    qml.RX(inpt[0], wires=0)
    qml.RY(inpt[0], wires=1)
    qml.CNOT(wires=[1, 0])
    return qml.expval(qml.PauliZ(0))


circuit_two_qubits_repeated_param.n_inputs = 1


@qml.qnode(dev_2)
def circuit_two_qubits_two_params(inpt):
    r"""Circuit with a single-qubit, two-param output function :math:`<Z>`.

    By-hand calculation of :math:`f(x)` gives :math:`<Z> = cos(x_1) cos(x_2)`
    Fourier coeffs are 0.25 for all :math:`+/-1` combinations, 0 elsewhere
    (Same as the circuit with one qubit and two params)
    """
    qml.RY(inpt[0], wires=0)
    qml.RX(inpt[1], wires=1)
    qml.CNOT(wires=[1, 0])
    return qml.expval(qml.PauliZ(0))


circuit_two_qubits_two_params.n_inputs = 2


@pytest.mark.parametrize("use_broadcasting", [False, True])
class TestFourierCoefficientCircuits:
    """Test calculation of Fourier coefficients for various circuits."""

    @pytest.mark.parametrize(
        "circuit,degree,expected_coeffs",
        [
            (circuit_one_qubit_one_param_rx, 1, np.array([0, 0.5, 0.5])),
            (circuit_one_qubit_one_param_rx, 2, np.array([0, 0.5, 0, 0, 0.5])),
            (circuit_one_qubit_one_param_h_ry, (1,), np.array([0, 0.5j, -0.5j])),
            (
                circuit_one_qubit_one_param_h_ry,
                3,
                np.array([0, 0.5j, 0, 0, 0, 0, -0.5j]),
            ),
            (
                circuit_one_qubit_one_param_rx_ry,
                2,
                np.array([0.5, 0, 0.25, 0.25, 0]),
            ),
            (
                circuit_one_qubit_one_param_rx_ry,
                4,
                np.array([0.5, 0, 0.25, 0, 0, 0, 0, 0.25, 0]),
            ),
            (
                circuit_two_qubits_repeated_param,
                (2,),
                np.array([0.5, 0, 0.25, 0.25, 0]),
            ),
            (
                circuit_two_qubits_repeated_param,
                3,
                np.array([0.5, 0, 0.25, 0, 0, 0.25, 0]),
            ),
        ],
    )
    def test_coefficients_one_param_circuits(
        self, circuit, degree, expected_coeffs, use_broadcasting
    ):
        """Test that coeffs for a single instance of a single parameter match the by-hand
        results regardless of input degree (max degree is 1)."""
        coeffs = coefficients(circuit, circuit.n_inputs, degree, use_broadcasting=use_broadcasting)
        assert np.allclose(coeffs, expected_coeffs)

    @pytest.mark.parametrize(
        "circuit,degree,expected_coeffs",
        [
            (
                circuit_two_qubits_two_params,
                1,
                np.array([[0, 0, 0], [0, 0.25, 0.25], [0, 0.25, 0.25]]),
            ),
            (
                circuit_two_qubits_two_params,
                (1, 1),
                np.array([[0, 0, 0], [0, 0.25, 0.25], [0, 0.25, 0.25]]),
            ),
            (
                circuit_one_qubit_two_params,
                1,
                np.array([[0, 0, 0], [0, 0.25, 0.25], [0, 0.25, 0.25]]),
            ),
            (
                circuit_one_qubit_two_params,
                (2, 1),
                np.array([[0, 0, 0], [0, 0.25, 0.25], [0, 0, 0], [0, 0, 0], [0, 0.25, 0.25]]),
            ),
        ],
    )
    def test_coefficients_two_param_circuits(
        self, circuit, degree, expected_coeffs, use_broadcasting
    ):
        """Test that coeffs for a single instance of a single parameter match the by-hand
        results regardless of input degree (max degree is 1)."""
        coeffs = coefficients(circuit, circuit.n_inputs, degree, use_broadcasting=use_broadcasting)
        assert np.allclose(coeffs, expected_coeffs)


class TestAntiAliasing:
    """Test that anti-aliasing techniques give correct results."""

    @pytest.mark.parametrize(
        "circuit,degree,expected_coeffs",
        [
            (
                circuit_two_qubits_repeated_param,
                1,
                np.array([0.5, 0, 0]),
            ),
        ],
    )
    def test_anti_aliasing_incorrect(self, circuit, degree, expected_coeffs):
        """Test that anti-aliasing function gives correct results when we ask for
        coefficients below the maximum degree."""
        coeffs_anti_aliased = coefficients(
            circuit, circuit.n_inputs, degree, lowpass_filter=True, filter_threshold=degree + 2
        )
        assert np.allclose(coeffs_anti_aliased, expected_coeffs)

        coeffs_regular = coefficients(circuit, circuit.n_inputs, degree)
        assert not np.allclose(coeffs_regular, expected_coeffs)

    @pytest.mark.parametrize(
        "circuit,degree,threshold",
        [
            (circuit_two_qubits_two_params, 1, None),
            (circuit_one_qubit_two_params, (1, 2), (2, 2)),
            (circuit_two_qubits_two_params, (1, 2), 3),
            (circuit_one_qubit_two_params, 1, (1, 2)),
        ],
    )
    def test_anti_aliasing(self, circuit, degree, threshold):
        """Test that the coefficients obtained through anti-aliasing are the
        same as the ones when we don't anti-alias at the correct degree."""
        coeffs_regular = coefficients(
            circuit,
            circuit.n_inputs,
            degree,
            lowpass_filter=False,
            filter_threshold=threshold,
        )
        coeffs_anti_aliased = coefficients(
            circuit,
            circuit.n_inputs,
            degree,
            lowpass_filter=True,
            filter_threshold=threshold,
        )

        assert np.allclose(coeffs_regular, coeffs_anti_aliased)


class TestInterfaces:
    """Test that coefficients are properly computed when QNodes use different interfaces."""

    @staticmethod
    def circuit(weights, inpt):
        """Testing circuit."""
        qml.RX(weights[0], wires=0)
        qml.RY(weights[1], wires=1)
        qml.RY(inpt[0], wires=0)
        qml.RX(inpt[1], wires=1)
        qml.CNOT(wires=[1, 0])
        return qml.expval(qml.PauliZ(0))

    dev = qml.device("default.qubit", wires=2)

    expected_result = np.array(
        [
            [0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j],
            [0.0 + 0.0j, 0.21502233 + 0.0j, 0.21502233 + 0.0j],
            [0.0 + 0.0j, 0.21502233 + 0.0j, 0.21502233 + 0.0j],
        ]
    )

    @pytest.mark.tf
    def test_coefficients_tf_interface(self):
        """Test that coefficients are correctly computed when using the Tensorflow interface."""
        import tensorflow as tf

        qnode = qml.QNode(self.circuit, self.dev)

        weights = tf.Variable([0.5, 0.2])

        obtained_result = coefficients(partial(qnode, weights), 2, 1)

        assert np.allclose(obtained_result, self.expected_result)

    @pytest.mark.torch
    def test_coefficients_torch_interface(self):
        """Test that coefficients are correctly computed when using the PyTorch interface."""
        import torch

        qnode = qml.QNode(self.circuit, self.dev)

        weights = torch.tensor([0.5, 0.2])

        obtained_result = coefficients(partial(qnode, weights), 2, 1)

        assert np.allclose(obtained_result, self.expected_result)

    @pytest.mark.jax
    def test_coefficients_jax_interface(self):
        """Test that coefficients are correctly computed when using the JAX interface."""
        import jax

        qnode = qml.QNode(self.circuit, self.dev, diff_method="parameter-shift")

        weights = jax.numpy.array([0.5, 0.2])

        obtained_result = coefficients(partial(qnode, weights), 2, 1)

        assert np.allclose(obtained_result, self.expected_result)
