# Copyright 2021 Xanadu Quantum Technologies Inc.

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
Tests for the measurements_from_computational_basis transform
"""
from functools import partial

import numpy as np
import pytest

import pennylane as qml
from pennylane.devices.preprocess import validate_device_wires
from pennylane.transforms.measurements_from_computational_basis import (
    measurements_from_computational_basis,
)


class TestTransform:
    """Tests for transforms modifying measurements"""

    @pytest.mark.parametrize("from_counts", (True, False))
    def test_measurements_from_computational_basis(self, from_counts):
        """Test the measurements_from_computational_basis transform with a single measurement"""

        dev = qml.device("lightning.qubit", wires=4, shots=1000)

        @qml.qnode(dev)
        def circuit(theta: float):
            qml.RX(theta, 0)
            qml.RX(theta / 2, 1)
            qml.RX(theta / 3, 2)
            return qml.expval(qml.Y(0))

        transformed_circuit = measurements_from_computational_basis(circuit, from_counts)

        theta = 1.2
        expected = circuit(theta)
        res = transformed_circuit(theta)

        assert np.allclose(expected, res, atol=0.05)

    # pylint: disable=unnecessary-lambda
    @pytest.mark.parametrize("from_counts", (True, False))
    @pytest.mark.parametrize(
        "input_measurement, expected_res",
        [
            (
                lambda: qml.expval(qml.PauliY(wires=0) @ qml.PauliY(wires=1)),
                lambda theta: np.sin(theta) * np.sin(theta / 2),
            ),
            (lambda: qml.var(qml.Y(wires=1)), lambda theta: 1 - np.sin(theta / 2) ** 2),
            (
                lambda: qml.probs(),
                lambda theta: np.outer(
                    np.outer(
                        [np.cos(theta / 2) ** 2, np.sin(theta / 2) ** 2],
                        [np.cos(theta / 4) ** 2, np.sin(theta / 4) ** 2],
                    ),
                    [1, 0, 0, 0],
                ).flatten(),
            ),
            (
                lambda: qml.probs(wires=[1]),
                lambda theta: [np.cos(theta / 4) ** 2, np.sin(theta / 4) ** 2],
            ),
        ],
    )
    @pytest.mark.parametrize("shots", [3000, (3000, 4000), (3000, 3500, 4000)])
    def test_measurements_from_computational_basis_analytic(
        self,
        from_counts,
        input_measurement,
        expected_res,
        shots,
    ):
        """Test the test_measurements_from_computational_basis transform with a single measurement,
        for measurements whose outcome can be directly compared to an expected analytic result."""

        dev = qml.device("default.qubit", wires=4, shots=shots)

        @partial(measurements_from_computational_basis, from_counts=from_counts)
        @partial(validate_device_wires, wires=dev.wires)
        @qml.qnode(dev)
        def circuit(theta: float):
            qml.RX(theta, 0)
            qml.RX(theta / 2, 1)
            return input_measurement()

        theta = 2.5
        res = circuit(theta)

        if len(dev.shots.shot_vector) != 1:
            assert len(res) == len(dev.shots.shot_vector)

        assert np.allclose(res, expected_res(theta), atol=0.05)


#     # pylint: disable=unnecessary-lambda
#     @pytest.mark.parametrize(
#         "input_measurement, expected_res",
#         [
#             (
#                 lambda: qml.expval(qml.PauliY(wires=0) @ qml.PauliY(wires=1)),
#                 lambda theta: np.sin(theta) * np.sin(theta / 2),
#             ),
#             (lambda: qml.var(qml.Y(wires=1)), lambda theta: 1 - np.sin(theta / 2) ** 2),
#             (
#                 lambda: qml.probs(),
#                 lambda theta: np.outer(
#                     np.outer(
#                         [np.cos(theta / 2) ** 2, np.sin(theta / 2) ** 2],
#                         [np.cos(theta / 4) ** 2, np.sin(theta / 4) ** 2],
#                     ),
#                     [1, 0, 0, 0],
#                 ).flatten(),
#             ),
#             (
#                 lambda: qml.probs(wires=[1]),
#                 lambda theta: [np.cos(theta / 4) ** 2, np.sin(theta / 4) ** 2],
#             ),
#         ],
#     )
#     def test_measurement_from_counts_single_measurement_analytic(
#         self, input_measurement, expected_res
#     ):
#         """Test the measurment_from_counts transform with a single measurements as part of the
#         Catalyst pipeline, for measurements whose outcome can be directly compared to an expected
#         analytic result."""
#
#         dev = qml.device("lightning.qubit", wires=4, shots=3000)
#
#         @qml.qjit
#         @partial(measurements_from_counts, device_wires=dev.wires)
#         @qml.qnode(dev)
#         def circuit(theta: float):
#             qml.RX(theta, 0)
#             qml.RX(theta / 2, 1)
#             return input_measurement()
#
#         mlir = qml.qjit(circuit, target="mlir").mlir
#         assert "expval" not in mlir
#         assert "counts" in mlir
#
#         theta = 2.5
#         res = circuit(theta)
#
#         if len(dev.shots.shot_vector) != 1:
#             assert len(res) == len(dev.shots.shot_vector)
#
#         assert np.allclose(res, expected_res(theta), atol=0.05)
#
#     def test_measurement_from_counts_raises_not_implemented(self):
#         """Test that an measurement not supported by the measurements_from_counts or
#         measurements_from_samples transform raises a NotImplementedError"""
#
#         dev = qml.device("lightning.qubit", wires=4, shots=1000)
#
#         @partial(measurements_from_counts, device_wires=dev.wires)
#         @qml.qnode(dev)
#         def circuit(theta: float):
#             qml.RX(theta, 0)
#             return qml.sample()
#
#         with pytest.raises(
#             NotImplementedError, match="not implemented with measurements_from_counts"
#         ):
#             qml.qjit(circuit)
#
#     def test_measurement_from_samples_raises_not_implemented(self):
#         """Test that an measurement not supported by the measurements_from_counts or
#         measurements_from_samples transform raises a NotImplementedError"""
#
#         dev = qml.device("lightning.qubit", wires=4, shots=1000)
#
#         @partial(measurements_from_samples, device_wires=dev.wires)
#         @qml.qnode(dev)
#         def circuit(theta: float):
#             qml.RX(theta, 0)
#             return qml.counts()
#
#         with pytest.raises(
#             NotImplementedError, match="not implemented with measurements_from_samples"
#         ):
#             qml.qjit(circuit)
#
#
#
#
# class TestIntegration():
#
#     def test_measurements_from_counts_multiple_measurements(self):
#         """Test the transforms for measurements_from_counts to other measurement types
#         as part of the Catalyst pipeline."""
#
#         dev = qml.device("default.qubit", wires=4, shots=5000)
#
#         @
#         @qml.qnode(dev)
#         def basic_circuit(theta: float):
#             qml.RY(theta, 0)
#             qml.RY(theta / 2, 1)
#             qml.RY(2 * theta, 2)
#             qml.RY(theta, 3)
#             return (
#                 qml.expval(qml.PauliX(wires=0) @ qml.PauliX(wires=1)),
#                 qml.var(qml.PauliX(wires=2)),
#                 qml.counts(qml.PauliX(wires=0) @ qml.PauliX(wires=1) @ qml.PauliX(wires=2)),
#                 qml.probs(wires=[3]),
#             )
#
#         transformed_circuit = measurements_from_counts(basic_circuit, dev.wires)
#
#         theta = 1.9
#         expval_res, var_res, counts_res, probs_res = qml.qjit(transformed_circuit)(theta)
#
#         expval_expected = np.sin(theta) * np.sin(theta / 2)
#         var_expected = 1 - np.sin(2 * theta) ** 2
#         counts_expected = basic_circuit(theta)[2]
#         probs_expected = [np.cos(theta / 2) ** 2, np.sin(theta / 2) ** 2]
#
#         assert np.isclose(expval_res, expval_expected, atol=0.05)
#         assert np.isclose(var_res, var_expected, atol=0.05)
#         assert np.allclose(probs_res, probs_expected, atol=0.05)
#
#         # counts comparison by converting catalyst format to PL style eigvals dict
#         basis_states, counts = counts_res
#         num_excitations_per_state = [
#             sum(int(i) for i in format(int(state), "01b")) for state in basis_states
#         ]
#         eigvals = [(-1) ** i for i in num_excitations_per_state]
#         eigval_counts_res = {
#             -1.0: sum(count for count, eigval in zip(counts, eigvals) if eigval == -1),
#             1.0: sum(count for count, eigval in zip(counts, eigvals) if eigval == 1),
#         }
#
#         # +/- 100 shots is pretty reasonable with 3000 shots total
#         assert np.isclose(eigval_counts_res[-1], counts_expected[-1], atol=100)
#         assert np.isclose(eigval_counts_res[1], counts_expected[1], atol=100)
#
#     def test_measurements_from_samples_multiple_measurements(self):
#         """Test the transform measurements_from_samples with multiple measurement types
#         as part of the Catalyst pipeline."""
#
#         dev = qml.device("lightning.qubit", wires=4, shots=5000)
#
#         @qml.qnode(dev)
#         def basic_circuit(theta: float):
#             qml.RY(theta, 0)
#             qml.RY(theta / 2, 1)
#             qml.RY(2 * theta, 2)
#             qml.RY(theta, 3)
#             return (
#                 qml.expval(qml.PauliX(wires=0) @ qml.PauliX(wires=1)),
#                 qml.var(qml.PauliX(wires=2)),
#                 qml.sample(qml.PauliX(wires=0) @ qml.PauliX(wires=1) @ qml.PauliX(wires=2)),
#                 qml.probs(wires=[3]),
#             )
#
#         transformed_circuit = measurements_from_samples(basic_circuit, dev.wires)
#
#         mlir = qml.qjit(transformed_circuit, target="mlir").mlir
#         assert "expval" not in mlir
#         assert "quantum.var" not in mlir
#         assert "sample" in mlir
#
#         theta = 1.9
#
#         expval_res, var_res, sample_res, probs_res = qml.qjit(transformed_circuit)(theta)
#
#         expval_expected = np.sin(theta) * np.sin(theta / 2)
#         var_expected = 1 - np.sin(2 * theta) ** 2
#         sample_expected = basic_circuit(theta)[2]
#         probs_expected = [np.cos(theta / 2) ** 2, np.sin(theta / 2) ** 2]
#
#         assert np.isclose(expval_res, expval_expected, atol=0.05)
#         assert np.isclose(var_res, var_expected, atol=0.05)
#         assert np.allclose(probs_res, probs_expected, atol=0.05)
#
#         # sample comparison
#         assert np.isclose(np.mean(sample_res), np.mean(sample_expected), atol=0.05)
#         assert len(sample_res) == len(sample_expected)
#         assert set(np.array(sample_res)) == set(sample_expected)
