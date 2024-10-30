# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the angle solver for QSP and QSVT."""

import numpy as np
import pytest

import pennylane as qml


@pytest.mark.parametrize(
    "P",
    [
        ([0.1, 0, 0.3, 0, -0.1]),
        ([0, 0.2, 0, 0.3]),
        ([-0.4, 0, 0.4, 0, -0.1, 0, 0.1]),
    ],
)
def test_complementary_polynomial(P):
    """Checks that |P(z)|^2 + |Q(z)|^2 = 1 in the unit circle"""

    Q = qml.math.complementary_poly(P)  # Calculate complementary polynomial Q

    # Define points on the unit circle
    theta_vals = np.linspace(0, 2 * np.pi, 100)
    unit_circle_points = np.exp(1j * theta_vals)

    for z in unit_circle_points:
        P_val = np.polyval(P, z)
        P_magnitude_squared = np.abs(P_val) ** 2

        Q_val = np.polyval(Q, z)
        Q_magnitude_squared = np.abs(Q_val) ** 2

        assert np.isclose(P_magnitude_squared + Q_magnitude_squared, 1, atol=1e-7)


@pytest.mark.parametrize(
    "angles",
    [
        ([0.1, 2, 0.3, 3, -0.1]),
        ([0, 0.2, 1, 0.3, 4, 2.4]),
        ([-0.4, 2, 0.4, 0, -0.1, 0, 0.1]),
    ],
)
def test_transform_angles(angles):
    """Test the transform_angles function"""

    new_angles = qml.math.transform_angles(angles, "QSP", "QSVT")
    assert np.allclose(angles, qml.math.transform_angles(new_angles, "QSVT", "QSP"))

    new_angles = qml.math.transform_angles(angles, "QSVT", "QSP")
    assert np.allclose(angles, qml.math.transform_angles(new_angles, "QSP", "QSVT"))


@pytest.mark.parametrize(
    "poly",
    [
        ([0.1, 0, 0.3, 0, -0.1]),
        ([0, 0.2, 0, 0.3]),
        ([-0.4, 0, 0.4, 0, -0.1, 0, 0.1]),
    ],
)
def test_correctness_QSP_angles(poly):
    """Tests that angles generate desired poly"""

    angles = qml.math.poly_to_angles(poly, "QSP")
    x = 0.5

    @qml.qnode(qml.device("default.qubit"))
    def circuit_qsp():

        qml.RX(2 * angles[0], wires=0)
        for ind, angle in enumerate(angles[1:]):
            qml.RZ(-2 * np.arccos(x), wires=0)
            qml.RX(2 * angle, wires=0)

        return qml.state()

    output = qml.matrix(circuit_qsp, wire_order=[0])()[0, 0]
    expected = sum(coef * (x**i) for i, coef in enumerate(poly))
    assert np.isclose(output.real, expected.real)


@pytest.mark.parametrize(
    "poly",
    [
        ([0.1, 0, 0.3, 0, -0.1]),
        ([0, 0.2, 0, 0.3]),
        ([-0.4, 0, 0.4, 0, -0.1, 0, 0.1]),
    ],
)
def test_correctness_QSVT_angles(poly):
    """Tests that angles generate desired poly"""

    angles = qml.math.poly_to_angles(poly, "QSVT")
    x = 0.5

    block_encoding = qml.RX(-2 * np.arccos(x), wires=0)
    projectors = [qml.PCPhase(angle, dim=1, wires=0) for angle in angles]

    @qml.qnode(qml.device("default.qubit"))
    def circuit_qsvt():
        qml.QSVT(block_encoding, projectors)
        return qml.state()

    output = qml.matrix(circuit_qsvt, wire_order=[0])()[0, 0]
    expected = sum(coef * (x**i) for i, coef in enumerate(poly))
    assert np.isclose(output.real, expected.real)


@pytest.mark.parametrize(
    ("poly", "msg_match"),
    [
        (
            [0.0, 1.0, 2.0],
            "Polynomial must have defined parity",
        ),
        (
            [0, 1j, 0, 3, 0, 2],
            "Array must not have an imaginary part",
        ),
    ],
)
def test_raise_error(poly, msg_match):
    """Test that proper errors are raised"""

    with pytest.raises(AssertionError, match=msg_match):
        _ = qml.math.poly_to_angles(poly, "QSVT")
