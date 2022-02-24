# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

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
Unit tests for the :mod:`pennylane.drawer.representation_resolver` module.
"""
from unittest.mock import Mock
import pytest
import numpy as np

import pennylane as qml
from pennylane.drawer.representation_resolver import RepresentationResolver
from pennylane.measurements import state


@pytest.fixture
def unicode_representation_resolver():
    """An instance of a RepresentationResolver with unicode charset."""
    return RepresentationResolver()


@pytest.fixture
def ascii_representation_resolver():
    """An instance of a RepresentationResolver with unicode charset."""
    return RepresentationResolver(charset=qml.drawer.charsets.AsciiCharSet)


def two_wire_quantum_tape():
    """An instance of a QuantumTape to be used as operation in a circuit."""
    with qml.tape.QuantumTape() as quantum_tape:
        qml.Hadamard(wires=0)
        qml.CNOT(wires=[0, 1])
    return quantum_tape


class TestRepresentationResolver:
    """Test the RepresentationResolver class."""

    @pytest.mark.parametrize(
        "list,element,index,list_after",
        [
            ([1, 2, 3], 2, 1, [1, 2, 3]),
            ([1, 2, 2, 3], 2, 1, [1, 2, 2, 3]),
            ([1, 2, 3], 4, 3, [1, 2, 3, 4]),
        ],
    )
    def test_index_of_array_or_append(self, list, element, index, list_after):
        """Test the method index_of_array_or_append."""

        assert RepresentationResolver.index_of_array_or_append(element, list) == index
        assert list == list_after

    @pytest.mark.parametrize(
        "par,expected",
        [
            (3, "3"),
            (5.236422, "5.24"),
        ],
    )
    def test_single_parameter_representation(self, unicode_representation_resolver, par, expected):
        """Test that single parameters are properly resolved."""
        assert unicode_representation_resolver.single_parameter_representation(par) == expected

    @pytest.mark.parametrize(
        "op,wire,target",
        [
            (qml.PauliX(wires=[1]), 1, "X"),
            (qml.CNOT(wires=[0, 1]), 1, "X"),
            (qml.CNOT(wires=[0, 1]), 0, "C"),
            (qml.Toffoli(wires=[0, 2, 1]), 1, "X"),
            (qml.Toffoli(wires=[0, 2, 1]), 0, "C"),
            (qml.Toffoli(wires=[0, 2, 1]), 2, "C"),
            (qml.CSWAP(wires=[0, 2, 1]), 1, "SWAP"),
            (qml.CSWAP(wires=[0, 2, 1]), 2, "SWAP"),
            (qml.CSWAP(wires=[0, 2, 1]), 0, "C"),
            (qml.PauliY(wires=[1]), 1, "Y"),
            (qml.PauliZ(wires=[1]), 1, "Z"),
            (qml.CZ(wires=[0, 1]), 1, "Z"),
            (qml.CZ(wires=[0, 1]), 0, "C"),
            (qml.Identity(wires=[1]), 1, "I"),
            (qml.Hadamard(wires=[1]), 1, "H"),
            (qml.PauliRot(3.14, "XX", wires=[0, 1]), 1, "RX(3.14)"),
            (qml.PauliRot(3.14, "YZ", wires=[0, 1]), 1, "RZ(3.14)"),
            (qml.PauliRot(3.14, "IXYZI", wires=[0, 1, 2, 3, 4]), 0, "RI(3.14)"),
            (qml.PauliRot(3.14, "IXYZI", wires=[0, 1, 2, 3, 4]), 1, "RX(3.14)"),
            (qml.PauliRot(3.14, "IXYZI", wires=[0, 1, 2, 3, 4]), 2, "RY(3.14)"),
            (qml.PauliRot(3.14, "IXYZI", wires=[0, 1, 2, 3, 4]), 3, "RZ(3.14)"),
            (qml.PauliRot(3.14, "IXYZI", wires=[0, 1, 2, 3, 4]), 4, "RI(3.14)"),
            (qml.MultiRZ(3.14, wires=[0, 1]), 0, "RZ(3.14)"),
            (qml.MultiRZ(3.14, wires=[0, 1]), 1, "RZ(3.14)"),
            (qml.CRX(3.14, wires=[0, 1]), 1, "RX(3.14)"),
            (qml.CRX(3.14, wires=[0, 1]), 0, "C"),
            (qml.CRY(3.14, wires=[0, 1]), 1, "RY(3.14)"),
            (qml.CRY(3.14, wires=[0, 1]), 0, "C"),
            (qml.CRZ(3.14, wires=[0, 1]), 1, "RZ(3.14)"),
            (qml.CRZ(3.14, wires=[0, 1]), 0, "C"),
            (qml.CRot(3.14, 2.14, 1.14, wires=[0, 1]), 1, "Rot(3.14, 2.14, 1.14)"),
            (qml.CRot(3.14, 2.14, 1.14, wires=[0, 1]), 0, "C"),
            (qml.PhaseShift(3.14, wires=[0]), 0, "Rϕ(3.14)"),
            (qml.Beamsplitter(1, 2, wires=[0, 1]), 1, "BS(1, 2)"),
            (qml.Beamsplitter(1, 2, wires=[0, 1]), 0, "BS(1, 2)"),
            (qml.Squeezing(1, 2, wires=[1]), 1, "S(1, 2)"),
            (qml.TwoModeSqueezing(1, 2, wires=[0, 1]), 1, "S(1, 2)"),
            (qml.TwoModeSqueezing(1, 2, wires=[0, 1]), 0, "S(1, 2)"),
            (qml.Displacement(1, 2, wires=[1]), 1, "D(1, 2)"),
            (qml.NumberOperator(wires=[1]), 1, "n"),
            (qml.Rotation(3.14, wires=[1]), 1, "R(3.14)"),
            (qml.ControlledAddition(3.14, wires=[0, 1]), 1, "X(3.14)"),
            (qml.ControlledAddition(3.14, wires=[0, 1]), 0, "C"),
            (qml.ControlledPhase(3.14, wires=[0, 1]), 1, "Z(3.14)"),
            (qml.ControlledPhase(3.14, wires=[0, 1]), 0, "C"),
            (qml.ThermalState(3, wires=[1]), 1, "Thermal(3)"),
            (
                qml.GaussianState(np.array([[2, 0], [0, 2]]), np.array([1, 2]), wires=[1]),
                1,
                "Gaussian(M0,M1)",
            ),
            (qml.QuadraticPhase(3.14, wires=[1]), 1, "P(3.14)"),
            (qml.RX(3.14, wires=[1]), 1, "RX(3.14)"),
            (qml.S(wires=[2]), 2, "S"),
            (qml.T(wires=[2]), 2, "T"),
            (qml.RX(3.14, wires=[1]), 1, "RX(3.14)"),
            (qml.RY(3.14, wires=[1]), 1, "RY(3.14)"),
            (qml.RZ(3.14, wires=[1]), 1, "RZ(3.14)"),
            (qml.Rot(3.14, 2.14, 1.14, wires=[1]), 1, "Rot(3.14, 2.14, 1.14)"),
            (qml.U1(3.14, wires=[1]), 1, "U1(3.14)"),
            (qml.U2(3.14, 2.14, wires=[1]), 1, "U2(3.14, 2.14)"),
            (qml.U3(3.14, 2.14, 1.14, wires=[1]), 1, "U3(3.14, 2.14, 1.14)"),
            (qml.BasisState(np.array([0, 1, 0]), wires=[1, 2, 3]), 1, "|0⟩"),
            (qml.BasisState(np.array([0, 1, 0]), wires=[1, 2, 3]), 2, "|1⟩"),
            (qml.BasisState(np.array([0, 1, 0]), wires=[1, 2, 3]), 3, "|0⟩"),
            (qml.QubitStateVector(np.array([0, 1, 0, 0]), wires=[1, 2]), 1, "QubitStateVector(M0)"),
            (qml.QubitStateVector(np.array([0, 1, 0, 0]), wires=[1, 2]), 2, "QubitStateVector(M0)"),
            (qml.QubitUnitary(np.eye(2), wires=[1]), 1, "U0"),
            (qml.QubitUnitary(np.eye(4), wires=[1, 2]), 2, "U0"),
            (qml.Kerr(3.14, wires=[1]), 1, "Kerr(3.14)"),
            (qml.CrossKerr(3.14, wires=[1, 2]), 1, "CrossKerr(3.14)"),
            (qml.CrossKerr(3.14, wires=[1, 2]), 2, "CrossKerr(3.14)"),
            (qml.CubicPhase(3.14, wires=[1]), 1, "V(3.14)"),
            (qml.InterferometerUnitary(np.eye(4), wires=[1, 3]), 1, "InterferometerUnitary(M0)"),
            (qml.InterferometerUnitary(np.eye(4), wires=[1, 3]), 3, "InterferometerUnitary(M0)"),
            (qml.CatState(3.14, 2.14, 1, wires=[1]), 1, "CatState(3.14, 2.14, 1)"),
            (qml.CoherentState(3.14, 2.14, wires=[1]), 1, "CoherentState(3.14, 2.14)"),
            (
                qml.FockDensityMatrix(np.kron(np.eye(4), np.eye(4)), wires=[1, 2]),
                1,
                "FockDensityMatrix(M0)",
            ),
            (
                qml.FockDensityMatrix(np.kron(np.eye(4), np.eye(4)), wires=[1, 2]),
                2,
                "FockDensityMatrix(M0)",
            ),
            (
                qml.DisplacedSqueezedState(3.14, 2.14, 1.14, 0.14, wires=[1]),
                1,
                "DisplacedSqueezedState(3.14, 2.14, 1.14, 0.14)",
            ),
            (qml.FockState(7, wires=[1]), 1, "|7⟩"),
            (qml.FockStateVector(np.array([4, 5, 7]), wires=[1, 2, 3]), 1, "|4⟩"),
            (qml.FockStateVector(np.array([4, 5, 7]), wires=[1, 2, 3]), 2, "|5⟩"),
            (qml.FockStateVector(np.array([4, 5, 7]), wires=[1, 2, 3]), 3, "|7⟩"),
            (qml.SqueezedState(3.14, 2.14, wires=[1]), 1, "SqueezedState(3.14, 2.14)"),
            (qml.Hermitian(np.eye(4), wires=[1, 2]), 1, "H0"),
            (qml.Hermitian(np.eye(4), wires=[1, 2]), 2, "H0"),
            (qml.X(wires=[1]), 1, "x"),
            (qml.P(wires=[1]), 1, "p"),
            (qml.FockStateProjector(np.array([4, 5, 7]), wires=[1, 2, 3]), 1, "|4,5,7╳4,5,7|"),
            (
                qml.PolyXP(np.array([1, 2, 0, -1.3, 6]), wires=[1]),
                2,
                "1+2x₀-1.3x₁+6p₁",
            ),
            (
                qml.PolyXP(
                    np.array([[1.2, 2.3, 4.5], [-1.2, 1.2, -1.5], [-1.3, 4.5, 2.3]]), wires=[1]
                ),
                1,
                "1.2+1.1x₀+3.2p₀+1.2x₀²+2.3p₀²+3x₀p₀",
            ),
            (
                qml.PolyXP(
                    np.array(
                        [
                            [1.2, 2.3, 4.5, 0, 0],
                            [-1.2, 1.2, -1.5, 0, 0],
                            [-1.3, 4.5, 2.3, 0, 0],
                            [0, 2.6, 0, 0, 0],
                            [0, 0, 0, -4.7, -1.0],
                        ]
                    ),
                    wires=[1],
                ),
                1,
                "1.2+1.1x₀+3.2p₀+1.2x₀²+2.3p₀²+3x₀p₀+2.6x₀x₁-p₁²-4.7x₁p₁",
            ),
            (qml.QuadOperator(3.14, wires=[1]), 1, "cos(3.14)x+sin(3.14)p"),
            (qml.PauliX(wires=[1]).inv(), 1, "X⁻¹"),
            (qml.CNOT(wires=[0, 1]).inv(), 1, "X⁻¹"),
            (qml.CNOT(wires=[0, 1]).inv(), 0, "C"),
            (qml.Toffoli(wires=[0, 2, 1]).inv(), 1, "X⁻¹"),
            (qml.Toffoli(wires=[0, 2, 1]).inv(), 0, "C"),
            (qml.Toffoli(wires=[0, 2, 1]).inv(), 2, "C"),
            (qml.measurements.sample(wires=[0, 1]), 0, "basis"),  # not providing an observable in
            (qml.measurements.sample(wires=[0, 1]), 1, "basis"),  # sample gets displayed as raw
            (two_wire_quantum_tape(), 0, "QuantumTape:T0"),
            (two_wire_quantum_tape(), 1, "QuantumTape:T0"),
        ],
    )
    def test_operator_representation_unicode(
        self, unicode_representation_resolver, op, wire, target
    ):
        """Test that an Operator instance is properly resolved."""
        assert unicode_representation_resolver.operator_representation(op, wire) == target

    @pytest.mark.parametrize(
        "op,wire,target",
        [
            (qml.PauliX(wires=[1]), 1, "X"),
            (qml.CNOT(wires=[0, 1]), 1, "X"),
            (qml.CNOT(wires=[0, 1]), 0, "C"),
            (qml.Toffoli(wires=[0, 2, 1]), 1, "X"),
            (qml.Toffoli(wires=[0, 2, 1]), 0, "C"),
            (qml.Toffoli(wires=[0, 2, 1]), 2, "C"),
            (qml.CSWAP(wires=[0, 2, 1]), 1, "SWAP"),
            (qml.CSWAP(wires=[0, 2, 1]), 2, "SWAP"),
            (qml.CSWAP(wires=[0, 2, 1]), 0, "C"),
            (qml.PauliY(wires=[1]), 1, "Y"),
            (qml.PauliZ(wires=[1]), 1, "Z"),
            (qml.CZ(wires=[0, 1]), 1, "Z"),
            (qml.CZ(wires=[0, 1]), 0, "C"),
            (qml.Identity(wires=[1]), 1, "I"),
            (qml.Hadamard(wires=[1]), 1, "H"),
            (qml.CRX(3.14, wires=[0, 1]), 1, "RX(3.14)"),
            (qml.CRX(3.14, wires=[0, 1]), 0, "C"),
            (qml.CRY(3.14, wires=[0, 1]), 1, "RY(3.14)"),
            (qml.CRY(3.14, wires=[0, 1]), 0, "C"),
            (qml.CRZ(3.14, wires=[0, 1]), 1, "RZ(3.14)"),
            (qml.CRZ(3.14, wires=[0, 1]), 0, "C"),
            (qml.CRot(3.14, 2.14, 1.14, wires=[0, 1]), 1, "Rot(3.14, 2.14, 1.14)"),
            (qml.CRot(3.14, 2.14, 1.14, wires=[0, 1]), 0, "C"),
            (qml.PhaseShift(3.14, wires=[0]), 0, "Rϕ(3.14)"),
            (qml.Beamsplitter(1, 2, wires=[0, 1]), 1, "BS(1, 2)"),
            (qml.Beamsplitter(1, 2, wires=[0, 1]), 0, "BS(1, 2)"),
            (qml.Squeezing(1, 2, wires=[1]), 1, "S(1, 2)"),
            (qml.TwoModeSqueezing(1, 2, wires=[0, 1]), 1, "S(1, 2)"),
            (qml.TwoModeSqueezing(1, 2, wires=[0, 1]), 0, "S(1, 2)"),
            (qml.Displacement(1, 2, wires=[1]), 1, "D(1, 2)"),
            (qml.NumberOperator(wires=[1]), 1, "n"),
            (qml.Rotation(3.14, wires=[1]), 1, "R(3.14)"),
            (qml.ControlledAddition(3.14, wires=[0, 1]), 1, "X(3.14)"),
            (qml.ControlledAddition(3.14, wires=[0, 1]), 0, "C"),
            (qml.ControlledPhase(3.14, wires=[0, 1]), 1, "Z(3.14)"),
            (qml.ControlledPhase(3.14, wires=[0, 1]), 0, "C"),
            (qml.ThermalState(3, wires=[1]), 1, "Thermal(3)"),
            (
                qml.GaussianState(np.array([[2, 0], [0, 2]]), np.array([1, 2]), wires=[1]),
                1,
                "Gaussian(M0,M1)",
            ),
            (qml.QuadraticPhase(3.14, wires=[1]), 1, "P(3.14)"),
            (qml.RX(3.14, wires=[1]), 1, "RX(3.14)"),
            (qml.S(wires=[2]), 2, "S"),
            (qml.T(wires=[2]), 2, "T"),
            (qml.RX(3.14, wires=[1]), 1, "RX(3.14)"),
            (qml.RY(3.14, wires=[1]), 1, "RY(3.14)"),
            (qml.RZ(3.14, wires=[1]), 1, "RZ(3.14)"),
            (qml.Rot(3.14, 2.14, 1.14, wires=[1]), 1, "Rot(3.14, 2.14, 1.14)"),
            (qml.U1(3.14, wires=[1]), 1, "U1(3.14)"),
            (qml.U2(3.14, 2.14, wires=[1]), 1, "U2(3.14, 2.14)"),
            (qml.U3(3.14, 2.14, 1.14, wires=[1]), 1, "U3(3.14, 2.14, 1.14)"),
            (qml.BasisState(np.array([0, 1, 0]), wires=[1, 2, 3]), 1, "|0>"),
            (qml.BasisState(np.array([0, 1, 0]), wires=[1, 2, 3]), 2, "|1>"),
            (qml.BasisState(np.array([0, 1, 0]), wires=[1, 2, 3]), 3, "|0>"),
            (qml.QubitStateVector(np.array([0, 1, 0, 0]), wires=[1, 2]), 1, "QubitStateVector(M0)"),
            (qml.QubitStateVector(np.array([0, 1, 0, 0]), wires=[1, 2]), 2, "QubitStateVector(M0)"),
            (qml.QubitUnitary(np.eye(2), wires=[1]), 1, "U0"),
            (qml.QubitUnitary(np.eye(4), wires=[1, 2]), 2, "U0"),
            (qml.Kerr(3.14, wires=[1]), 1, "Kerr(3.14)"),
            (qml.CrossKerr(3.14, wires=[1, 2]), 1, "CrossKerr(3.14)"),
            (qml.CrossKerr(3.14, wires=[1, 2]), 2, "CrossKerr(3.14)"),
            (qml.CubicPhase(3.14, wires=[1]), 1, "V(3.14)"),
            (qml.InterferometerUnitary(np.eye(4), wires=[1, 3]), 1, "InterferometerUnitary(M0)"),
            (qml.InterferometerUnitary(np.eye(4), wires=[1, 3]), 3, "InterferometerUnitary(M0)"),
            (qml.CatState(3.14, 2.14, 1, wires=[1]), 1, "CatState(3.14, 2.14, 1)"),
            (qml.CoherentState(3.14, 2.14, wires=[1]), 1, "CoherentState(3.14, 2.14)"),
            (
                qml.FockDensityMatrix(np.kron(np.eye(4), np.eye(4)), wires=[1, 2]),
                1,
                "FockDensityMatrix(M0)",
            ),
            (
                qml.FockDensityMatrix(np.kron(np.eye(4), np.eye(4)), wires=[1, 2]),
                2,
                "FockDensityMatrix(M0)",
            ),
            (
                qml.DisplacedSqueezedState(3.14, 2.14, 1.14, 0.14, wires=[1]),
                1,
                "DisplacedSqueezedState(3.14, 2.14, 1.14, 0.14)",
            ),
            (qml.FockState(7, wires=[1]), 1, "|7>"),
            (qml.FockStateVector(np.array([4, 5, 7]), wires=[1, 2, 3]), 1, "|4>"),
            (qml.FockStateVector(np.array([4, 5, 7]), wires=[1, 2, 3]), 2, "|5>"),
            (qml.FockStateVector(np.array([4, 5, 7]), wires=[1, 2, 3]), 3, "|7>"),
            (qml.SqueezedState(3.14, 2.14, wires=[1]), 1, "SqueezedState(3.14, 2.14)"),
            (qml.Hermitian(np.eye(4), wires=[1, 2]), 1, "H0"),
            (qml.Hermitian(np.eye(4), wires=[1, 2]), 2, "H0"),
            (qml.X(wires=[1]), 1, "x"),
            (qml.P(wires=[1]), 1, "p"),
            (qml.FockStateProjector(np.array([4, 5, 7]), wires=[1, 2, 3]), 1, "|4,5,7X4,5,7|"),
            (
                qml.PolyXP(np.array([1, 2, 0, -1.3, 6]), wires=[1]),
                2,
                "1+2x_0-1.3x_1+6p_1",
            ),
            (
                qml.PolyXP(
                    np.array([[1.2, 2.3, 4.5], [-1.2, 1.2, -1.5], [-1.3, 4.5, 2.3]]), wires=[1]
                ),
                1,
                "1.2+1.1x_0+3.2p_0+1.2x_0^2+2.3p_0^2+3x_0p_0",
            ),
            (
                qml.PolyXP(
                    np.array(
                        [
                            [1.2, 2.3, 4.5, 0, 0],
                            [-1.2, 1.2, -1.5, 0, 0],
                            [-1.3, 4.5, 2.3, 0, 0],
                            [0, 2.6, 0, 0, 0],
                            [0, 0, 0, -4.7, 0],
                        ]
                    ),
                    wires=[1],
                ),
                1,
                "1.2+1.1x_0+3.2p_0+1.2x_0^2+2.3p_0^2+3x_0p_0+2.6x_0x_1-4.7x_1p_1",
            ),
            (qml.QuadOperator(3.14, wires=[1]), 1, "cos(3.14)x+sin(3.14)p"),
            (qml.QuadOperator(3.14, wires=[1]), 1, "cos(3.14)x+sin(3.14)p"),
            (qml.PauliX(wires=[1]).inv(), 1, "X^-1"),
            (qml.CNOT(wires=[0, 1]).inv(), 1, "X^-1"),
            (qml.CNOT(wires=[0, 1]).inv(), 0, "C"),
            (qml.Toffoli(wires=[0, 2, 1]).inv(), 1, "X^-1"),
            (qml.Toffoli(wires=[0, 2, 1]).inv(), 0, "C"),
            (qml.Toffoli(wires=[0, 2, 1]).inv(), 2, "C"),
            (qml.measurements.sample(wires=[0, 1]), 0, "basis"),  # not providing an observable in
            (qml.measurements.sample(wires=[0, 1]), 1, "basis"),  # sample gets displayed as raw
            (two_wire_quantum_tape(), 0, "QuantumTape:T0"),
            (two_wire_quantum_tape(), 1, "QuantumTape:T0"),
        ],
    )
    def test_operator_representation_ascii(self, ascii_representation_resolver, op, wire, target):
        """Test that an Operator instance is properly resolved."""
        assert ascii_representation_resolver.operator_representation(op, wire) == target

    @pytest.mark.parametrize(
        "obs,wire,target",
        [
            (qml.expval(qml.PauliX(wires=[1])), 1, "⟨X⟩"),
            (qml.expval(qml.PauliY(wires=[1])), 1, "⟨Y⟩"),
            (qml.expval(qml.PauliZ(wires=[1])), 1, "⟨Z⟩"),
            (qml.expval(qml.Hadamard(wires=[1])), 1, "⟨H⟩"),
            (qml.expval(qml.Hermitian(np.eye(4), wires=[1, 2])), 1, "⟨H0⟩"),
            (qml.expval(qml.Hermitian(np.eye(4), wires=[1, 2])), 2, "⟨H0⟩"),
            (qml.expval(qml.NumberOperator(wires=[1])), 1, "⟨n⟩"),
            (qml.expval(qml.X(wires=[1])), 1, "⟨x⟩"),
            (qml.expval(qml.P(wires=[1])), 1, "⟨p⟩"),
            (
                qml.expval(qml.FockStateProjector(np.array([4, 5, 7]), wires=[1, 2, 3])),
                1,
                "⟨|4,5,7╳4,5,7|⟩",
            ),
            (
                qml.expval(qml.PolyXP(np.array([1, 2, 0, -1.3, 6]), wires=[1])),
                2,
                "⟨1+2x₀-1.3x₁+6p₁⟩",
            ),
            (
                qml.expval(
                    qml.PolyXP(
                        np.array([[1.2, 2.3, 4.5], [-1.2, 1.2, -1.5], [-1.3, 4.5, 2.3]]), wires=[1]
                    )
                ),
                1,
                "⟨1.2+1.1x₀+3.2p₀+1.2x₀²+2.3p₀²+3x₀p₀⟩",
            ),
            (qml.expval(qml.QuadOperator(3.14, wires=[1])), 1, "⟨cos(3.14)x+sin(3.14)p⟩"),
            (qml.var(qml.PauliX(wires=[1])), 1, "Var[X]"),
            (qml.var(qml.PauliY(wires=[1])), 1, "Var[Y]"),
            (qml.var(qml.PauliZ(wires=[1])), 1, "Var[Z]"),
            (qml.var(qml.Hadamard(wires=[1])), 1, "Var[H]"),
            (qml.var(qml.Hermitian(np.eye(4), wires=[1, 2])), 1, "Var[H0]"),
            (qml.var(qml.Hermitian(np.eye(4), wires=[1, 2])), 2, "Var[H0]"),
            (qml.var(qml.NumberOperator(wires=[1])), 1, "Var[n]"),
            (qml.var(qml.X(wires=[1])), 1, "Var[x]"),
            (qml.var(qml.P(wires=[1])), 1, "Var[p]"),
            (
                qml.var(qml.FockStateProjector(np.array([4, 5, 7]), wires=[1, 2, 3])),
                1,
                "Var[|4,5,7╳4,5,7|]",
            ),
            (
                qml.var(qml.PolyXP(np.array([1, 2, 0, -1.3, 6]), wires=[1])),
                2,
                "Var[1+2x₀-1.3x₁+6p₁]",
            ),
            (
                qml.var(
                    qml.PolyXP(
                        np.array([[1.2, 2.3, 4.5], [-1.2, 1.2, -1.5], [-1.3, 4.5, 2.3]]), wires=[1]
                    )
                ),
                1,
                "Var[1.2+1.1x₀+3.2p₀+1.2x₀²+2.3p₀²+3x₀p₀]",
            ),
            (qml.var(qml.QuadOperator(3.14, wires=[1])), 1, "Var[cos(3.14)x+sin(3.14)p]"),
            (qml.sample(qml.PauliX(wires=[1])), 1, "Sample[X]"),
            (qml.sample(qml.PauliY(wires=[1])), 1, "Sample[Y]"),
            (qml.sample(qml.PauliZ(wires=[1])), 1, "Sample[Z]"),
            (qml.sample(qml.Hadamard(wires=[1])), 1, "Sample[H]"),
            (qml.sample(qml.Hermitian(np.eye(4), wires=[1, 2])), 1, "Sample[H0]"),
            (qml.sample(qml.Hermitian(np.eye(4), wires=[1, 2])), 2, "Sample[H0]"),
            (qml.sample(qml.NumberOperator(wires=[1])), 1, "Sample[n]"),
            (qml.sample(qml.X(wires=[1])), 1, "Sample[x]"),
            (qml.sample(qml.P(wires=[1])), 1, "Sample[p]"),
            (
                qml.sample(qml.FockStateProjector(np.array([4, 5, 7]), wires=[1, 2, 3])),
                1,
                "Sample[|4,5,7╳4,5,7|]",
            ),
            (
                qml.sample(qml.PolyXP(np.array([1, 2, 0, -1.3, 6]), wires=[1])),
                2,
                "Sample[1+2x₀-1.3x₁+6p₁]",
            ),
            (
                qml.sample(
                    qml.PolyXP(
                        np.array([[1.2, 2.3, 4.5], [-1.2, 1.2, -1.5], [-1.3, 4.5, 2.3]]), wires=[1]
                    )
                ),
                1,
                "Sample[1.2+1.1x₀+3.2p₀+1.2x₀²+2.3p₀²+3x₀p₀]",
            ),
            (qml.sample(qml.QuadOperator(3.14, wires=[1])), 1, "Sample[cos(3.14)x+sin(3.14)p]"),
            (
                qml.expval(qml.PauliX(wires=[1]) @ qml.PauliY(wires=[2]) @ qml.PauliZ(wires=[3])),
                1,
                "⟨X ⊗ Y ⊗ Z⟩",
            ),
            (
                qml.expval(
                    qml.FockStateProjector(np.array([4, 5, 7]), wires=[1, 2, 3]) @ qml.X(wires=[4])
                ),
                1,
                "⟨|4,5,7╳4,5,7| ⊗ x⟩",
            ),
            (
                qml.expval(
                    qml.FockStateProjector(np.array([4, 5, 7]), wires=[1, 2, 3]) @ qml.X(wires=[4])
                ),
                2,
                "⟨|4,5,7╳4,5,7| ⊗ x⟩",
            ),
            (
                qml.expval(
                    qml.FockStateProjector(np.array([4, 5, 7]), wires=[1, 2, 3]) @ qml.X(wires=[4])
                ),
                3,
                "⟨|4,5,7╳4,5,7| ⊗ x⟩",
            ),
            (
                qml.expval(
                    qml.FockStateProjector(np.array([4, 5, 7]), wires=[1, 2, 3]) @ qml.X(wires=[4])
                ),
                4,
                "⟨|4,5,7╳4,5,7| ⊗ x⟩",
            ),
            (
                qml.sample(
                    qml.Hermitian(np.eye(4), wires=[1, 2]) @ qml.Hermitian(np.eye(4), wires=[0, 3])
                ),
                0,
                "Sample[H0 ⊗ H0]",
            ),
            (
                qml.sample(
                    qml.Hermitian(np.eye(4), wires=[1, 2])
                    @ qml.Hermitian(2 * np.eye(4), wires=[0, 3])
                ),
                0,
                "Sample[H0 ⊗ H1]",
            ),
            (qml.probs([0]), 0, "Probs"),
            (state(), 0, "State"),
        ],
    )
    def test_output_representation_unicode(
        self, unicode_representation_resolver, obs, wire, target
    ):
        """Test that an Observable instance with return type is properly resolved."""
        assert unicode_representation_resolver.output_representation(obs, wire) == target

    def test_fallback_output_representation_unicode(self, unicode_representation_resolver):
        """Test that an Observable instance with return type is properly resolved."""
        obs = qml.PauliZ(0)
        obs.return_type = "TestReturnType"

        assert unicode_representation_resolver.output_representation(obs, 0) == "TestReturnType[Z]"

    @pytest.mark.parametrize(
        "obs,wire,target",
        [
            (qml.expval(qml.PauliX(wires=[1])), 1, "<X>"),
            (qml.expval(qml.PauliY(wires=[1])), 1, "<Y>"),
            (qml.expval(qml.PauliZ(wires=[1])), 1, "<Z>"),
            (qml.expval(qml.Hadamard(wires=[1])), 1, "<H>"),
            (qml.expval(qml.Hermitian(np.eye(4), wires=[1, 2])), 1, "<H0>"),
            (qml.expval(qml.Hermitian(np.eye(4), wires=[1, 2])), 2, "<H0>"),
            (qml.expval(qml.NumberOperator(wires=[1])), 1, "<n>"),
            (qml.expval(qml.X(wires=[1])), 1, "<x>"),
            (qml.expval(qml.P(wires=[1])), 1, "<p>"),
            (
                qml.expval(qml.FockStateProjector(np.array([4, 5, 7]), wires=[1, 2, 3])),
                1,
                "<|4,5,7X4,5,7|>",
            ),
            (
                qml.expval(qml.PolyXP(np.array([1, 2, 0, -1.3, 6]), wires=[1])),
                2,
                "<1+2x_0-1.3x_1+6p_1>",
            ),
            (
                qml.expval(
                    qml.PolyXP(
                        np.array([[1.2, 2.3, 4.5], [-1.2, 1.2, -1.5], [-1.3, 4.5, 2.3]]), wires=[1]
                    )
                ),
                1,
                "<1.2+1.1x_0+3.2p_0+1.2x_0^2+2.3p_0^2+3x_0p_0>",
            ),
            (qml.expval(qml.QuadOperator(3.14, wires=[1])), 1, "<cos(3.14)x+sin(3.14)p>"),
            (qml.var(qml.PauliX(wires=[1])), 1, "Var[X]"),
            (qml.var(qml.PauliY(wires=[1])), 1, "Var[Y]"),
            (qml.var(qml.PauliZ(wires=[1])), 1, "Var[Z]"),
            (qml.var(qml.Hadamard(wires=[1])), 1, "Var[H]"),
            (qml.var(qml.Hermitian(np.eye(4), wires=[1, 2])), 1, "Var[H0]"),
            (qml.var(qml.Hermitian(np.eye(4), wires=[1, 2])), 2, "Var[H0]"),
            (qml.var(qml.NumberOperator(wires=[1])), 1, "Var[n]"),
            (qml.var(qml.X(wires=[1])), 1, "Var[x]"),
            (qml.var(qml.P(wires=[1])), 1, "Var[p]"),
            (
                qml.var(qml.FockStateProjector(np.array([4, 5, 7]), wires=[1, 2, 3])),
                1,
                "Var[|4,5,7X4,5,7|]",
            ),
            (
                qml.var(qml.PolyXP(np.array([1, 2, 0, -1.3, 6]), wires=[1])),
                2,
                "Var[1+2x_0-1.3x_1+6p_1]",
            ),
            (
                qml.var(
                    qml.PolyXP(
                        np.array([[1.2, 2.3, 4.5], [-1.2, 1.2, -1.5], [-1.3, 4.5, 2.3]]), wires=[1]
                    )
                ),
                1,
                "Var[1.2+1.1x_0+3.2p_0+1.2x_0^2+2.3p_0^2+3x_0p_0]",
            ),
            (qml.var(qml.QuadOperator(3.14, wires=[1])), 1, "Var[cos(3.14)x+sin(3.14)p]"),
            (qml.sample(qml.PauliX(wires=[1])), 1, "Sample[X]"),
            (qml.sample(qml.PauliY(wires=[1])), 1, "Sample[Y]"),
            (qml.sample(qml.PauliZ(wires=[1])), 1, "Sample[Z]"),
            (qml.sample(qml.Hadamard(wires=[1])), 1, "Sample[H]"),
            (qml.sample(qml.Hermitian(np.eye(4), wires=[1, 2])), 1, "Sample[H0]"),
            (qml.sample(qml.Hermitian(np.eye(4), wires=[1, 2])), 2, "Sample[H0]"),
            (qml.sample(qml.NumberOperator(wires=[1])), 1, "Sample[n]"),
            (qml.sample(qml.X(wires=[1])), 1, "Sample[x]"),
            (qml.sample(qml.P(wires=[1])), 1, "Sample[p]"),
            (
                qml.sample(qml.FockStateProjector(np.array([4, 5, 7]), wires=[1, 2, 3])),
                1,
                "Sample[|4,5,7X4,5,7|]",
            ),
            (
                qml.sample(qml.PolyXP(np.array([1, 2, 0, -1.3, 6]), wires=[1])),
                2,
                "Sample[1+2x_0-1.3x_1+6p_1]",
            ),
            (
                qml.sample(
                    qml.PolyXP(
                        np.array([[1.2, 2.3, 4.5], [-1.2, 1.2, -1.5], [-1.3, 4.5, 2.3]]), wires=[1]
                    )
                ),
                1,
                "Sample[1.2+1.1x_0+3.2p_0+1.2x_0^2+2.3p_0^2+3x_0p_0]",
            ),
            (qml.sample(qml.QuadOperator(3.14, wires=[1])), 1, "Sample[cos(3.14)x+sin(3.14)p]"),
            (
                qml.expval(qml.PauliX(wires=[1]) @ qml.PauliY(wires=[2]) @ qml.PauliZ(wires=[3])),
                1,
                "<X @ Y @ Z>",
            ),
            (
                qml.expval(
                    qml.FockStateProjector(np.array([4, 5, 7]), wires=[1, 2, 3]) @ qml.X(wires=[4])
                ),
                1,
                "<|4,5,7X4,5,7| @ x>",
            ),
            (
                qml.expval(
                    qml.FockStateProjector(np.array([4, 5, 7]), wires=[1, 2, 3]) @ qml.X(wires=[4])
                ),
                2,
                "<|4,5,7X4,5,7| @ x>",
            ),
            (
                qml.expval(
                    qml.FockStateProjector(np.array([4, 5, 7]), wires=[1, 2, 3]) @ qml.X(wires=[4])
                ),
                3,
                "<|4,5,7X4,5,7| @ x>",
            ),
            (
                qml.expval(
                    qml.FockStateProjector(np.array([4, 5, 7]), wires=[1, 2, 3]) @ qml.X(wires=[4])
                ),
                4,
                "<|4,5,7X4,5,7| @ x>",
            ),
            (
                qml.sample(
                    qml.Hermitian(np.eye(4), wires=[1, 2]) @ qml.Hermitian(np.eye(4), wires=[0, 3])
                ),
                0,
                "Sample[H0 @ H0]",
            ),
            (
                qml.sample(
                    qml.Hermitian(np.eye(4), wires=[1, 2])
                    @ qml.Hermitian(2 * np.eye(4), wires=[0, 3])
                ),
                0,
                "Sample[H0 @ H1]",
            ),
            (qml.probs([0]), 0, "Probs"),
            (state(), 0, "State"),
        ],
    )
    def test_output_representation_ascii(self, ascii_representation_resolver, obs, wire, target):
        """Test that an Observable instance with return type is properly resolved."""
        assert ascii_representation_resolver.output_representation(obs, wire) == target

    def test_element_representation_none(self, unicode_representation_resolver):
        """Test that element_representation properly handles None."""
        assert unicode_representation_resolver.element_representation(None, 0) == ""

    def test_element_representation_str(self, unicode_representation_resolver):
        """Test that element_representation properly handles strings."""
        assert unicode_representation_resolver.element_representation("Test", 0) == "Test"

    def test_element_representation_calls_output(self, unicode_representation_resolver):
        """Test that element_representation calls output_representation for returned observables."""

        unicode_representation_resolver.output_representation = Mock()

        obs = qml.sample(qml.PauliX(3))
        wire = 3

        unicode_representation_resolver.element_representation(obs, wire)

        assert unicode_representation_resolver.output_representation.call_args[0] == (obs, wire)

    def test_element_representation_calls_operator(self, unicode_representation_resolver):
        """Test that element_representation calls operator_representation for all operators that are not returned."""

        unicode_representation_resolver.operator_representation = Mock()

        op = qml.PauliX(3)
        wire = 3

        unicode_representation_resolver.element_representation(op, wire)

        assert unicode_representation_resolver.operator_representation.call_args[0] == (op, wire)
