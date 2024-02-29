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
Tests that application of gates and state preparations works correctly on a
device by checking expectation values.
"""
# pylint: disable=no-self-use
# pylint: disable=too-many-arguments
# pylint: disable=pointless-statement
from math import sqrt, pi

import pytest
import numpy as np
from flaky import flaky

import pennylane as qml

pytestmark = pytest.mark.skip_unsupported

np.random.seed(42)


# ===============================================================


@flaky(max_runs=10)
class TestGatesQubitExpval:
    """Test some expectation values obtained from qubit-based devices after
    application of gates."""

    # This test checks two Z expvals
    @pytest.mark.parametrize(
        "par,wires,expected_output",
        [([1, 1], [0, 1], [-1, -1]), ([1], [0], [-1, 1]), ([1], [1], [1, -1])],
    )
    def test_basis_state_2_qubit_subset(self, device, tol, par, wires, expected_output):
        """Tests qubit basis state preparation on subsets of qubits"""
        n_wires = 2
        dev = device(n_wires)

        @qml.qnode(dev)
        def circuit():
            qml.BasisState(np.array(par), wires=wires)
            return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))

        assert np.allclose(circuit(), expected_output, atol=tol(dev.shots))

    # This test checks three Z expvals
    @pytest.mark.parametrize(
        "par,wires,expected_output",
        [
            (
                [1j / np.sqrt(10), (1 - 2j) / np.sqrt(10), 0, 0, 0, 2 / np.sqrt(10), 0, 0],
                [0, 1, 2],
                [1 / 5.0, 1.0, -4 / 5.0],
            ),
            ([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], [0, 2], [0.0, 1.0, 0.0]),
            ([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], [0, 1], [0.0, 0.0, 1.0]),
            ([0, 1, 0, 0, 0, 0, 0, 0], [2, 1, 0], [-1.0, 1.0, 1.0]),
            ([0, 1j, 0, 0, 0, 0, 0, 0], [0, 2, 1], [1.0, -1.0, 1.0]),
            ([0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)], [1, 0], [-1.0, 0.0, 1.0]),
            ([0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2)], [0, 1], [0.0, -1.0, 1.0]),
        ],
    )
    def test_state_vector_3_qubit_subset(self, device, tol, par, wires, expected_output):
        """Tests qubit state vector preparation on subsets of 3 qubits"""

        n_wires = 3
        dev = device(n_wires)

        par = np.array(par)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(par, wires=wires)
            return qml.expval(qml.Z(0)), qml.expval(qml.Z(1)), qml.expval(qml.Z(2))

        assert np.allclose(circuit(), expected_output, atol=tol(dev.shots))

    # This test uses initial state |0> and checks one Z expval
    @pytest.mark.parametrize(
        "name,par,expected_output",
        [
            ("PhaseShift", [pi / 2], 1),
            ("PhaseShift", [-pi / 4], 1),
            ("RX", [pi / 2], 0),
            ("RX", [-pi / 4], 1 / sqrt(2)),
            ("RY", [pi / 2], 0),
            ("RY", [-pi / 4], 1 / sqrt(2)),
            ("RZ", [pi / 2], 1),
            ("RZ", [-pi / 4], 1),
            ("MultiRZ", [pi / 2], 1),
            ("MultiRZ", [-pi / 4], 1),
            ("Rot", [pi / 2, 0, 0], 1),
            ("Rot", [0, pi / 2, 0], 0),
            ("Rot", [0, 0, pi / 2], 1),
            ("Rot", [pi / 2, -pi / 4, -pi / 4], 1 / sqrt(2)),
            ("Rot", [-pi / 4, pi / 2, pi / 4], 0),
            ("Rot", [-pi / 4, pi / 4, pi / 2], 1 / sqrt(2)),
            (
                "QubitUnitary",
                [
                    np.array(
                        [
                            [1j / sqrt(2), 1j / sqrt(2)],
                            [1j / sqrt(2), -1j / sqrt(2)],
                        ]
                    )
                ],
                0,
            ),
            (
                "QubitUnitary",
                [
                    np.array(
                        [
                            [-1j / sqrt(2), 1j / sqrt(2)],
                            [1j / sqrt(2), 1j / sqrt(2)],
                        ]
                    )
                ],
                0,
            ),
        ],
    )
    def test_supported_gate_single_wire_with_parameters(
        self, device, tol, name, par, expected_output
    ):
        """Tests supported parametrized gates that act on a single wire"""

        n_wires = 1
        dev = device(n_wires)
        op = getattr(qml.ops, name)

        @qml.qnode(dev)
        def circuit():
            op(*par, wires=0)
            return qml.expval(qml.Z(0))

        assert np.isclose(circuit(), expected_output, atol=tol(dev.shots))

    # This test uses initial state 1/2|00>+sqrt(3)/2|11> and checks two Z expvals
    @pytest.mark.parametrize(
        "name,par,expected_output",
        [
            ("CRX", [0], [-1 / 2, -1 / 2]),
            ("CRX", [-pi], [-1 / 2, 1]),
            ("CRX", [pi / 2], [-1 / 2, 1 / 4]),
            ("CRY", [0], [-1 / 2, -1 / 2]),
            ("CRY", [-pi], [-1 / 2, 1]),
            ("CRY", [pi / 2], [-1 / 2, 1 / 4]),
            ("CRZ", [0], [-1 / 2, -1 / 2]),
            ("CRZ", [-pi], [-1 / 2, -1 / 2]),
            ("CRZ", [pi / 2], [-1 / 2, -1 / 2]),
            ("MultiRZ", [0], [-1 / 2, -1 / 2]),
            ("MultiRZ", [-pi], [-1 / 2, -1 / 2]),
            ("MultiRZ", [pi / 2], [-1 / 2, -1 / 2]),
            ("CRot", [pi / 2, 0, 0], [-1 / 2, -1 / 2]),
            ("CRot", [0, pi / 2, 0], [-1 / 2, 1 / 4]),
            ("CRot", [0, 0, pi / 2], [-1 / 2, -1 / 2]),
            ("CRot", [pi / 2, 0, -pi], [-1 / 2, -1 / 2]),
            ("CRot", [0, pi / 2, -pi], [-1 / 2, 1 / 4]),
            ("CRot", [-pi, 0, pi / 2], [-1 / 2, -1 / 2]),
            (
                "QubitUnitary",
                [
                    np.array(
                        [
                            [1, 0, 0, 0],
                            [0, 1 / sqrt(2), 1 / sqrt(2), 0],
                            [0, 1 / sqrt(2), -1 / sqrt(2), 0],
                            [0, 0, 0, 1],
                        ]
                    )
                ],
                [-1 / 2, -1 / 2],
            ),
            (
                "QubitUnitary",
                [
                    np.array(
                        [
                            [-1, 0, 0, 0],
                            [0, 1 / sqrt(2), 1 / sqrt(2), 0],
                            [0, 1 / sqrt(2), -1 / sqrt(2), 0],
                            [0, 0, 0, -1],
                        ]
                    )
                ],
                [-1 / 2, -1 / 2],
            ),
        ],
    )
    def test_supported_gate_two_wires_with_parameters(
        self, device, tol, name, par, expected_output
    ):
        """Tests supported parametrized gates that act on two wires"""

        n_wires = 2
        dev = device(n_wires)
        op = getattr(qml.ops, name)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(np.array([1 / 2, 0, 0, sqrt(3) / 2]), wires=[0, 1])
            op(*par, wires=[0, 1])
            return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))

        assert np.allclose(circuit(), expected_output, atol=tol(dev.shots))

    # This test uses initial state |0> and checks one Z expval
    @pytest.mark.parametrize(
        "name,expected_output",
        [
            ("PauliX", -1),
            ("PauliY", -1),
            ("PauliZ", 1),
            ("Hadamard", 0),
        ],
    )
    def test_supported_gate_single_wire_no_parameters(self, device, tol, name, expected_output):
        """Tests supported non-parametrized gates that act on a single wire"""
        n_wires = 1
        dev = device(n_wires)

        op = getattr(qml.ops, name)

        @qml.qnode(dev)
        def circuit():
            op(wires=0)
            return qml.expval(qml.Z(0))

        assert np.isclose(circuit(), expected_output, atol=tol(dev.shots))

    # This test uses initial state |Phi+> and checks two Z expvals
    @pytest.mark.parametrize(
        "name,expected_output",
        [
            ("CNOT", [-1 / 2, 1]),
            ("SWAP", [-1 / 2, -1 / 2]),
            ("CZ", [-1 / 2, -1 / 2]),
        ],
    )
    def test_supported_gate_two_wires_no_parameters(self, device, tol, name, expected_output):
        """Tests supported parametrized gates that act on two wires"""
        n_wires = 2
        dev = device(n_wires)

        op = getattr(qml.ops, name)
        if isinstance(dev, qml.Device) and not dev.supports_operation(op):
            pytest.skip("operation not supported")

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(np.array([1 / 2, 0, 0, sqrt(3) / 2]), wires=[0, 1])
            op(wires=[0, 1])
            return qml.expval(qml.Z(0)), qml.expval(qml.Z(1))

        assert np.allclose(circuit(), expected_output, atol=tol(dev.shots))

    @pytest.mark.parametrize(
        "name,expected_output",
        [
            ("CSWAP", [-1, -1, 1]),
        ],
    )
    def test_supported_gate_three_wires_no_parameters(self, device, tol, name, expected_output):
        """Tests supported non-parametrized gates that act on three wires"""
        n_wires = 3
        dev = device(n_wires)

        op = getattr(qml.ops, name)

        @qml.qnode(dev)
        def circuit():
            qml.BasisState(np.array([1, 0, 1]), wires=[0, 1, 2])
            op(wires=[0, 1, 2])
            return qml.expval(qml.Z(0)), qml.expval(qml.Z(1)), qml.expval(qml.Z(2))

        assert np.allclose(circuit(), expected_output, atol=tol(dev.shots))
