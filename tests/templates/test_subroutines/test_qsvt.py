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
Tests for the QSVT template and qsvt wrapper function.
"""
import pytest
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates.subroutines.qsvt import *


class TestQSVT:
    """Test the qml.QSVT template."""

    @pytest.mark.parametrize(
        ("U_A", "lst_projectors", "wires", "expected_output"),
        [
            (
                qml.BlockEncode([[0.1, 0.2], [0.3, 0.4]], wires=[0, 1]),
                [qml.PCPhase(0.5, dim=2, wires=[0, 1]), qml.PCPhase(0.5, dim=2, wires=[0, 1])],
                [0, 1],
                -0.0687454719487044,
            ),
            (
                qml.BlockEncode([[0.3, 0.1], [0.2, 0.9]], wires=[0, 1]),
                [qml.PCPhase(0.5, dim=2, wires=[0, 1]), qml.PCPhase(0.3, dim=2, wires=[0, 1])],
                [0, 1],
                -0.14289392236365794,
            ),
        ],
    )
    def test_output(self, U_A, lst_projectors, wires, expected_output):
        """Test that qml.QSVT produces the intended measurements."""
        dev = qml.device("default.qubit", wires=len(wires))

        @qml.qnode(dev)
        def circuit():
            qml.QSVT(U_A, lst_projectors, wires)
            return qml.expval(qml.PauliY(wires=0))

        assert np.isclose(circuit(), expected_output)

    @pytest.mark.parametrize(
        ("U_A", "lst_projectors", "wires", "results"),
        [
            (
                qml.BlockEncode(0.1, wires=0),
                [qml.PCPhase(0.2, dim=1, wires=0), qml.PCPhase(0.3, dim=1, wires=0)],
                0,
                [
                    qml.PCPhase(0.2, dim=2, wires=[0]),
                    qml.BlockEncode(np.array([[0.1]]), wires=[0]),
                    qml.PCPhase(0.3, dim=2, wires=[0]),
                ],
            ),
            (
                qml.PauliZ(wires=0),
                [qml.RZ(0.1, wires=0), qml.RY(0.2, wires=0), qml.RZ(0.3, wires=1)],
                [0, 1],
                [
                    qml.RZ(0.1, wires=[0]),
                    qml.PauliZ(wires=[0]),
                    qml.RY(0.2, wires=[0]),
                    qml.adjoint(qml.PauliZ(wires=[0])),
                    qml.RZ(0.3, wires=[1]),
                ],
            ),
        ],
    )
    def test_queuing_ops(self, U_A, lst_projectors, wires, results):
        """Test that qml.QSVT queues operations in the correct order."""
        with qml.tape.QuantumTape() as tape:
            qml.QSVT(U_A, lst_projectors, wires)

        for idx, val in enumerate(tape.expand().operations):
            assert val.name == results[idx].name
            assert val.parameters == results[idx].parameters

    def test_queuing_ops_defined_in_circuit(self):
        """Test that qml.QSVT queues operations correctly when they are called in the qnode."""
        lst_projectors = [qml.PCPhase(0.2, dim=1, wires=0), qml.PCPhase(0.3, dim=1, wires=0)]
        wires = [0, 1]
        results = [
            qml.PCPhase(0.2, dim=1, wires=[0]),
            qml.PauliX(wires=[0]),
            qml.PCPhase(0.3, dim=1, wires=[0]),
        ]

        with qml.tape.QuantumTape() as tape:
            qml.QSVT(qml.PauliX(wires=0), lst_projectors, wires)

        for idx, val in enumerate(tape.expand().operations):
            assert val.name == results[idx].name
            assert val.parameters == results[idx].parameters

    def qfunc(A):
        """Used to test queuing in the next test."""
        return qml.PauliX(wires=0)

    def qfunc2(A):
        """Used to test queuing in the next test."""
        return qml.prod(qml.PauliX(wires=0), qml.RZ(A[0][0], wires=0))

    def lst_phis(phis):
        """Used to test queuing in the next test."""
        return [qml.PCPhase(i, 2, wires=[0, 1]) for i in phis]

    @pytest.mark.parametrize(
        ("quantum_function", "phi_func", "A", "phis", "results"),
        [
            (
                qfunc,
                lst_phis,
                np.array([[0.1, 0.2], [0.3, 0.4]]),
                np.array([0.2, 0.3]),
                [
                    qml.PCPhase(0.2, dim=2, wires=[0]),
                    qml.PauliX(wires=[0]),
                    qml.PCPhase(0.3, dim=2, wires=[0]),
                ],
            ),
            (
                qfunc2,
                lst_phis,
                np.array([[0.1, 0.2], [0.3, 0.4]]),
                np.array([0.1, 0.2]),
                [
                    qml.PCPhase(0.1, dim=2, wires=[0]),
                    qml.prod(qml.PauliX(wires=0), qml.RZ(0.1, wires=0)),
                    qml.PCPhase(0.2, dim=2, wires=[0]),
                ],
            ),
        ],
    )
    def test_queuing_callables(self, quantum_function, phi_func, A, phis, results):
        """Test that qml.QSVT queues operations correctly when a function is called inside the qnode"""

        with qml.tape.QuantumTape() as tape:
            qml.QSVT(quantum_function(A), phi_func(phis), wires=[0, 1])

        for idx, val in enumerate(tape.expand().operations):
            assert val.name == results[idx].name
            assert val.parameters == results[idx].parameters

    @pytest.mark.parametrize(
        ("UA", "A", "projectors", "phis", "wires", "output"),
        [
            (
                qml.BlockEncode([[0.2, 0.2], [0.1, 0.1]], wires=[0, 1]),
                [[0.2, 0.2], [0.1, 0.1]],
                [qml.PCPhase(0.2, dim=2, wires=[0, 1])],
                [0.2],
                [0, 1],
                0.1,
            )
        ],
    )
    def test_qsvt_grad(self, UA, A, projectors, phis, wires, output):
        def circuit(A, phis):
            with qml.queuing.QueuingManager.stop_recording():
                UA = qml.BlockEncode(A)
            qml.QSVT(UA(A), [projectors], wires=wires)
            return qml.expval(qml.PauliZ(wires=0))

        qml.grad(circuit)()


class Testqsvt:
    """Test the qml.qsvt function."""

    @pytest.mark.parametrize(
        ("A", "phis", "wires", "operations"),
        [
            (
                [[0.1, 0.2], [0.3, 0.4]],
                [0.2, 0.3],
                [0, 1],
                [
                    qml.PCPhase(0.2, dim=2, wires=[0, 1]),
                    qml.BlockEncode([[0.1, 0.2], [0.3, 0.4]], wires=[0, 1]),
                    qml.PCPhase(0.3, dim=2, wires=[0, 1]),
                ],
            ),
            (
                [[0.3, 0.1], [0.2, 0.9]],
                [0.1, 0.2, 0.3],
                [0, 1],
                [
                    qml.PCPhase(0.1, dim=2, wires=[0, 1]),
                    qml.BlockEncode([[0.3, 0.1], [0.2, 0.9]], wires=[0, 1]),
                    qml.PCPhase(0.2, dim=2, wires=[0, 1]),
                    qml.adjoint(qml.BlockEncode([[0.3, 0.1], [0.2, 0.9]], wires=[0, 1])),
                    qml.PCPhase(0.3, dim=2, wires=[0, 1]),
                ],
            ),
        ],
    )
    def test_output(self, A, phis, wires, operations):
        """Test that qml.qsvt produces the correct output."""
        dev = qml.device("default.qubit", wires=len(wires))

        @qml.qnode(dev)
        def circuit():
            qml.qsvt(A, phis, wires)
            return qml.expval(qml.PauliZ(wires=0))

        @qml.qnode(dev)
        def circuit_correct():
            for op in operations:
                qml.apply(op)
            return qml.expval(qml.PauliZ(wires=0))

        assert np.allclose(qml.matrix(circuit)(), qml.matrix(circuit_correct)())

    @pytest.mark.parametrize(
        ("A", "phis", "wires", "result"),
        [
            (
                [[0.1, 0.2], [0.3, 0.4]],
                [-1.520692517929803, 0.05010380886509347],
                [0, 1],
                0.01,
            ),  # angles from pyqsp give 0.1*x
            (
                0.3,
                [-0.8104500678299933, 1.520692517929803, 0.7603462589648997],
                [0],
                0.009,
            ),  # angles from pyqsp give 0.1*x**2
        ],
    )
    def test_output_wx(self, A, phis, wires, result):
        """Test that qml.qsvt produces the correct output."""
        dev = qml.device("default.qubit", wires=len(wires))

        @qml.qnode(dev)
        def circuit():
            qml.qsvt(A, phis, wires, convention="Wx")
            return qml.expval(qml.PauliZ(wires=0))

        assert np.isclose(np.real(qml.matrix(circuit)())[0][0], result, rtol=1e-3)

    @pytest.mark.torch
    @pytest.mark.parametrize(
        ("input_matrix", "angles", "wires"),
        [([[0.1, 0.2], [0.3, 0.4]], [0.1, 0.2], [0, 1])],
    )
    def test_qsvt_torch(self, input_matrix, angles, wires):
        """Test that the qsvt function matrix is correct for torch."""
        import torch

        default_matrix = qml.matrix(qml.qsvt(input_matrix, angles, wires))

        input_matrix = torch.tensor(input_matrix)
        angles = torch.tensor(angles)

        op = qml.qsvt(input_matrix, angles, wires)

        assert np.allclose(qml.matrix(op), default_matrix)
        assert qml.math.get_interface(qml.matrix(op)) == "torch"

    @pytest.mark.jax
    @pytest.mark.parametrize(
        ("input_matrix", "angles", "wires"),
        [([[0.1, 0.2], [0.3, 0.4]], [0.1, 0.2], [0, 1])],
    )
    def test_blockencode_jax(self, input_matrix, angles, wires):
        """Test that the qsvt function matrix is correct for jax."""
        import jax.numpy as jnp

        default_matrix = qml.matrix(qml.qsvt(input_matrix, angles, wires))

        input_matrix = jnp.array(input_matrix)
        angles = jnp.array(angles)

        op = qml.qsvt(input_matrix, angles, wires)

        assert np.allclose(qml.matrix(op), default_matrix)
        assert qml.math.get_interface(qml.matrix(op)) == "jax"
