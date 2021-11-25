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
"""Tests for the gradients.param_shift_hessian module."""

import math
from autograd.differential_operators import jacobian
import pennylane as qml
from pennylane import numpy as np
from pennylane.gradients import param_shift_hessian
from pennylane.ops.qubit.non_parametric_ops import PauliZ


class TestParameterShiftHessian:
    """Tests for the param_shift_hessian method"""

    def test_2term_shift_rules1(self):
        """Test that the correct hessian is calculated for a single RX operator"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0))

        x = np.array(0.1, requires_grad=True)

        jacobian = qml.jacobian(qml.grad(circuit))(x)
        hessian = qml.gradients.param_shift_hessian(circuit)(x)

        print("\n", jacobian, "=?", hessian)

        assert np.allclose(jacobian, hessian)

    def test_2term_shift_rules2(self):
        """Test that the correct hessian is calculated for a single RY operator"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.RY(x, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=[0, 1])

        x = np.array(0.1, requires_grad=True)

        jacobian = qml.jacobian(qml.jacobian(circuit))(x)
        hessian = qml.gradients.param_shift_hessian(circuit)(x)

        print("\n", jacobian, "\n\t=?\n", hessian)

        assert np.allclose(jacobian, hessian)

    def test_2term_shift_rules3(self):
        """Test that the correct hessian is calculated for two 2-term shift rule operators"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(1))

        x = np.array([0.1, 0.2], requires_grad=True)

        jacobian = qml.jacobian(qml.jacobian(circuit))(x)
        hessian = qml.gradients.param_shift_hessian(circuit)(x)

        print("\n", jacobian, "\n\t=?\n", hessian)

        assert np.allclose(jacobian, hessian)

    def test_2term_shift_rules4(self):
        """Test that the correct hessian is calculated for two 2-term shift rule operators"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=1)

        x = np.array([0.1, 0.2], requires_grad=True)

        jacobian = qml.jacobian(qml.jacobian(circuit))(x)
        hessian = qml.gradients.param_shift_hessian(circuit)(x)

        print("\n", jacobian, "\n\t=?\n", hessian)

        assert np.allclose(jacobian, hessian)

    def test_2term_shift_rules5(self):
        """Test that the purely "quantum" hessian has the correct shape"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RZ(x[2], wires=1)
            qml.Rot(x[0], x[1], x[2], wires=1)
            return qml.probs(wires=0)

        x = np.array([0.1, 0.2, 0.3], requires_grad=True)
        shape = (2, 6, 6)  # (num_output_vals, num_gate_args, num_gate_args)

        hessian = qml.gradients.param_shift_hessian(circuit, hybrid=False)(x)

        print("\n", hessian)

        assert qml.math.shape(hessian) == shape

    def test_2term_shift_rules6(self):
        """Test that the correct hessian is calculated when reusing parameters"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RZ(x[2], wires=1)
            qml.Rot(x[0], x[1], x[2], wires=1)
            return qml.probs(wires=0)

        x = np.array([0.1, 0.2, 0.3], requires_grad=True)

        jacobian = qml.jacobian(qml.jacobian(circuit))(x)
        hessian = qml.gradients.param_shift_hessian(circuit)(x)

        print("\n", jacobian, "\n\t=?\n", hessian)

        assert np.allclose(jacobian, hessian)

    def test_2term_shift_rules7(self):
        """Test that the correct hessian is calculated when manipulating parameters"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.RX(x[0] + x[1] + x[2], wires=0)
            qml.RY(x[1] - x[0] + 3 * x[2], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RZ(x[2] / x[0] - x[1], wires=1)
            return qml.probs(wires=0)

        x = np.array([0.1, 0.2, 0.3], requires_grad=True)

        jacobian = qml.jacobian(qml.jacobian(circuit))(x)
        hessian = qml.gradients.param_shift_hessian(circuit)(x)

        print("\n", jacobian, "\n\t=?\n", hessian)

        assert np.allclose(jacobian, hessian)

    def test_2term_shift_rules8(self):
        """Test that the correct hessian is calculated for higher dimensional QNode outputs"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.probs(wires=0), qml.probs(wires=1)

        x = np.ones([2], requires_grad=True)

        jacobian = qml.jacobian(qml.jacobian(circuit))(x)
        hessian = qml.gradients.param_shift_hessian(circuit)(x)

        print("\n", jacobian, "\n\t=?\n", hessian)

        assert np.allclose(jacobian, hessian)

    def test_2term_shift_rules9(self):
        """Test that the correct hessian is calculated for higher dimensional cl. jacobians"""
        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.RX(x[0, 0], wires=0)
            qml.RY(x[0, 1], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RX(x[0, 2], wires=0)
            qml.RY(x[0, 0], wires=0)
            return qml.probs(wires=0), qml.probs(wires=1)

        x = np.ones([1, 3], requires_grad=True)

        jacobian = qml.jacobian(qml.jacobian(circuit))(x)
        hessian = qml.gradients.param_shift_hessian(circuit)(x)

        print("\n", jacobian, "\n\t=?\n", hessian)

        assert np.allclose(jacobian, hessian)

    # Some bounds we could choose to meet on the efficiency of the hessian implementation
    # for operations with two eigenvalues (2-term shift rule):
    # - < jacobian(jacobian())
    # - <= 2^d * (m+d-1)C(d)      see arXiv:2008.06517 p. 4
    # - <= 3^m                    see arXiv:2008.06517 p. 4
    # here d=2 is the derivative order, m is the number of variational parameters (w.r.t. gate args)

    def test_less_quantum_invocations1(self):
        """Test that the hessian invokes less hardware executions than double differentiation"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.RX(x, wires=0)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(1))

        x = np.array(0.1, requires_grad=True)

        with qml.Tracker(dev) as tracker:
            hessian = qml.gradients.param_shift_hessian(circuit)(x)
            hessian_qruns = tracker.totals["executions"]
            jacobian = qml.jacobian(qml.jacobian(circuit))(x)
            jacobian_qruns = tracker.totals["executions"] - hessian_qruns

        print("\n", hessian_qruns, "<", jacobian_qruns, "?")
        print("\n", hessian_qruns, "<=", 2 ** 2 * 1, "?")
        print("\n", hessian_qruns, "<=", 3 ** 1, "?")

        assert np.allclose(hessian, jacobian)
        assert hessian_qruns < jacobian_qruns
        assert hessian_qruns <= 2 ** 2 * 1  # 1 = (1+2-1)C(2)
        assert hessian_qruns <= 3 ** 1

    def test_less_quantum_invocations2(self):
        """Test that the hessian invokes less hardware executions than double differentiation"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RY(x[1], wires=0)
            return qml.expval(qml.PauliZ(1))

        x = np.array([0.1, 0.2], requires_grad=True)

        with qml.Tracker(dev) as tracker:
            hessian = qml.gradients.param_shift_hessian(circuit)(x)
            hessian_qruns = tracker.totals["executions"]
            jacobian = qml.jacobian(qml.jacobian(circuit))(x)
            jacobian_qruns = tracker.totals["executions"] - hessian_qruns

        print("\n", hessian_qruns, "<", jacobian_qruns, "?")
        print("\n", hessian_qruns, "<=", 2 ** 2 * 3, "?")
        print("\n", hessian_qruns, "<=", 3 ** 2, "?")

        assert np.allclose(hessian, jacobian)
        assert hessian_qruns < jacobian_qruns
        assert hessian_qruns <= 2 ** 2 * 3  # 3 = (2+2-1)C(2)
        assert hessian_qruns <= 3 ** 2

    def test_less_quantum_invocations3(self):
        """Test that the hessian invokes less hardware executions than double differentiation"""

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev, diff_method="parameter-shift", max_diff=2)
        def circuit(x):
            qml.RX(x[0], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.RY(x[1], wires=0)
            qml.RZ(x[2], wires=1)
            return qml.probs(wires=[0, 1])

        x = np.array([0.1, 0.2, 0.3], requires_grad=True)

        with qml.Tracker(dev) as tracker:
            hessian = qml.gradients.param_shift_hessian(circuit)(x)
            hessian_qruns = tracker.totals["executions"]
            jacobian = qml.jacobian(qml.jacobian(circuit))(x)
            jacobian_qruns = tracker.totals["executions"] - hessian_qruns

        print("\n", hessian_qruns, "<", jacobian_qruns, "?")
        print("\n", hessian_qruns, "<=", 2 ** 2 * 6, "?")
        print("\n", hessian_qruns, "<=", 3 ** 3, "?")

        assert np.allclose(hessian, jacobian)
        assert hessian_qruns < jacobian_qruns
        assert hessian_qruns <= 2 ** 2 * 6  # 6 = (3+2-1)C(2)
        assert hessian_qruns <= 3 ** 3
