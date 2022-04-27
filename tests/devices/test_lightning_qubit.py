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
Tests for the accessibility of the Lightning-Qubit device
"""
import pennylane as qml
from pennylane import numpy as np
import pytest


def test_integration():
    """Test that the execution of lightning.qubit is possible and agrees with default.qubit"""
    wires = 2
    layers = 2
    dev_l = qml.device("lightning.qubit", wires=wires)
    dev_d = qml.device("default.qubit", wires=wires)

    def circuit(weights):
        qml.templates.StronglyEntanglingLayers(weights, wires=range(wires))
        return qml.expval(qml.PauliZ(0))

    np.random.seed(1967)
    weights = np.random.random(
        qml.templates.StronglyEntanglingLayers.shape(layers, wires), requires_grad=True
    )

    qn_l = qml.QNode(circuit, dev_l)
    qn_d = qml.QNode(circuit, dev_d)

    assert np.allclose(qn_l(weights), qn_d(weights))
    assert np.allclose(qml.grad(qn_l)(weights), qml.grad(qn_d)(weights))


class TestDtypePreserved:
    """Test that the user-defined dtype of the device is preserved for QNode
    evaluation"""

    @pytest.mark.parametrize("r_dtype", [np.float32, np.float64])
    @pytest.mark.parametrize(
        "measurement",
        [
            qml.expval(qml.PauliY(0)),
            qml.var(qml.PauliY(0)),
            qml.probs(wires=[1]),
            qml.probs(wires=[2, 0]),
        ],
    )
    def test_real_dtype(self, qubit_device_3_wires, r_dtype, measurement, tol):
        """Test that the default qubit plugin provides correct result for a simple circuit"""
        p = 0.543

        dev = qml.device("lightning.qubit", wires=3)
        dev.R_DTYPE = r_dtype

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.apply(measurement)

        res = circuit(p)
        assert res.dtype == r_dtype

    @pytest.mark.parametrize("c_dtype", [np.complex64, np.complex128])
    @pytest.mark.parametrize(
        "measurement",
        [qml.state(), qml.density_matrix(wires=[1]), qml.density_matrix(wires=[2, 0])],
    )
    def test_complex_dtype(self, qubit_device_3_wires, c_dtype, measurement, tol):
        """Test that the default qubit plugin provides correct result for a simple circuit"""
        p = 0.543

        dev = qml.device("lightning.qubit", wires=3)
        dev.C_DTYPE = c_dtype

        @qml.qnode(dev, diff_method="parameter-shift")
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.apply(measurement)

        res = circuit(p)
        assert res.dtype == c_dtype
