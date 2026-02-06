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
import pytest
from flaky import flaky

import pennylane as qp
from pennylane import numpy as np
from pennylane.exceptions import QuantumFunctionError


def test_integration():
    """Test that the execution of lightning.qubit is possible and agrees with default.qubit"""
    wires = 2
    layers = 2
    dev_l = qp.device("lightning.qubit", wires=wires)
    dev_d = qp.device("default.qubit", wires=wires)

    def circuit(weights):
        qp.templates.StronglyEntanglingLayers(weights, wires=range(wires))
        return qp.expval(qp.PauliZ(0))

    weights = np.random.random(
        qp.templates.StronglyEntanglingLayers.shape(layers, wires), requires_grad=True
    )

    qn_l = qp.QNode(circuit, dev_l)
    qn_d = qp.QNode(circuit, dev_d)

    assert np.allclose(qn_l(weights), qn_d(weights))
    assert np.allclose(qp.grad(qn_l)(weights), qp.grad(qn_d)(weights))


def test_no_backprop_auto_interface():
    """Test that lightning.qubit does not support the backprop
    differentiation method."""

    dev = qp.device("lightning.qubit", wires=2)

    def circuit():
        """Simple quantum function."""
        return qp.expval(qp.PauliZ(0))

    with pytest.raises(QuantumFunctionError, match="does not support backprop"):
        qp.QNode(circuit, dev, diff_method="backprop")


def test_finite_shots_adjoint():
    """Test that shots and adjoint diff raises an error."""

    dev = qp.device("lightning.qubit", wires=2)

    def circuit():
        """Simple quantum function."""
        return qp.expval(qp.PauliZ(0))

    with pytest.raises(QuantumFunctionError, match="does not support adjoint"):
        qp.set_shots(qp.QNode(circuit, dev, diff_method="adjoint"), shots=2)()


@flaky(max_runs=5)
def test_finite_shots(seed):
    """Test that shots in LQ and DQ give the same results."""

    dev = qp.device("lightning.qubit", wires=2, seed=seed)
    dq = qp.device("default.qubit", seed=seed)

    def circuit():
        qp.RX(np.pi / 4, 0)
        qp.RY(-np.pi / 4, 1)
        return qp.expval(qp.PauliY(0))

    circ0 = qp.set_shots(qp.QNode(circuit, dev, diff_method=None), shots=50000)
    circ1 = qp.set_shots(qp.QNode(circuit, dq, diff_method=None), shots=50000)

    assert np.allclose(circ0(), circ1(), rtol=0.01)


class TestDtypePreserved:
    """Test that the user-defined dtype of the device is preserved for QNode
    evaluation"""

    # pylint: disable=too-few-public-methods

    @pytest.mark.parametrize(
        "c_dtype",
        [
            np.complex64,
            np.complex128,
        ],
    )
    @pytest.mark.parametrize(
        "measurement",
        [
            qp.state(),
            qp.density_matrix(wires=[1]),
            qp.density_matrix(wires=[2, 0]),
            qp.expval(qp.PauliY(0)),
            qp.var(qp.PauliY(0)),
            qp.probs(wires=[1]),
            qp.probs(wires=[0, 2]),
        ],
    )
    def test_dtype(self, c_dtype, measurement):
        """Test that the default qubit plugin provides correct result for a simple circuit"""
        p = 0.543

        dev = qp.device("lightning.qubit", wires=3, c_dtype=c_dtype)

        @qp.qnode(dev)
        def circuit(x):
            qp.RX(x, wires=0)
            return qp.apply(measurement)

        res = circuit(p)

        if isinstance(measurement, (qp.measurements.StateMP, qp.measurements.DensityMatrixMP)):
            expected_dtype = c_dtype
        else:
            expected_dtype = np.float64 if c_dtype == np.complex128 else np.float32
        if isinstance(res, np.ndarray):
            assert res.dtype == expected_dtype
        else:
            assert isinstance(res, float)
