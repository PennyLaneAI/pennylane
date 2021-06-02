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
Tests for the Fourier spectrum transform.
"""
import pytest
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.fourier.spectrum import spectrum, _join_spectra, _get_spectrum


class TestHelpers:
    @pytest.mark.parametrize(
        "spectrum1, spectrum2, expected",
        [
            ([-1, 0, 1], [-1, 0, 1], [-2, -1, 0, 1, 2]),
            ([-3, 0, 3], [-5, 0, 5], [-8, -5, -3, -2, 0, 2, 3, 5, 8]),
            ([-2, -1, 0, 1, 2], [-1, 0, 1], [-3, -2, -1, 0, 1, 2, 3]),
            ([-0.5, 0, 0.5], [-1, 0, 1], [-1.5, -1, -0.5, 0, 0.5, 1.0, 1.5]),
        ],
    )
    def test_join_spectra(self, spectrum1, spectrum2, expected, tol):
        """Test that spectra are joined correctly."""
        joined = _join_spectra(spectrum1, spectrum2)
        assert np.allclose(joined, expected, atol=tol, rtol=0)

    @pytest.mark.parametrize(
        "op, expected",
        [
            (qml.RX(0.1, wires=0), [-1, 0, 1]),  # generator is a class
            (qml.RY(0.1, wires=0), [-1, 0, 1]),  # generator is a class
            (qml.RZ(0.1, wires=0), [-1, 0, 1]),  # generator is a class
            (qml.PhaseShift(0.5, wires=0), [-1, 0, 1]),  # generator is a array
            (qml.ControlledPhaseShift(0.5, wires=[0, 1]), [-1, 0, 1]),  # generator is array
        ],
    )
    def test_get_spectrum(self, op, expected, tol):
        """Test that the spectrum is correctly extracted from an operator."""
        spec = _get_spectrum(op)
        assert np.allclose(spec, expected, atol=tol, rtol=0)

    def test_get_spectrum_complains_no_generator(self):
        """Test that an error is raised if the operator has no generator defined."""

        # Observables have no generator attribute
        with pytest.raises(ValueError, match="generator of operation"):
            _get_spectrum(qml.P(wires=0))

        # CNOT is an operation where generator is an abstract property
        with pytest.raises(ValueError, match="generator of operation"):
            _get_spectrum(qml.CNOT(wires=[0,1]))


def circuit(x, w):
    """Test circuit"""
    for l in range(2):
        for i in range(3):
            qml.RX(x[i], wires=0, id="x" + str(i))
            qml.RY(w[l][i], wires=0)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
    qml.RZ(x[0], wires=0, id="x0")
    return qml.expval(qml.PauliZ(wires=0))


expected_result = {
    "x0": [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
    "x1": [-2.0, -1.0, 0.0, 1.0, 2.0],
    "x2": [-2.0, -1.0, 0.0, 1.0, 2.0],
}


class TestInterfaces:
    """Test that inputs are correctly identified and spectra computed in
    all interfaces."""

    def test_integration_autograd(self):
        """Test that the spectra of a circuit with lots of edge cases is calculated correctly
        in the autograd interface."""

        x = pnp.array([1, 2, 3], requires_grad=False)
        w = pnp.array([[-1, -2, -3], [-4, -5, -6]], requires_grad=True)

        dev = qml.device("default.qubit", wires=3)
        qnode = qml.QNode(circuit, dev, interface="autograd")

        res = spectrum(qnode)(x, w)
        for (k1, v1), (k2, v2) in zip(res.items(), expected_result.items()):
            assert k1 == k2
            assert v1 == v2

    def test_integration_torch(self):
        """Test that the spectra of a circuit with lots of edge cases is calculated correctly
        in the torch interface."""

        torch = pytest.importorskip("torch")
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
        w = torch.tensor([[-1, -2, -3], [-4, -5, -6]], requires_grad=False)

        dev = qml.device("default.qubit", wires=3)
        qnode = qml.QNode(circuit, dev, interface="torch")

        res = spectrum(qnode)(x, w)
        assert res
        for (k1, v1), (k2, v2) in zip(res.items(), expected_result.items()):
            assert k1 == k2
            assert v1 == v2

    def test_integration_tf(self):
        """Test that the spectra of a circuit with lots of edge cases is calculated correctly
        in the tf interface."""
        tf = pytest.importorskip("tensorflow")

        dev = qml.device("default.qubit", wires=3)
        qnode = qml.QNode(circuit, dev, interface="tf")

        with tf.GradientTape() as tape:
            x = tf.Variable([1.0, 2.0, 3.0, 4.0, 5.0])
            w = tf.constant([[-1, -2, -3], [-4, -5, -6]])
            # the spectrum function has to be called in a tape context
            res = spectrum(qnode)(x, w)

        assert res
        for (k1, v1), (k2, v2) in zip(res.items(), expected_result.items()):
            assert k1 == k2
            assert v1 == v2

    def test_integration_jax(self):
        """Test that the spectra of a circuit with lots of edge cases is calculated correctly
        in the jax interface."""

        jax = pytest.importorskip("jax")
        from jax import numpy as jnp

        x = jnp.array([1, 2, 3, 4, 5])
        w = [[-1, -2, -3], [-4, -5, -6]]

        dev = qml.device("default.qubit", wires=3)
        qnode = qml.QNode(circuit, dev, interface="jax")

        res = spectrum(qnode)(x, w)

        assert res
        for (k1, v1), (k2, v2) in zip(res.items(), expected_result.items()):
            assert k1 == k2
            assert v1 == v2
