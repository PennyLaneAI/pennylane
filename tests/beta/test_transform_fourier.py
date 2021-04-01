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
Tests for the fourier qnode transforms.
"""
import pytest
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.fourier.spectrum import spectrum, _join_spectra, _get_spectrum, _simplify_tape


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
            (qml.RX(0.1, wires=0), [-1, 0, 1]),
            (qml.RY(0.1, wires=0), [-1, 0, 1]),
            (qml.RZ(0.1, wires=0), [-1, 0, 1]),
            (qml.PhaseShift(0.5, wires=0), [-1, 0, 1]),
            (qml.ControlledPhaseShift(0.5, wires=[0, 1]), [-1, 0, 1]),
        ],
    )
    def test_get_spectrum(self, op, expected, tol):
        """Test that the spectrum is correctly extracted from an operator."""
        spec = _get_spectrum(op)
        assert np.allclose(spec, expected, atol=tol, rtol=0)

    def test_get_spectrum_complains_wrong_op(self):
        """Test that an error is raised if the operator has no generator defined."""
        op = qml.Rot(0.1, 0.1, 0.1, wires=0)
        with pytest.raises(ValueError, match="no generator defined"):
            _get_spectrum(op)


class TestSimplify:
    """Tests for the _simplify_tape function."""

    def test_no_changes(self, tol):
        """Test the _simplify_tape function for a tape that does not need to be changed"""

        x = [0.1, 0.2, 0.3]

        with qml.tape.QuantumTape() as tape_already_simple:
            qml.RX(x[0], wires=0)
            qml.RY(x[1], wires=2)
            qml.PhaseShift(x[2], wires=0)

        new_tape = _simplify_tape(tape_already_simple, original_inputs=x)
        for op1, op2 in zip(new_tape.operations, tape_already_simple.operations):
            assert op1.name == op2.name
            assert np.allclose(op1.parameters, op2.parameters, atol=tol, rtol=0)
            assert op1.wires == op2.wires

    def test_new_trainable_parameter_created(self, tol):
        """Test that the _simplify_tape function throws an error when the expansion of an operation
        would change a parameter."""

        x = [0.1, 0.2, 0.3]

        with qml.tape.QuantumTape() as tape_not_simplifiable:
            qml.RX(x[0], wires=0)
            qml.CRot(x[0], x[1], x[2], wires=[0, 1])
            qml.PhaseShift(x[2], wires=0)

        with pytest.raises(ValueError, match="transforms the inputs"):
            _simplify_tape(tape_not_simplifiable, original_inputs=x)

    def test_track_inputs(self):
        """Test that _simplify_tape can distinguish whether a map is performed on an
        input (which is not allowed) or other parameters (which is fine)."""

        no_input = pnp.array([0.1, 0.2, 0.3], requires_grad=False)
        inpt = pnp.array([-0.1, -0.2, -0.3], requires_grad=True)

        with qml.tape.QuantumTape() as tape_exception:
            qml.Rot(no_input[0], no_input[1], no_input[2], wires=[0])
            qml.CRot(inpt[0], inpt[1], inpt[2], wires=[0, 1])

        with pytest.raises(ValueError, match="transforms the inputs"):
            _simplify_tape(tape_exception, original_inputs=inpt)

        with qml.tape.QuantumTape() as tape_ok:
            qml.Rot(inpt[0], inpt[1], inpt[2], wires=[0])
            qml.CRot(no_input[0], no_input[1], no_input[2], wires=[0, 1])

        expanded = _simplify_tape(tape_ok, original_inputs=inpt)
        queue = [op.name for op in expanded.operations]
        assert queue == ["RZ", "RY", "RZ", "CRot"]


class TestIntegration:
    """Integration tests."""

    # Todo: add tests in all interfaces
    # Todo: make test pass for one op taking inputs and other parameters

    def crazy_logic(self):
        """Test that the spectra of a circuit with lots of edge cases is calculated correctly."""

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def circuit(x, z):
            # use an embedding
            qml.templates.AngleEmbedding(x[0:3], wires=[0, 1, 2])
            qml.RX(x[0], wires=1)
            qml.Rot(x[1], x[0], x[1], wires=1)
            qml.CNOT(wires=[1, 2])
            qml.RX(x[4], wires=2)
            qml.RX(z[0], wires=2)
            # feed non-inputs into an operation that cannot be simplified by expansion
            qml.CRot(z[0], z[0], z[1], wires=[0, 1])
            return qml.expval(qml.PauliZ(wires=2))

        x = pnp.array([0.1, 0.2, 0.3, 0.4, 0.5], requires_grad=True)
        z = pnp.array([-0.1, -0.2], requires_grad=False)

        expected = {
            x[0]: [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            x[1]: [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
            x[2]: [-1.0, 0.0, 1.0],
            x[4]: [-1.0, 0.0, 1.0],
        }

        res = spectrum(circuit)(x, z)

        for (k1, v1), (k2, v2) in zip(res.items(), expected.items()):
            assert k1 == k2
            assert v1 == v2
