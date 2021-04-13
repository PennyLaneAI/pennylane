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
from pennylane.interfaces.autograd import AutogradInterface


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


# Dummy operations

class SingleParamOp(qml.operation.Operation):
    """A dummy operation which defines a generator and takes a single parameter
    without processing it."""

    num_params = 1
    num_wires = 1
    par_domain = None
    generator = [np.array([[2, 0], [0, 4]]), 1]

    def __init__(self, a):
        super().__init__(a, wires=0, do_queue=True)


class OpProcessingFirstParam(qml.operation.Operation):
    """A dummy operation which processes the first of three inputs."""

    num_params = 3
    num_wires = 1
    par_domain = None

    def __init__(self, a, b, c):
        # process a
        a = a / 2
        super().__init__(a, b, c, wires=0, do_queue=True)

    def expand(self):
        a = self.parameters[0]
        b = self.parameters[1]
        c = self.parameters[2]

        with qml.tape.QuantumTape() as tape:
            SingleParamOp(a)
            SingleParamOp(b)
            SingleParamOp(c)
        return tape


class TestSimplify:
    """Tests for the _simplify_tape function."""

    def test_exception_if_classical_processing_of_input(self):
        """Test that classical processing of an input leads to an error."""

        x = pnp.array([0.1, 0.2, 0.3], requires_grad=True)

        with qml.tape.JacobianTape() as tape_x:
            OpProcessingFirstParam(x[0], x[1], x[2])

        # turn into tape with interface,
        # to simulate what a qnode would do
        tape_x = AutogradInterface.apply(tape_x)

        # processing on first input -> abort
        with pytest.raises(ValueError, match="Aborting the expansion."):
            _simplify_tape(tape_x, original_inputs=[x[0], x[1], x[2]])

    def test_exception_classical_processing_if_other_params_noninputs(self):
        """Test that classical processing of an input leads to an error
        even if the other gate parameters are non-inputs."""

        x = pnp.array([0.1, 0.2, 0.3], requires_grad=True)
        z = pnp.array([-0.1, -0.2, -0.3], requires_grad=False)

        with qml.tape.JacobianTape() as tape_xz:
            OpProcessingFirstParam(x[0], z[1], z[2])

        # turn into tape with interface,
        # to simulate what a qnode would do
        tape_xz = AutogradInterface.apply(tape_xz)

        # processing on first input -> abort, even if other parameters are non-inputs
        with pytest.raises(ValueError, match="Aborting the expansion."):
            _simplify_tape(tape_xz, original_inputs=[x[0], x[1], x[2]])

    def test_no_expansion_if_no_input(self):
        """Test that if all parameters are non-inputs, no expansion happens."""

        x = pnp.array([0.1, 0.2, 0.3], requires_grad=True)
        z = pnp.array([-0.1, -0.2, -0.3], requires_grad=False)

        with qml.tape.JacobianTape() as tape_z:
            OpProcessingFirstParam(z[0], z[1], z[2])

        # turn into tape with interface,
        # to simulate what a qnode would do
        tape_z = AutogradInterface.apply(tape_z)

        # no inputs enter gate -> no simplification necessary
        new_tape = _simplify_tape(tape_z, original_inputs=[x[0], x[1], x[2]])
        names = [gate.name for gate in new_tape.operations]
        assert names == ["OpProcessingFirstParam"]

    def test_no_expansion_if_classical_processing_on_noninput(self):
        """Test that classical processing of a non-input, while no processing is performed on inputs,
         leads to a valid expansion."""

        x = pnp.array([0.1, 0.2, 0.3], requires_grad=True)
        z = pnp.array([-0.1, -0.2, -0.3], requires_grad=False)

        with qml.tape.JacobianTape() as tape_zx:
            OpProcessingFirstParam(z[0], x[1], x[2])

        # turn into tape with interface,
        # to simulate what a qnode would do
        tape_zx = AutogradInterface.apply(tape_zx)

        # the processed parameter is no input, can just expand
        new_tape = _simplify_tape(tape_zx, original_inputs=[x[0], x[1], x[2]])
        names = [gate.name for gate in new_tape.operations]
        assert names == ["SingleParamOp", "SingleParamOp", "SingleParamOp"]

    def test_no_changes_real_gate(self, tol):
        """Test simplification of circuit with real gates."""

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

    def test_exception_processing_on_input_real_gate(self):
        """Test that simplification throws an error when the expansion of an operation
        would change a parameter, using real gates."""

        x = [0.1, 0.2, 0.3]

        with qml.tape.QuantumTape() as tape_not_simplifiable:
            qml.RX(x[0], wires=0)
            qml.CRot(x[0], x[1], x[2], wires=[0, 1])
            qml.PhaseShift(x[2], wires=0)

        with pytest.raises(ValueError, match="transforms the inputs"):
            _simplify_tape(tape_not_simplifiable, original_inputs=x)

    def test_track_inputs_real_gate(self):
        """Test that simplification can distinguish whether a map is performed on an
        input (which is not allowed) or other parameters (which is fine)."""

        no_input = pnp.array([0.1, 0.2, 0.3], requires_grad=False)
        inpt = pnp.array([-0.1, -0.2, -0.3], requires_grad=True)

        with qml.tape.QuantumTape() as tape_exception:
            qml.Rot(no_input[0], no_input[1], no_input[2], wires=[0])
            qml.CRot(inpt[0], inpt[1], inpt[2], wires=[0, 1])

        tape_exception = AutogradInterface.apply(tape_exception)

        # cannot simplify tape due to CRot
        with pytest.raises(ValueError, match="transforms the inputs"):
            _simplify_tape(tape_exception, original_inputs=inpt)

        # here non-inputs enter the CRot, so all is good
        with qml.tape.QuantumTape() as tape_ok:
            qml.Rot(inpt[0], inpt[1], inpt[2], wires=[0])
            qml.CRot(no_input[0], no_input[1], no_input[2], wires=[0, 1])

        tape_ok = AutogradInterface.apply(tape_ok)

        expanded = _simplify_tape(tape_ok, original_inputs=inpt)
        queue = [op.name for op in expanded.operations]
        assert queue == ["RZ", "RY", "RZ", "CRot"]


class TestIntegration:
    """Test that inputs are correctly identified and spectra computed in
    all interfaces."""

    def circuit(self, x, z):
        # use an embedding
        qml.templates.AngleEmbedding(x[0:3], wires=[0, 1, 2])
        qml.RX(x[0], wires=1)
        # mixed entries
        qml.Rot(z[1], x[0], x[1], wires=1)
        qml.CNOT(wires=[1, 2])
        qml.RX(x[4], wires=2)
        qml.RX(z[0], wires=2)
        # feed non-inputs into an operation that cannot be simplified by expansion
        qml.CRot(z[0], z[0], z[1], wires=[0, 1])
        return qml.expval(qml.PauliZ(wires=2))

    expected_result = {
        1: [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
        2: [-2.0, -1.0, 0.0, 1.0, 2.0],
        3: [-1.0, 0.0, 1.0],
        5: [-1.0, 0.0, 1.0],
    }

    def test_integration_autograd(self):
        """Test that the spectra of a circuit with lots of edge cases is calculated correctly
        in the autograd interface."""

        x = pnp.array([1, 2, 3, 4, 5], requires_grad=True)
        z = pnp.array([-1, -2], requires_grad=False)

        dev = qml.device("default.qubit", wires=3)
        circuit = qml.QNode(self.circuit, dev, interface='autograd')

        res = spectrum(circuit)(x, z)
        for (k1, v1), (k2, v2) in zip(res.items(), self.expected_result.items()):
            assert k1 == k2
            assert v1 == v2

    def test_integration_torch(self):
        """Test that the spectra of a circuit with lots of edge cases is calculated correctly
        in the torch interface."""

        torch = pytest.importorskip("torch")
        x = torch.tensor([1., 2., 3., 4., 5.], requires_grad=True)
        z = torch.tensor([-1., -2.], requires_grad=False)

        dev = qml.device("default.qubit", wires=3)
        circuit = qml.QNode(self.circuit, dev, interface='torch')

        res = spectrum(circuit)(x, z)
        for (k1, v1), (k2, v2) in zip(res.items(), self.expected_result.items()):
            assert k1 == k2
            assert v1 == v2

    def test_integration_tf(self):
        """Test that the spectra of a circuit with lots of edge cases is calculated correctly
        in the tf interface."""
        tf = pytest.importorskip("tensorflow")

        dev = qml.device("default.qubit", wires=3)
        circuit = qml.QNode(self.circuit, dev, interface='tf')

        with tf.GradientTape() as tape:
            x = tf.Variable([1., 2., 3., 4., 5.])
            z = tf.constant([-1., -2.])
            # the spectrum function has to be called in a tape context
            res = spectrum(circuit)(x, z)

        for (k1, v1), (k2, v2) in zip(res.items(), self.expected_result.items()):
            assert k1 == k2
            assert v1 == v2
