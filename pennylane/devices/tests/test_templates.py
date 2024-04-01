# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests that various templates work correctly on a device."""
# pylint: disable=no-self-use

# Can generate a list of all templates using the following code:
#
# from inspect import getmembers, isclass
# all_templates = [i for (_, i) in getmembers(qml.templates) if isclass(i) and issubclass(i, qml.operation.Operator)]

from functools import partial
import pytest
import numpy as np
from scipy.stats import norm

import pennylane as qml
from pennylane import math


pytestmark = pytest.mark.skip_unsupported


def check_op_supported(op, dev):
    """Skip test if device does not support an operation. Works with both device APIs"""
    if isinstance(dev, qml.Device):
        if op.name not in dev.operations:
            pytest.skip("operation not supported.")
    else:
        prog, _ = dev.preprocess()
        tape = qml.tape.QuantumScript([op])
        try:
            prog((tape,))
        except qml.DeviceError:
            pytest.skip("operation not supported on the device")


class TestTemplates:  # pylint:disable=too-many-public-methods
    """Test various templates."""

    def test_AQFT(self, device, tol):
        """Test the AQFT template."""
        wires = 3
        dev = device(wires=wires)

        @qml.qnode(dev)
        def circuit_aqft():
            qml.X(0)
            qml.Hadamard(1)
            qml.AQFT(order=1, wires=range(wires))
            return qml.probs()

        expected = [0.25, 0.125, 0.0, 0.125, 0.25, 0.125, 0.0, 0.125]
        assert np.allclose(circuit_aqft(), expected, atol=tol(dev.shots))

    def test_AllSinglesDoubles(self, device, tol):
        """Test the AllSinglesDoubles template."""
        qubits = 4
        dev = device(qubits)

        electrons = 2

        # Define the HF state
        hf_state = qml.qchem.hf_state(electrons, qubits)

        # Generate all single and double excitations
        singles, doubles = qml.qchem.excitations(electrons, qubits)

        wires = range(qubits)

        @qml.qnode(dev)
        def circuit(weights, hf_state, singles, doubles):
            qml.AllSinglesDoubles(weights, wires, hf_state, singles, doubles)
            return qml.expval(qml.Z(0))

        # Evaluate the QNode for a given set of parameters
        params = np.array([0.12, 1.23, 2.34])
        res = circuit(params, hf_state, singles=singles, doubles=doubles)
        assert np.isclose(res, 0.6905612772956113, atol=tol(dev.shots))

    def test_AmplitudeEmbedding(self, device, tol):
        """Test the AmplitudeEmbedding template."""
        dev = device(2)

        @qml.qnode(dev)
        def circuit(f):
            qml.AmplitudeEmbedding(features=f, wires=range(2))
            return qml.probs()

        res = circuit([1 / 2] * 4)
        expected = [1 / 4] * 4
        assert np.allclose(res, expected, atol=tol(dev.shots))

    def test_AngleEmbedding(self, device, tol):
        """Test the AngleEmbedding template."""
        n_wires = 3
        dev = device(n_wires)

        @qml.qnode(dev)
        def circuit(feature_vector):
            qml.AngleEmbedding(features=feature_vector, wires=range(n_wires), rotation="X")
            qml.Hadamard(0)
            return qml.probs(wires=range(3))

        x = [np.pi / 2] * 3
        res = circuit(x)
        expected = [0.125] * 8
        assert np.allclose(res, expected, atol=tol(dev.shots))

    def test_ApproxTimeEvolution(self, device, tol):
        """Test the ApproxTimeEvolution template."""
        n_wires = 2
        dev = device(n_wires)
        wires = range(n_wires)

        coeffs = [1, 1]
        obs = [qml.X(0), qml.X(1)]
        hamiltonian = qml.Hamiltonian(coeffs, obs)

        @qml.qnode(dev)
        def circuit(time):
            qml.ApproxTimeEvolution(hamiltonian, time, 1)
            return [qml.expval(qml.Z(i)) for i in wires]

        res = circuit(1)
        expected = [-0.41614684, -0.41614684]
        assert np.allclose(res, expected, atol=tol(dev.shots))

    def test_ArbitraryStatePreparation(self, device, tol):
        """Test the ArbitraryStatePreparation template."""
        dev = device(2)

        @qml.qnode(dev)
        def circuit(weights):
            qml.ArbitraryStatePreparation(weights, wires=[0, 1])
            return qml.probs()

        weights = np.arange(1, 7) / 10
        res = circuit(weights)
        expected = [0.784760658335564, 0.0693785880617069, 0.00158392607496555, 0.1442768275277600]
        assert np.allclose(res, expected, atol=tol(dev.shots))

    def test_ArbitraryUnitary(self, device, tol):
        """Test the ArbitraryUnitary template."""
        dev = device(1)

        @qml.qnode(dev)
        def circuit(weights):
            qml.ArbitraryUnitary(weights, wires=[0])
            return qml.probs()

        weights = np.arange(3)
        res = circuit(weights)
        expected = [0.77015115293406, 0.22984884706593]
        assert np.allclose(res, expected, atol=tol(dev.shots))

    def test_BasicEntanglerLayers(self, device, tol):
        """Test the BasicEntanglerLayers template."""
        n_wires = 3
        dev = device(n_wires)

        @qml.qnode(dev)
        def circuit(weights):
            qml.BasicEntanglerLayers(weights=weights, wires=range(n_wires))
            return [qml.expval(qml.Z(i)) for i in range(n_wires)]

        params = [[np.pi, np.pi, np.pi]]
        res = circuit(params)
        expected = [1.0, 1.0, -1.0]
        assert np.allclose(res, expected, atol=tol(dev.shots))

    def test_BasisEmbedding(self, device, tol):
        """Test the BasisEmbedding template."""
        dev = device(3)

        @qml.qnode(dev)
        def circuit(basis):
            qml.BasisEmbedding(basis, wires=range(3))
            return qml.probs()

        basis = (1, 0, 1)
        res = circuit(basis)

        basis_idx = np.dot(basis, 2 ** np.arange(3))
        expected = np.zeros(8)
        expected[basis_idx] = 1.0
        assert np.allclose(res, expected, atol=tol(dev.shots))

    def test_BasisRotation(self, device, tol):
        """Test the BasisRotation template."""
        dev = device(2)
        if dev.shots or "mixed" in dev.name or "Mixed" in dev.name:
            pytest.skip("test only works with analytic-mode pure statevector simulators")

        unitary_matrix = np.array(
            [
                [-0.77228482 + 0.0j, -0.02959195 + 0.63458685j],
                [0.63527644 + 0.0j, -0.03597397 + 0.77144651j],
            ]
        )
        eigen_values = np.array([-1.45183325, 3.47550075])
        exp_state = np.array([0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, -0.43754907 - 0.89919453j])

        @qml.qnode(dev)
        def circuit():
            qml.PauliX(0)
            qml.PauliX(1)
            qml.adjoint(qml.BasisRotation(wires=[0, 1], unitary_matrix=unitary_matrix))
            for idx, eigenval in enumerate(eigen_values):
                qml.RZ(-eigenval, wires=[idx])
            qml.BasisRotation(wires=[0, 1], unitary_matrix=unitary_matrix)
            return qml.state()

        assert np.allclose(
            [math.fidelity_statevector(circuit(), exp_state)], [1.0], atol=tol(dev.shots)
        )

    def test_BasisStatePreparation(self, device, tol):
        """Test the BasisStatePreparation template."""
        dev = device(4)

        @qml.qnode(dev)
        def circuit(basis_state):
            qml.BasisStatePreparation(basis_state, wires=range(4))
            return [qml.expval(qml.Z(i)) for i in range(4)]

        basis_state = [0, 1, 1, 0]
        res = circuit(basis_state)
        expected = [1.0, -1.0, -1.0, 1.0]
        assert np.allclose(res, expected, atol=tol(dev.shots))

    @pytest.mark.xfail(reason="most devices do not support CV")
    def test_CVNeuralNetLayers(self, device):
        """Test the CVNeuralNetLayers template."""
        dev = device(2)

        @qml.qnode(dev)
        def circuit(weights):
            qml.CVNeuralNetLayers(*weights, wires=[0, 1])
            return qml.expval(qml.QuadX(0))

        shapes = qml.CVNeuralNetLayers.shape(n_layers=2, n_wires=2)
        weights = [np.random.random(shape) for shape in shapes]

        circuit(weights)

    def test_CommutingEvolution(self, device, tol):
        """Test the CommutingEvolution template."""
        n_wires = 2
        dev = device(n_wires)
        coeffs = [1, -1]
        obs = [qml.X(0) @ qml.Y(1), qml.Y(0) @ qml.X(1)]
        hamiltonian = qml.Hamiltonian(coeffs, obs)
        frequencies = (2, 4)

        @qml.qnode(dev)
        def circuit(time):
            qml.X(0)
            qml.CommutingEvolution(hamiltonian, time, frequencies)
            return qml.expval(qml.Z(0))

        res = circuit(1)
        expected = 0.6536436208636115
        assert np.isclose(res, expected, atol=tol(dev.shots))

    def test_ControlledSequence(self, device, tol):
        """Test the ControlledSequence template."""
        dev = device(4)

        @qml.qnode(dev)
        def circuit():
            for i in range(3):
                qml.Hadamard(wires=i)
            qml.ControlledSequence(qml.RX(0.25, wires=3), control=[0, 1, 2])
            qml.adjoint(qml.QFT)(wires=range(3))
            return qml.probs(wires=range(3))

        res = circuit()
        expected = [
            0.92059345,
            0.02637178,
            0.00729619,
            0.00423258,
            0.00360545,
            0.00423258,
            0.00729619,
            0.02637178,
        ]
        assert np.allclose(res, expected, atol=tol(dev.shots))

    def test_CosineWindow(self, device, tol):
        """Test the CosineWindow template."""
        dev = device(2)

        @qml.qnode(dev)
        def circuit():
            qml.CosineWindow(wires=range(2))
            return qml.probs()

        res = circuit()
        expected = [0.0, 0.25, 0.5, 0.25]
        assert np.allclose(res, expected, atol=tol(dev.shots))

    @pytest.mark.xfail(reason="most devices do not support CV")
    def test_DisplacementEmbedding(self, device, tol):
        """Test the DisplacementEmbedding template."""
        dev = device(3)

        @qml.qnode(dev)
        def circuit(feature_vector):
            qml.DisplacementEmbedding(features=feature_vector, wires=range(3))
            qml.QuadraticPhase(0.1, wires=1)
            return qml.expval(qml.NumberOperator(wires=1))

        X = [1, 2, 3]

        res = circuit(X)
        expected = 4.1215690638748494
        assert np.isclose(res, expected, atol=tol(dev.shots))

    def test_FermionicDoubleExcitation(self, device, tol):
        """Test the FermionicDoubleExcitation template."""
        dev = device(5)
        if getattr(dev, "short_name", None) == "cirq.mixedsimulator" and dev.shots:
            pytest.xfail(reason="device is generating negative probabilities")

        @qml.qnode(dev)
        def circuit(weight, wires1=None, wires2=None):
            qml.FermionicDoubleExcitation(weight, wires1=wires1, wires2=wires2)
            return qml.expval(qml.Z(0))

        res = circuit(1.34817, wires1=[0, 1], wires2=[2, 3, 4])
        expected = 1.0
        assert np.isclose(res, expected, atol=tol(dev.shots))

    def test_FermionicSingleExcitation(self, device, tol):
        """Test the FermionicSingleExcitation template."""
        dev = device(3)
        if getattr(dev, "short_name", None) == "cirq.mixedsimulator" and dev.shots:
            pytest.xfail(reason="device is generating negative probabilities")

        @qml.qnode(dev)
        def circuit(weight, wires=None):
            qml.FermionicSingleExcitation(weight, wires=wires)
            return qml.expval(qml.Z(0))

        res = circuit(0.56, wires=[0, 1, 2])
        expected = 1.0
        assert np.isclose(res, expected, atol=tol(dev.shots))

    def test_FlipSign(self, device, tol):
        """Test the FlipSign template."""
        dev = device(2)
        if dev.shots:
            pytest.skip("test only works with analytic-mode simulations")
        basis_state = [1, 0]

        @qml.qnode(dev)
        def circuit():
            for wire in list(range(2)):
                qml.Hadamard(wires=wire)
            qml.FlipSign(basis_state, wires=list(range(2)))
            return qml.state()

        res = circuit()
        expected = [0.5, 0.5, -0.5, 0.5]
        if "mixed" in dev.name or "Mixed" in dev.name:
            expected = math.dm_from_state_vector(expected)
        assert np.allclose(res, expected, atol=tol(dev.shots))

    def test_GroverOperator(self, device, tol):
        """Test the GroverOperator template."""
        n_wires = 3
        dev = device(n_wires)
        wires = list(range(n_wires))

        def oracle():
            qml.Hadamard(wires[-1])
            qml.Toffoli(wires=wires)
            qml.Hadamard(wires[-1])

        @qml.qnode(dev)
        def circuit(num_iterations=1):
            for wire in wires:
                qml.Hadamard(wire)

            for _ in range(num_iterations):
                oracle()
                qml.GroverOperator(wires=wires)
            return qml.probs(wires)

        res = circuit(num_iterations=2)
        expected = [
            0.0078125,
            0.0078125,
            0.0078125,
            0.0078125,
            0.0078125,
            0.0078125,
            0.0078125,
            0.9453125,
        ]
        assert np.allclose(res, expected, atol=tol(dev.shots))

    def test_HilbertSchmidt(self, device, tol):
        """Test the HilbertSchmidt template."""
        dev = device(2)
        u_tape = qml.tape.QuantumScript([qml.Hadamard(0)])

        def v_function(params):
            qml.RZ(params[0], wires=1)

        @qml.qnode(dev)
        def hilbert_test(v_params, v_function, v_wires, u_tape):
            qml.HilbertSchmidt(v_params, v_function=v_function, v_wires=v_wires, u_tape=u_tape)
            return qml.probs(u_tape.wires + v_wires)

        def cost_hst(parameters, v_function, v_wires, u_tape):
            # pylint:disable=unsubscriptable-object
            return (
                1
                - hilbert_test(
                    v_params=parameters, v_function=v_function, v_wires=v_wires, u_tape=u_tape
                )[0]
            )

        res = cost_hst([0], v_function=v_function, v_wires=[1], u_tape=u_tape)
        expected = 1.0
        assert np.isclose(res, expected, atol=tol(dev.shots))

    def test_IQPEmbedding(self, device, tol):
        """Test the IQPEmbedding template."""
        dev = device(3)

        @qml.qnode(dev)
        def circuit(features):
            qml.IQPEmbedding(features, wires=range(3), n_repeats=4)
            return [qml.expval(qml.Z(w)) for w in range(3)]

        res = circuit([1.0, 2.0, 3.0])
        expected = [0.40712208, 0.32709118, 0.89125407]
        assert np.allclose(res, expected, atol=tol(dev.shots))

    @pytest.mark.xfail(reason="most devices do not support CV")
    def test_Interferometer(self, device):
        """Test the Interferometer template."""
        dev = device(4)

        @qml.qnode(dev)
        def circuit(params):
            qml.Interferometer(*params, wires=range(4))
            return qml.expval(qml.Identity(0))

        shapes = [[6], [6], [4]]
        params = []
        for shape in shapes:
            params.append(np.random.random(shape))

        _ = circuit(params)

    def test_LocalHilbertSchmidt(self, device, tol):
        """Test the LocalHilbertSchmidt template."""
        dev = device(4)
        u_tape = qml.tape.QuantumScript([qml.CZ(wires=(0, 1))])

        def v_function(params):
            qml.RZ(params[0], wires=2)
            qml.RZ(params[1], wires=3)
            qml.CNOT(wires=[2, 3])
            qml.RZ(params[2], wires=3)
            qml.CNOT(wires=[2, 3])

        @qml.qnode(dev)
        def local_hilbert_test(v_params, v_function, v_wires, u_tape):
            qml.LocalHilbertSchmidt(v_params, v_function=v_function, v_wires=v_wires, u_tape=u_tape)
            return qml.probs(u_tape.wires + v_wires)

        def cost_lhst(parameters, v_function, v_wires, u_tape):
            # pylint:disable=unsubscriptable-object
            return (
                1
                - local_hilbert_test(
                    v_params=parameters, v_function=v_function, v_wires=v_wires, u_tape=u_tape
                )[0]
            )

        res = cost_lhst(
            [3 * np.pi / 2, 3 * np.pi / 2, np.pi / 2],
            v_function=v_function,
            v_wires=[2, 3],
            u_tape=u_tape,
        )
        expected = 0.5
        assert np.isclose(res, expected, atol=tol(dev.shots))

    def test_MERA(self, device, tol):
        """Test the MERA template."""

        def block(weights, wires):
            qml.CNOT(wires=[wires[0], wires[1]])
            qml.RY(weights[0], wires=wires[0])
            qml.RY(weights[1], wires=wires[1])

        n_wires = 4
        n_block_wires = 2
        n_params_block = 2
        n_blocks = qml.MERA.get_n_blocks(range(n_wires), n_block_wires)
        template_weights = [[0.1, -0.3]] * n_blocks
        dev = device(n_wires)

        @qml.qnode(dev)
        def circuit(template_weights):
            qml.MERA(range(n_wires), n_block_wires, block, n_params_block, template_weights)
            return qml.expval(qml.Z(1))

        res = circuit(template_weights)
        expected = 0.799260896638786
        assert np.isclose(res, expected, atol=tol(dev.shots))

    def test_MPS(self, device, tol):
        """Test the MPS template."""

        def block(weights, wires):
            qml.CNOT(wires=[wires[0], wires[1]])
            qml.RY(weights[0], wires=wires[0])
            qml.RY(weights[1], wires=wires[1])

        n_wires = 4
        n_block_wires = 2
        n_params_block = 2
        n_blocks = qml.MPS.get_n_blocks(range(n_wires), n_block_wires)
        template_weights = [[0.1, -0.3]] * n_blocks
        dev = device(n_wires)

        @qml.qnode(dev)
        def circuit(template_weights):
            qml.MPS(range(n_wires), n_block_wires, block, n_params_block, template_weights)
            return qml.expval(qml.Z(n_wires - 1))

        res = circuit(template_weights)
        expected = 0.8719048589118708
        assert np.isclose(res, expected, atol=tol(dev.shots))

    def test_MottonenStatePreparation_probs(self, device, tol):
        """Test the MottonenStatePreparation template (up to a phase)."""
        dev = device(3)

        @qml.qnode(dev)
        def circuit(state):
            qml.MottonenStatePreparation(state_vector=state, wires=range(3))
            return qml.probs()

        state = np.array([1, 2j, 3, 4j, 5, 6j, 7, 8j])
        state = state / np.linalg.norm(state)
        res = circuit(state)
        expected = np.abs(state**2)
        assert np.allclose(res, expected, atol=tol(dev.shots))

    def test_MottonenStatePreparation_state(self, device, tol):
        """Test the MottonenStatePreparation template on analytic-mode devices."""
        dev = device(3)
        if dev.shots:
            pytest.skip("test only works with analytic-mode simulations")

        @qml.qnode(dev)
        def circuit(state):
            qml.MottonenStatePreparation(state_vector=state, wires=range(3))
            return qml.state()

        state = np.array([1, 2j, 3, 4j, 5, 6j, 7, 8j])
        state = state / np.linalg.norm(state)
        res = circuit(state)
        expected = state
        if "mixed" in dev.name or "Mixed" in dev.name:
            expected = math.dm_from_state_vector(expected)
        if np.allclose(res, expected, atol=tol(dev.shots)):
            # GlobalPhase supported
            return
        # GlobalPhase not supported
        global_phase = qml.math.sum(-1 * qml.math.angle(expected) / len(expected))
        global_phase = np.exp(-1j * global_phase)
        assert np.allclose(expected / res, global_phase)

    def test_Permute(self, device, tol):
        """Test the Permute template."""
        dev = device(2)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep([1 / np.sqrt(2), 0.4, 0.5, 0.3], wires=[0, 1])
            qml.Permute([1, 0], [0, 1])
            return qml.probs()

        res = circuit()
        expected = [0.5, 0.25, 0.16, 0.09]
        assert np.allclose(res, expected, atol=tol(dev.shots))

    def test_QAOAEmbedding(self, device, tol):
        """Test the QAOAEmbedding template."""
        dev = device(2)

        @qml.qnode(dev)
        def circuit(weights, f=None):
            qml.QAOAEmbedding(features=f, weights=weights, wires=range(2))
            return qml.expval(qml.Z(0))

        features = [1.0, 2.0]
        layer1 = [0.1, -0.3, 1.5]
        layer2 = [3.1, 0.2, -2.8]
        weights = [layer1, layer2]

        res = circuit(weights, f=features)
        expected = 0.49628561029741747
        assert np.isclose(res, expected, atol=tol(dev.shots))

    def test_QDrift(self, device, tol):
        """Test the QDrift template."""
        dev = device(2)
        coeffs = [0.25, 0.75]
        ops = [qml.X(0), qml.Z(0)]
        H = qml.dot(coeffs, ops)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            qml.QDrift(H, time=1.2, n=10, seed=10)
            return qml.probs()

        res = circuit()
        expected = [0.65379493, 0.0, 0.34620507, 0.0]
        assert np.allclose(res, expected, atol=tol(dev.shots))

    def test_QFT(self, device, tol):
        """Test the QFT template."""
        wires = 3
        dev = device(wires)

        @qml.qnode(dev)
        def circuit_qft(state):
            qml.StatePrep(state, wires=range(wires))
            qml.QFT(wires=range(wires))
            return qml.probs()

        res = circuit_qft([0.8, 0.6] + [0.0] * 6)
        expected = [0.245, 0.20985281, 0.125, 0.04014719, 0.005, 0.04014719, 0.125, 0.20985281]
        assert np.allclose(res, expected, atol=tol(dev.shots))

    def test_QSVT(self, device, tol):
        """Test the QSVT template."""
        dev = device(2)
        A = np.array([[0.1]])
        block_encode = qml.BlockEncode(A, wires=[0, 1])
        check_op_supported(block_encode, dev)
        shifts = [qml.PCPhase(i + 0.1, dim=1, wires=[0, 1]) for i in range(3)]

        @qml.qnode(dev)
        def circuit():
            qml.QSVT(block_encode, shifts)
            return qml.expval(qml.Z(1))

        res = circuit()
        expected = 0.9370953557566887
        assert np.isclose(res, expected, atol=tol(dev.shots))

    def test_QuantumMonteCarlo(self, device, tol):
        """Test the QuantumMonteCarlo template."""
        m = 2
        M = 2**m

        xmax = np.pi  # bound to region [-pi, pi]
        xs = np.linspace(-xmax, xmax, M)

        probs = np.array([norm().pdf(x) for x in xs])
        probs /= np.sum(probs)

        def func(i):
            return np.sin(xs[i]) ** 2

        n = 3
        N = 2**n

        target_wires = range(m + 1)
        estimation_wires = range(m + 1, n + m + 1)
        dev = device(wires=n + m + 1)
        check_op_supported(qml.ControlledQubitUnitary(np.eye(2), [1], [0]), dev)

        @qml.qnode(dev)
        def circuit():
            qml.QuantumMonteCarlo(
                probs,
                func,
                target_wires=target_wires,
                estimation_wires=estimation_wires,
            )
            return qml.probs(estimation_wires)

        # pylint:disable=unsubscriptable-object
        phase_estimated = np.argmax(circuit()[: int(N / 2)]) / N
        res = (1 - np.cos(np.pi * phase_estimated)) / 2
        expected = 0.3086582838174551
        assert np.isclose(res, expected, atol=tol(dev.shots))

    def test_QuantumPhaseEstimation(self, device, tol):
        """Test the QuantumPhaseEstimation template."""
        unitary = qml.RX(np.pi / 2, wires=[0]) @ qml.CNOT(wires=[0, 1])
        eigenvector = np.array([-1 / 2, -1 / 2, 1 / 2, 1 / 2])

        n_estimation_wires = 3
        estimation_wires = range(2, n_estimation_wires + 2)
        target_wires = [0, 1]

        dev = device(wires=n_estimation_wires + 2)

        @qml.qnode(dev)
        def circuit():
            qml.StatePrep(eigenvector, wires=target_wires)
            qml.QuantumPhaseEstimation(
                unitary,
                estimation_wires=estimation_wires,
            )
            return qml.probs(estimation_wires)

        res = np.argmax(circuit()) / 2**n_estimation_wires
        expected = 0.125
        assert np.isclose(res, expected, atol=tol(dev.shots))

    def test_QutritBasisStatePreparation(self, device, tol):
        """Test the QutritBasisStatePreparation template."""
        dev = device(4)
        if "qutrit" not in dev.name:
            pytest.skip("QutritBasisState template only works on qutrit devices")

        @qml.qnode(dev)
        def circuit(basis_state, obs):
            qml.QutritBasisStatePreparation(basis_state, wires=range(4))
            return [qml.expval(qml.THermitian(obs, wires=i)) for i in range(4)]

        basis_state = [0, 1, 1, 0]
        obs = np.array([[1, 1, 0], [1, -1, 0], [0, 0, np.sqrt(2)]]) / np.sqrt(2)

        res = circuit(basis_state, obs)
        expected = np.array([1, -1, -1, 1]) / np.sqrt(2)
        assert np.allclose(res, expected, atol=tol(dev.shots))

    def test_RandomLayers(self, device, tol):
        """Test the RandomLayers template."""
        dev = device(2)
        weights = np.array([[0.1, -2.1, 1.4]])

        @qml.qnode(dev)
        def circuit(weights):
            qml.RandomLayers(weights=weights, wires=range(2), seed=42)
            return qml.expval(qml.Z(0))

        res = circuit(weights)
        expected = 0.9950041652780259
        assert np.isclose(res, expected, atol=tol(dev.shots))

    def test_Select(self, device, tol):
        """Test the Select template."""
        dev = device(4)
        check_op_supported(qml.MultiControlledX(wires=[0, 1, 2]), dev)

        ops = [qml.X(2), qml.X(3), qml.Y(2), qml.SWAP([2, 3])]

        @qml.qnode(dev)
        def circuit():
            qml.Select(ops, control=[0, 1])
            return qml.probs()

        res = circuit()
        expected = np.zeros(16)
        expected[2] = 1.0
        assert np.allclose(res, expected, atol=tol(dev.shots))

    def test_SimplifiedTwoDesign(self, device, tol):
        """Test the SimplifiedTwoDesign template."""
        n_wires = 3
        dev = device(n_wires)

        @qml.qnode(dev)
        def circuit(init_weights, weights):
            qml.SimplifiedTwoDesign(
                initial_layer_weights=init_weights, weights=weights, wires=range(n_wires)
            )
            return [qml.expval(qml.Z(i)) for i in range(n_wires)]

        init_weights = [np.pi, np.pi, np.pi]
        weights_layer1 = [[0.0, np.pi], [0.0, np.pi]]
        weights_layer2 = [[np.pi, 0.0], [np.pi, 0.0]]
        weights = [weights_layer1, weights_layer2]

        res = circuit(init_weights, weights)
        expected = [1.0, -1.0, 1.0]
        assert np.allclose(res, expected, atol=tol(dev.shots))

    @pytest.mark.xfail(reason="most devices do not support CV")
    def test_SqueezingEmbedding(self, device, tol):
        """Test the SqueezingEmbedding template."""
        dev = device(2)

        @qml.qnode(dev)
        def circuit(feature_vector):
            qml.SqueezingEmbedding(features=feature_vector, wires=range(3))
            qml.QuadraticPhase(0.1, wires=1)
            return qml.expval(qml.NumberOperator(wires=1))

        X = [1, 2, 3]

        res = circuit(X)
        expected = 13.018280763205285
        assert np.isclose(res, expected, atol=tol(dev.shots))

    def test_StronglyEntanglingLayers(self, device, tol):
        """Test the StronglyEntanglingLayers template."""
        dev = device(4)

        @qml.qnode(dev)
        def circuit(parameters):
            qml.StronglyEntanglingLayers(weights=parameters, wires=range(4))
            return qml.expval(qml.Z(0))

        shape = qml.StronglyEntanglingLayers.shape(n_layers=2, n_wires=4)
        params = np.arange(1, np.prod(shape) + 1).reshape(shape) / 10
        res = circuit(params)
        expected = -0.07273693957824906
        assert np.isclose(res, expected, atol=tol(dev.shots))

    def test_TTN(self, device, tol):
        """Test the TTN template."""

        def block(weights, wires):
            qml.CNOT(wires=[wires[0], wires[1]])
            qml.RY(weights[0], wires=wires[0])
            qml.RY(weights[1], wires=wires[1])

        n_wires = 4
        n_block_wires = 2
        n_params_block = 2
        n_blocks = qml.TTN.get_n_blocks(range(n_wires), n_block_wires)
        template_weights = [[0.1, -0.3]] * n_blocks
        dev = device(n_wires)

        @qml.qnode(dev)
        def circuit(template_weights):
            qml.TTN(range(n_wires), n_block_wires, block, n_params_block, template_weights)
            return qml.expval(qml.Z(n_wires - 1))

        res = circuit(template_weights)
        expected = 0.7845726663667097
        assert np.isclose(res, expected, atol=tol(dev.shots))

    def test_TrotterProduct(self, device, tol):
        """Test the TrotterProduct template."""
        dev = device(2)
        coeffs = [0.25, 0.75]
        ops = [qml.X(0), qml.Z(0)]
        H = qml.dot(coeffs, ops)

        @qml.qnode(dev)
        def circuit():
            qml.Hadamard(0)
            qml.TrotterProduct(H, time=2.4, order=2)
            return qml.probs()

        res = circuit()
        expected = [0.37506708, 0.0, 0.62493292, 0.0]
        assert np.allclose(res, expected, atol=tol(dev.shots))

    def test_TwoLocalSwapNetwork(self, device, tol):
        """Test the TwoLocalSwapNetwork template."""
        dev = device(3)

        def acquaintances(index, wires, param=None):  # pylint:disable=unused-argument
            return qml.CNOT(index)

        @qml.qnode(dev)
        def circuit(state):
            qml.StatePrep(state, range(3))
            qml.TwoLocalSwapNetwork(dev.wires, acquaintances, fermionic=True, shift=False)
            return qml.probs()

        state = np.arange(8, dtype=float)
        state /= np.linalg.norm(state)
        probs = state**2
        res = circuit(state)
        order = np.argsort(np.argsort(res))
        tol = tol(dev.shots)
        assert all(np.isclose(val, probs[i], atol=tol) for i, val in zip(order, res))


class TestMoleculeTemplates:
    """Test templates using the H2 molecule."""

    @pytest.fixture(scope="class")
    def h2(self):
        """Return attributes needed for H2."""
        symbols, coordinates = (["H", "H"], np.array([0.0, 0.0, -0.66140414, 0.0, 0.0, 0.66140414]))
        h, qubits = qml.qchem.molecular_hamiltonian(symbols, coordinates)
        electrons = 2
        ref_state = qml.qchem.hf_state(electrons, qubits)
        return qubits, ref_state, h

    def test_GateFabric(self, device, tol, h2):
        """Test the GateFabric template."""
        qubits, ref_state, H = h2
        dev = device(qubits)
        if getattr(dev, "short_name", None) == "cirq.mixedsimulator" and dev.shots:
            pytest.xfail(reason="device is generating negative probabilities")

        @qml.qnode(dev)
        def circuit(weights):
            qml.GateFabric(weights, wires=[0, 1, 2, 3], init_state=ref_state, include_pi=True)
            return qml.expval(H)

        layers = 2
        shape = qml.GateFabric.shape(n_layers=layers, n_wires=qubits)
        weights = np.array([0.1, 0.2, 0.3, 0.4]).reshape(shape)
        res = circuit(weights)
        expected = -0.9453094224618628
        assert np.isclose(res, expected, atol=tol(dev.shots))

    def test_ParticleConservingU1(self, device, tol, h2):
        """Test the ParticleConservingU1 template."""
        qubits, ref_state, h = h2
        dev = device(qubits)
        ansatz = partial(qml.ParticleConservingU1, init_state=ref_state, wires=dev.wires)

        @qml.qnode(dev)
        def circuit(params):
            ansatz(params)
            return qml.expval(h)

        layers = 2
        shape = qml.ParticleConservingU1.shape(layers, qubits)
        params = np.arange(1, 13).reshape(shape) / 10
        res = circuit(params)
        expected = -0.5669084184194393
        assert np.isclose(res, expected, atol=tol(dev.shots))

    def test_ParticleConservingU2(self, device, tol, h2):
        """Test the ParticleConservingU2 template."""
        qubits, ref_state, h = h2
        dev = device(qubits)
        ansatz = partial(qml.ParticleConservingU2, init_state=ref_state, wires=dev.wires)

        @qml.qnode(dev)
        def circuit(params):
            ansatz(params)
            return qml.expval(h)

        layers = 1
        shape = qml.ParticleConservingU2.shape(layers, qubits)
        params = np.arange(1, 8).reshape(shape) / 10
        res = circuit(params)
        expected = -0.8521967086461301
        assert np.isclose(res, expected, atol=tol(dev.shots))

    def test_UCCSD(self, device, tol, h2):
        """Test the UCCSD template."""
        qubits, hf_state, H = h2
        electrons = 2
        singles, doubles = qml.qchem.excitations(electrons, qubits)
        s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
        dev = device(qubits)

        @qml.qnode(dev)
        def circuit(params, wires, s_wires, d_wires, hf_state):
            qml.UCCSD(params, wires, s_wires, d_wires, hf_state)
            return qml.expval(H)

        params = np.arange(len(singles) + len(doubles)) / 4
        res = circuit(
            params, wires=range(qubits), s_wires=s_wires, d_wires=d_wires, hf_state=hf_state
        )
        expected = -1.0864433121798176
        assert np.isclose(res, expected, atol=tol(dev.shots))

    def test_kUpCCGSD(self, device, tol, h2):
        """Test the kUpCCGSD template."""
        qubits, ref_state, H = h2
        dev = device(qubits)
        if getattr(dev, "short_name", None) == "cirq.mixedsimulator" and dev.shots:
            pytest.xfail(reason="device is generating negative probabilities")

        @qml.qnode(dev)
        def circuit(weights):
            qml.kUpCCGSD(weights, wires=[0, 1, 2, 3], k=1, delta_sz=0, init_state=ref_state)
            return qml.expval(H)

        layers = 1
        shape = qml.kUpCCGSD.shape(k=layers, n_wires=qubits, delta_sz=0)
        weights = np.arange(np.prod(shape)).reshape(shape) / 10
        res = circuit(weights)
        expected = -1.072648130451027
        assert np.isclose(res, expected, atol=tol(dev.shots))
