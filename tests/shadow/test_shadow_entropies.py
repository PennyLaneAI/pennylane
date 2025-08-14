# Copyright 2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for shadow entropies"""
# pylint:disable=no-self-use, import-outside-toplevel, redefined-outer-name, unpacking-non-sequence, too-few-public-methods, not-an-iterable, inconsistent-return-statements

import pytest

import pennylane as qml
import pennylane.numpy as np
from pennylane.shadows import ClassicalShadow
from pennylane.shadows.classical_shadow import _project_density_matrix_spectrum


def max_entangled_circuit(wires, shots=10000, interface="autograd"):
    """maximally entangled state preparation circuit"""
    dev = qml.device("default.qubit", wires=wires)

    @qml.set_shots(shots)
    @qml.qnode(dev, interface=interface)
    def circuit():
        qml.Hadamard(wires=0)
        for i in range(1, wires):
            qml.CNOT(wires=[0, i])
        return qml.classical_shadow(wires=range(wires))

    return circuit


def expected_entropy_ising_xx(param, alpha):
    """
    Return the analytical entropy for the IsingXX.
    """
    eig_1 = (1 + np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
    eig_2 = (1 - np.sqrt(1 - 4 * np.cos(param / 2) ** 2 * np.sin(param / 2) ** 2)) / 2
    eigs = [eig_1, eig_2]
    eigs = np.array([eig for eig in eigs if eig > 0])

    if alpha == 1 or qml.math.isclose(alpha, 1):
        return qml.math.entr(eigs)

    return qml.math.log(qml.math.sum(eigs**alpha)) / (1 - alpha)


class TestShadowEntropies:
    """Tests for entropies in ClassicalShadow class"""

    @pytest.mark.parametrize("n_wires", [2, 4])
    @pytest.mark.parametrize("base", [np.exp(1), 2])
    def test_constant_distribution(self, n_wires, base):
        """Test for state with constant eigenvalues of reduced state that all entropies are the same"""

        bits, recipes = max_entangled_circuit(wires=n_wires)()
        shadow = ClassicalShadow(bits, recipes)

        entropies = [shadow.entropy(wires=[0], alpha=alpha, base=base) for alpha in [1, 2, 3]]
        assert np.allclose(entropies, entropies[0], atol=1e-2)
        expected = np.log(2) / np.log(base)
        assert np.allclose(entropies, expected, atol=2e-2)

    def test_non_constant_distribution(self):
        """Test entropies match roughly with exact solution for a non-constant distribution using other PennyLane functionalities"""
        n_wires = 4
        # exact solution
        dev_exact = qml.device("default.qubit", wires=range(n_wires))
        dev = qml.device("default.qubit", wires=range(n_wires))

        @qml.qnode(dev_exact)
        def qnode_exact(x):
            for i in range(n_wires):
                qml.RY(x[i], wires=i)

            for i in range(n_wires - 1):
                qml.CNOT((i, i + 1))

            return qml.state()

        # classical shadow qnode
        @qml.set_shots(100000)
        @qml.qnode(dev)
        def qnode(x):
            for i in range(n_wires):
                qml.RY(x[i], wires=i)

            for i in range(n_wires - 1):
                qml.CNOT((i, i + 1))

            return qml.classical_shadow(wires=dev.wires)

        x = np.arange(n_wires, requires_grad=True)

        bitstrings, recipes = qnode(x)
        shadow = ClassicalShadow(bitstrings, recipes)

        # Get the full dm of the qnode_exact
        state = qnode_exact(x)
        rho = qml.math.dm_from_state_vector(state)

        # Check for the correct entropies for all possible 2-site reduced density matrix (rdm)
        for rdm_wires in [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]:
            # this is intentionally not done in a parametrize loop because this would re-execute the quantum function

            # exact solution
            rdm = qml.math.reduce_dm(rho, indices=rdm_wires)
            evs = qml.math.eigvalsh(rdm)

            evs = evs[np.where(evs > 0)]

            exact_2 = -np.log(np.trace(rdm @ rdm))

            alpha = 1.5
            exact_alpha = qml.math.log(qml.math.sum(evs**alpha)) / (1 - alpha)

            exact_vn = qml.math.entr(evs)
            exact = [exact_vn, exact_alpha, exact_2]

            # shadow estimate
            entropies = [shadow.entropy(wires=rdm_wires, alpha=alpha) for alpha in [1, 1.5, 2]]

            assert np.allclose(entropies, exact, atol=1e-1)

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["autograd", "torch", "jax"])
    def test_analytic_entropy(self, interface):
        """Test entropies on analytic results"""
        dev = qml.device("default.qubit", wires=2)

        @qml.set_shots(100000)
        @qml.qnode(dev, interface=interface)
        def circuit():
            qml.IsingXX(0.5, wires=[0, 1])
            return qml.classical_shadow(wires=range(2))

        param = 0.5
        bits, recipes = circuit()
        shadow = qml.ClassicalShadow(bits, recipes)

        # explicitly not use pytest parametrize to reuse the same measurements
        for alpha in [1, 2, 3]:
            for base in [2, np.exp(1)]:
                for reduced_wires in [[0], [1]]:
                    entropy = shadow.entropy(wires=reduced_wires, base=base, alpha=alpha)

                    expected_entropy = expected_entropy_ising_xx(param, alpha) / np.log(base)

                    assert qml.math.allclose(entropy, expected_entropy, atol=1e-1)

    def test_closest_density_matrix(self):
        """Test that the closest density matrix from the estimator is valid"""
        wires = 3
        dev = qml.device("default.qubit", wires=range(wires))

        # Just a simple circuit
        @qml.qnode(dev)
        def qnode(x):
            for i in range(wires):
                qml.RY(x[i], wires=i)

            for i in range(wires - 1):
                qml.CNOT((i, i + 1))

            return qml.state()

        x = np.linspace(0.5, 1.5, num=wires)
        state = qnode(x)
        rho = qml.math.dm_from_state_vector(state)
        lambdas = _project_density_matrix_spectrum(rho)
        assert np.isclose(np.sum(lambdas), 1.0)
        assert all(lambdas > 0)


rho0 = np.zeros((2**3, 2**3))
rho0[0, 0] = 1.0

rho1 = np.diag([-0.1, -0.1, -0.1, 1.3])
rho2 = np.diag([-0.1, -0.1, 0.1, 1.1])


@pytest.mark.parametrize("rho", [rho0, rho1, rho2])
def test_project_density_matrix_spectrum(rho):
    """Test the function _project_density_matrix_spectrum behaves as expected for trivial case"""
    new_lambdas = _project_density_matrix_spectrum(rho)
    assert qml.math.allclose(new_lambdas, [1.0])
