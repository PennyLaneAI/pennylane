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

np.random.seed(777)


def max_entangled_circuit(wires, shots=10000, interface="autograd"):
    """maximally entangled state preparation circuit"""
    dev = qml.device("default.qubit", wires=wires, shots=shots)

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

        entropies = [
            shadow.entropy(wires=[0], alpha=alpha, atol=1e-2, base=base) for alpha in [1, 2, 3]
        ]
        assert np.allclose(entropies, entropies[0], atol=1e-2)
        expected = np.log(2) / np.log(base)
        assert np.allclose(entropies, expected, atol=1e-2)

    def test_non_constant_distribution(
        self,
    ):
        """Test entropies match roughly with exact solution for a non-constant distribution using other PennyLane functionalities"""
        n_wires = 4
        # exact solution
        dev = qml.device("default.qubit", wires=range(n_wires), shots=100000)

        @qml.qnode(dev)
        def qnode_exact(x):
            for i in range(n_wires):
                qml.RY(x[i], wires=i)

            for i in range(n_wires - 1):
                qml.CNOT((i, i + 1))

            return qml.state()

        # classical shadow qnode
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

        # Check for the correct entropies for all possible 2-site reduced density matrix (rdm)
        for rdm_wires in [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]:
            # this is intentionally not done in a parametrize loop because this would re-execute the quantum function

            # exact solution
            rdm = qml.qinfo.reduced_dm(qnode_exact, wires=rdm_wires)(x)
            evs = qml.math.eigvalsh(rdm)
            print(np.round(evs, 3))
            evs = evs[np.where(evs > 0)]

            exact_2 = -np.log(np.trace(rdm @ rdm))

            alpha = 1.5
            exact_alpha = qml.math.log(qml.math.sum(evs**alpha)) / (1 - alpha)

            exact_vn = qml.math.entr(evs)
            exact = [exact_vn, exact_alpha, exact_2]

            # shadow estimate
            entropies = [
                shadow.entropy(wires=rdm_wires, alpha=alpha, atol=1e-10) for alpha in [1, 1.5, 2]
            ]

            assert np.allclose(entropies, exact, atol=1e-1)

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize("interface", ["autograd", "torch", "tf", "jax"])
    def test_analytic_entropy(self, interface):
        """Test entropies on analytic results"""
        dev = qml.device("default.qubit", wires=2, shots=100000)

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
