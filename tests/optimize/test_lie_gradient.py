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
Unit tests for the ``LieGradientOptimizer``.
"""
import pytest

from scipy.sparse.linalg import expm

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize.lie_gradient import LieGradientOptimizer


@qml.qnode(qml.device("default.qubit", wires=2))
def circuit_1():
    """Simple circuit"""
    qml.Hadamard(wires=[0])
    qml.Hadamard(wires=[1])
    return qml.state()


@qml.qnode(qml.device("default.qubit", wires=2))
def circuit_2():
    """Simply parameterized circuit"""
    qml.RX(0.1, wires=[0])
    qml.RY(0.5, wires=[1])
    qml.CNOT(wires=[0, 1])
    qml.RY(0.6, wires=[0])
    return qml.state()


@qml.qnode(qml.device("default.qubit", wires=3))
def circuit_3():
    """Three-qubit circuit"""
    qml.RY(0.5, wires=[0])
    qml.RY(0.6, wires=[1])
    qml.RY(0.7, wires=[2])
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.RX(-0.6, wires=[0])
    qml.RX(-0.3, wires=[1])
    qml.RX(-0.2, wires=[2])
    return qml.state()


hamiltonian_1 = qml.Hamiltonian(
    coeffs=[-1.0] * 3,
    observables=[qml.PauliX(0), qml.PauliZ(1), qml.PauliY(0) @ qml.PauliX(1)],
)

hamiltonian_2 = qml.Hamiltonian(
    coeffs=[-0.2, 0.3, -0.15],
    observables=[
        qml.PauliY(1),
        qml.PauliZ(0) @ qml.PauliZ(1),
        qml.PauliX(0) @ qml.PauliX(1),
    ],
)

hamiltonian_3 = qml.Hamiltonian(
    coeffs=[-2.0], observables=[qml.PauliY(0) @ qml.PauliY(1) @ qml.PauliY(2)]
)


@pytest.mark.parametrize(
    "circuit,hamiltonian",
    [
        (circuit_1, hamiltonian_1),
        (circuit_1, hamiltonian_2),
        (circuit_2, hamiltonian_1),
        (circuit_2, hamiltonian_2),
        (circuit_3, hamiltonian_1),
        (circuit_3, hamiltonian_2),
        (circuit_3, hamiltonian_3),
    ],
)
def test_lie_gradient_omegas(circuit, hamiltonian):
    """Test that we calculate the lie gradient coefficients Tr{[rho, H] P_j} correctly"""
    wires = circuit.device.wires
    phi = circuit()
    rho = np.outer(phi, phi.conj())
    hamiltonian_np = qml.utils.sparse_hamiltonian(hamiltonian, wires).toarray()
    lie_gradient_np = hamiltonian_np @ rho - rho @ hamiltonian_np
    opt = LieGradientOptimizer(circuit=circuit, hamiltonian=hamiltonian)
    ops = opt.get_su_n_operators()[0]
    omegas_np = []
    for op in ops:
        op = qml.utils.expand(op.matrix, op.wires, wires)
        omegas_np.append(-np.trace(lie_gradient_np @ op).imag / 2)
    omegas = opt.get_omegas()
    assert np.allclose(omegas, omegas_np)


@pytest.mark.parametrize(
    "circuit,hamiltonian",
    [
        (circuit_1, hamiltonian_1),
        (circuit_1, hamiltonian_2),
        (circuit_2, hamiltonian_1),
        (circuit_2, hamiltonian_2),
        (circuit_3, hamiltonian_1),
        (circuit_3, hamiltonian_2),
        (circuit_3, hamiltonian_3),
    ],
)
def test_lie_gradient_evolution(circuit, hamiltonian):
    """Test that the optimizer produces the correct unitary to append"""
    wires = circuit.device.wires
    phi = circuit()
    rho = np.outer(phi, phi.conj())
    hamiltonian_np = qml.utils.sparse_hamiltonian(hamiltonian, wires).toarray()
    lie_gradient_np = hamiltonian_np @ rho - rho @ hamiltonian_np

    phi_exact = expm(0.001 * lie_gradient_np) @ phi
    rho_exact = np.outer(phi_exact, phi_exact.conj())
    opt = LieGradientOptimizer(circuit=circuit, hamiltonian=hamiltonian, stepsize=0.001)
    opt.step()

    phi_pl = opt.circuit([], [])

    rho_pl = np.outer(phi_pl, phi.conj())
    assert np.allclose(rho_exact, rho_pl, atol=1e-2)


@pytest.mark.parametrize(
    "circuit,hamiltonian",
    [
        (circuit_1, hamiltonian_1),
        (circuit_1, hamiltonian_2),
        (circuit_2, hamiltonian_1),
        (circuit_2, hamiltonian_2),
        (circuit_3, hamiltonian_1),
    ],
)
def test_list_gradient_step(circuit, hamiltonian):
    """Test that we can take subsequent steps with the optimizer"""
    opt = LieGradientOptimizer(circuit=circuit, hamiltonian=hamiltonian, stepsize=0.1)
    opt.step()
    opt.step()


def test_lie_gradient_circuit_input_1():
    """Test that a type error is raise for non-QNode circuits"""

    def circuit():
        qml.RY(0.5, wires=0)

    hamiltonian = qml.Hamiltonian(coeffs=[-1.0], observables=[qml.PauliX(0)])
    with pytest.raises(TypeError, match="`circuit` must be a `qml.QNode`"):
        LieGradientOptimizer(circuit=circuit, hamiltonian=hamiltonian, stepsize=0.001)


def test_lie_gradient_circuit_input_2():
    """Test that the optimizer does not care about what is returned"""

    @qml.qnode(qml.device("default.qubit", wires=3))
    def circuit():
        qml.RY(0.5, wires=0)
        return qml.expval(qml.PauliX(0))

    hamiltonian = qml.Hamiltonian(coeffs=[-1.0], observables=[qml.PauliX(0)])
    opt = LieGradientOptimizer(circuit=circuit, hamiltonian=hamiltonian, stepsize=0.001)
    opt.step()


def test_lie_gradient_hamiltonian_input_1():
    """Test that a type error is raise for non-QNode circuits"""

    @qml.qnode(qml.device("default.qubit", wires=3))
    def circuit():
        qml.RY(0.5, wires=0)
        return qml.state()

    hamiltonian = qml.PauliX(0)
    with pytest.raises(
        TypeError, match="`hamiltonian` must be a `qml.Hamiltonian`"
    ):
        LieGradientOptimizer(circuit=circuit, hamiltonian=hamiltonian, stepsize=0.001)


def test_lie_gradient_nqubits(capsys):
    """Test that a type error is raise for non-QNode circuits"""

    @qml.qnode(qml.device("default.qubit", wires=5))
    def circuit():
        qml.RY(0.5, wires=0)
        return qml.state()

    hamiltonian = qml.Hamiltonian(coeffs=[-1.0], observables=[qml.PauliX(0)])
    LieGradientOptimizer(circuit=circuit, hamiltonian=hamiltonian, stepsize=0.001)
    out, _ = capsys.readouterr()
    assert out.startswith("WARNING: The exact Lie gradient is exponentially")
