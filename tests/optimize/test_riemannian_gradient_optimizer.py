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
Unit tests for the ``RiemannianGradientOptimizer``.
"""
import numpy as np
import pytest
from scipy.sparse.linalg import expm

import pennylane as qp
from pennylane.optimize import RiemannianGradientOptimizer


def circuit_1():
    """Simple circuit."""
    qp.Hadamard(wires=[0])
    qp.Hadamard(wires=[1])


def circuit_2():
    """Simply parametrized circuit."""
    qp.RX(0.1, wires=[0])
    qp.RY(0.5, wires=[1])
    qp.CNOT(wires=[0, 1])
    qp.RY(0.6, wires=[0])


def circuit_3():
    """Three-qubit circuit."""
    qp.RY(0.5, wires=[0])
    qp.RY(0.6, wires=[1])
    qp.RY(0.7, wires=[2])
    qp.CNOT(wires=[0, 1])
    qp.CNOT(wires=[1, 2])
    qp.RX(-0.6, wires=[0])
    qp.RX(-0.3, wires=[1])
    qp.RX(-0.2, wires=[2])


hamiltonian_1 = qp.Hamiltonian(
    coeffs=[-1.0] * 3,
    observables=[qp.PauliX(0), qp.PauliZ(1), qp.PauliY(0) @ qp.PauliX(1)],
)

hamiltonian_2 = qp.Hamiltonian(
    coeffs=[-0.2, 0.3, -0.15],
    observables=[
        qp.PauliY(1),
        qp.PauliZ(0) @ qp.PauliZ(1),
        qp.PauliX(0) @ qp.PauliX(1),
    ],
)

hamiltonian_3 = qp.Hamiltonian(
    coeffs=[-2.0], observables=[qp.PauliY(0) @ qp.PauliY(1) @ qp.PauliY(2)]
)


@pytest.mark.parametrize(
    "circuit,hamiltonian",
    [
        (circuit_1, hamiltonian_1),
        (circuit_1, hamiltonian_2),
        (circuit_2, hamiltonian_1),
        (circuit_2, hamiltonian_2),
        (circuit_3, hamiltonian_3),
    ],
)
def test_riemannian_gradient_omegas(circuit, hamiltonian):
    """Test that we calculate the Riemannian gradient coefficients Tr{[rho, H] P_j} correctly."""
    # pylint: disable=no-member

    nqubits = max(max(ps.wires) for ps in hamiltonian.ops) + 1
    wires = range(nqubits)
    dev = qp.device("default.qubit", wires=nqubits)

    @qp.qnode(dev)
    def get_state():
        circuit()
        return qp.state()

    @qp.qnode(dev)
    def test_circuit():
        circuit()
        return qp.expval(hamiltonian)

    phi = get_state()
    rho = np.outer(phi, phi.conj())
    hamiltonian_np = hamiltonian.sparse_matrix(wire_order=wires).toarray()
    riemannian_gradient_np = hamiltonian_np @ rho - rho @ hamiltonian_np
    opt = RiemannianGradientOptimizer(circuit=test_circuit)
    ops = opt.get_su_n_operators(None)[0]
    omegas_np = []
    for op in ops:
        op = qp.math.expand_matrix(op.matrix(), op.wires, wires)
        omegas_np.append(1j * np.trace(riemannian_gradient_np @ op))
    omegas = opt.get_omegas()
    assert np.allclose(omegas, omegas_np)


@pytest.mark.parametrize(
    "circuit,hamiltonian",
    [
        (circuit_1, hamiltonian_1),
        (circuit_1, hamiltonian_2),
        (circuit_2, hamiltonian_1),
        (circuit_2, hamiltonian_2),
        (circuit_3, hamiltonian_3),
    ],
)
def test_riemannian_gradient_omegas_restricted(circuit, hamiltonian):
    """Test that we calculate the (restricted) Riemannian gradient coefficients correctly."""
    # pylint: disable=no-member
    nqubits = max(max(ps.wires) for ps in hamiltonian.ops) + 1
    wires = range(nqubits)
    dev = qp.device("default.qubit", wires=nqubits)

    @qp.qnode(dev)
    def get_state():
        circuit()
        return qp.state()

    @qp.qnode(dev)
    def test_circuit():
        circuit()
        return qp.expval(hamiltonian)

    phi = get_state()
    rho = np.outer(phi, phi.conj())
    hamiltonian_np = hamiltonian.sparse_matrix(wire_order=wires).toarray()
    riemannian_gradient_np = hamiltonian_np @ rho - rho @ hamiltonian_np

    restriction = qp.Hamiltonian(
        coeffs=[1.0] * 3,
        observables=[qp.PauliX(0), qp.PauliY(1), qp.PauliY(0) @ qp.PauliY(1)],
    )

    opt = RiemannianGradientOptimizer(circuit=test_circuit, restriction=restriction)
    ops = opt.get_su_n_operators(restriction)[0]
    omegas_np = []
    for op in ops:
        op = qp.math.expand_matrix(op.matrix(), op.wires, wires)
        omegas_np.append(1j * np.trace(riemannian_gradient_np @ op))
    omegas = opt.get_omegas()

    assert np.allclose(omegas, omegas_np)


@pytest.mark.parametrize(
    "circuit,hamiltonian",
    [
        (circuit_1, hamiltonian_1),
        (circuit_1, hamiltonian_2),
        (circuit_2, hamiltonian_1),
        (circuit_3, hamiltonian_3),
    ],
)
def test_riemannian_gradient_evolution(circuit, hamiltonian):
    """Test that the optimizer produces the correct unitary to append."""
    # pylint: disable=no-member
    nqubits = max(max(ps.wires) for ps in hamiltonian.ops) + 1
    wires = range(nqubits)
    dev = qp.device("default.qubit", wires=nqubits)

    @qp.qnode(dev)
    def get_state():
        circuit()
        return qp.state()

    @qp.qnode(dev)
    def test_circuit():
        circuit()
        return qp.expval(hamiltonian)

    phi = get_state()
    rho = np.outer(phi, phi.conj())
    hamiltonian_np = hamiltonian.sparse_matrix(wire_order=wires).toarray()
    riemannian_gradient_np = hamiltonian_np @ rho - rho @ hamiltonian_np

    phi_exact = expm(-0.1 * riemannian_gradient_np * 2**nqubits) @ phi
    rho_exact = np.outer(phi_exact, phi_exact.conj())
    opt = RiemannianGradientOptimizer(circuit=test_circuit, stepsize=0.1, exact=True)
    opt.step_and_cost()
    cost_pl = opt.circuit()
    cost_exact = np.trace(rho_exact @ hamiltonian_np)
    assert np.allclose(cost_pl, cost_exact, atol=1e-4)


@pytest.mark.parametrize(
    "circuit,hamiltonian",
    [
        (circuit_1, hamiltonian_1),
        (circuit_1, hamiltonian_2),
        (circuit_2, hamiltonian_1),
        (circuit_2, hamiltonian_2),
        (circuit_3, hamiltonian_3),
    ],
)
def test_riemannian_gradient_step(circuit, hamiltonian):
    """Test that we can take subsequent steps with the optimizer."""
    nqubits = max(max(ps.wires) for ps in hamiltonian.ops) + 1

    dev = qp.device("default.qubit", wires=nqubits)

    @qp.qnode(dev)
    def test_circuit():
        circuit()
        return qp.expval(hamiltonian)

    opt = RiemannianGradientOptimizer(circuit=test_circuit)
    opt.step()
    opt.step()


@pytest.mark.parametrize(
    "circuit,hamiltonian",
    [
        (circuit_1, hamiltonian_1),
        (circuit_1, hamiltonian_2),
        (circuit_2, hamiltonian_1),
        (circuit_2, hamiltonian_2),
        (circuit_3, hamiltonian_3),
    ],
)
def test_riemannian_gradient_step_trotterstep(circuit, hamiltonian):
    """Test that we can take subsequent steps with the optimizer."""
    nqubits = max(max(ps.wires) for ps in hamiltonian.ops) + 1

    dev = qp.device("default.qubit", wires=nqubits)

    @qp.qnode(dev)
    def test_circuit():
        circuit()
        return qp.expval(hamiltonian)

    opt = RiemannianGradientOptimizer(circuit=test_circuit, trottersteps=3)
    opt.step()
    opt.step()


def test_riemannian_gradient_circuit_input_1_check():
    """Test that a type error is raise for non-QNode circuits."""

    def circuit():
        qp.RY(0.5, wires=0)

    with pytest.raises(TypeError, match="circuit must be a QNode"):
        RiemannianGradientOptimizer(circuit=circuit, stepsize=0.001)


def test_riemannian_gradient_hamiltonian_input_1_check():
    """Test that a type error is raise for non-QNode circuits."""

    @qp.qnode(qp.device("default.qubit", wires=3))
    def circuit():
        qp.RY(0.5, wires=0)
        return qp.state()

    with pytest.raises(
        TypeError,
        match="circuit must return the expectation value of a Hamiltonian",
    ):
        RiemannianGradientOptimizer(circuit=circuit, stepsize=0.001)


def test_riemannian_gradient_nqubits_check():
    """Test that we warn if the system is too big."""

    @qp.qnode(qp.device("default.qubit", wires=5))
    def circuit():
        qp.RY(0.5, wires=0)
        return qp.expval(qp.Hamiltonian(coeffs=[-1.0], observables=[qp.PauliX(0)]))

    with pytest.warns(UserWarning, match="The exact Riemannian gradient is exponentially"):
        RiemannianGradientOptimizer(circuit=circuit, stepsize=0.001)


def test_riemannian_gradient_restriction_check():
    """Test that a type error is raise for non-QNode circuits."""

    @qp.qnode(qp.device("default.qubit", wires=3))
    def circuit():
        qp.RY(0.5, wires=0)
        return qp.expval(qp.Hamiltonian(coeffs=[-1.0], observables=[qp.PauliX(0)]))

    restriction = "not_a_hamiltonian"
    with pytest.raises(
        TypeError,
        match="restriction must be a Hamiltonian",
    ):
        RiemannianGradientOptimizer(circuit=circuit, restriction=restriction, stepsize=0.001)


@pytest.mark.slow
def test_docstring_example():
    """Test the docstring example with Trotterized evolution."""
    hamiltonian = qp.Hamiltonian(
        coeffs=[-1.0] * 3,
        observables=[qp.PauliX(0), qp.PauliZ(1), qp.PauliY(0) @ qp.PauliX(1)],
    )

    @qp.qnode(qp.device("default.qubit", wires=2))
    def quant_fun():
        qp.RX(0.1, wires=[0])
        qp.RY(0.5, wires=[1])
        qp.CNOT(wires=[0, 1])
        qp.RY(0.6, wires=[0])
        return qp.expval(hamiltonian)

    opt = RiemannianGradientOptimizer(circuit=quant_fun, stepsize=0.1)

    for _ in range(12):
        circuit, cost = opt.step_and_cost()
    circuit()
    assert np.isclose(cost, -2.236068, atol=1e-3)


def test_docstring_example_exact():
    """Test that the optimizer works with matrix exponential."""

    hamiltonian = qp.Hamiltonian(
        coeffs=[-1.0] * 3,
        observables=[qp.PauliX(0), qp.PauliZ(1), qp.PauliY(0) @ qp.PauliX(1)],
    )

    @qp.qnode(qp.device("default.qubit", wires=2))
    def quant_fun():
        qp.RX(0.1, wires=[0])
        qp.RY(0.5, wires=[1])
        qp.CNOT(wires=[0, 1])
        qp.RY(0.6, wires=[0])
        return qp.expval(hamiltonian)

    opt = RiemannianGradientOptimizer(circuit=quant_fun, stepsize=0.1, exact=True)

    for _ in range(12):
        circuit, cost = opt.step_and_cost()
    circuit()
    assert np.isclose(cost, -2.236068, atol=1e-3)


def test_example_shots():
    """Test that the optimizer works with finite shots."""
    hamiltonian = qp.Hamiltonian(
        coeffs=[-1.0] * 3,
        observables=[qp.PauliX(0), qp.PauliZ(1), qp.PauliY(0) @ qp.PauliX(1)],
    )

    @qp.qnode(qp.device("default.qubit", wires=2), diff_method=None, shots=1000)
    def quant_fun():
        qp.RX(0.1, wires=[0])
        qp.RY(0.5, wires=[1])
        qp.CNOT(wires=[0, 1])
        qp.RY(0.6, wires=[0])
        return qp.expval(hamiltonian)

    opt = RiemannianGradientOptimizer(circuit=quant_fun, stepsize=0.1, exact=False)

    for _ in range(3):
        opt.step_and_cost()
