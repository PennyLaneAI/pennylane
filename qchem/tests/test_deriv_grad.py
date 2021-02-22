import numpy as np
import pytest

import pennylane as qml
from pennylane import qchem

import shutil


delta0 = 0.00529
x0 = np.array([-0.0211, -0.002, 0.0, 0.8345, 0.4519, 0.0, 1.4769, -0.273, 0.0])
coeffs0 = np.array(
    [
        0.27770444969014774,
        -0.0017650003476157269,
        -0.0017650003476185035,
        -0.12236476454403046,
        -0.12236476454402768,
        0.001633367057292669,
        -0.001633367057292669,
        -0.001633367057292669,
        0.001633367057292669,
        0.006294156274917891,
        0.007927523332212817,
        0.007927523332212817,
        0.006294156274917891,
        0.008903324791978307,
    ]
)

delta1 = 0.08
x1 = np.array([-0.028, -0.001, 0.0, 0.79, 0.43, 0.0, 1.5, -0.3, 0.0])
coeffs1 = np.array(
    [
        0.07943488923762594,
        0.000684379865657387,
        0.0006843798656566526,
        -0.024610457680021434,
        -0.024610457680021066,
        -1.745329030477912e-05,
        1.745329030477912e-05,
        1.745329030477912e-05,
        -1.745329030477912e-05,
        -0.00381901315026642,
        -0.0038364664405712797,
        -0.0038364664405712797,
        -0.00381901315026642,
        0.01401540609625499,
    ]
)

ops = [
    qml.Identity(wires=[0]),
    qml.PauliZ(wires=[0]),
    qml.PauliZ(wires=[1]),
    qml.PauliZ(wires=[2]),
    qml.PauliZ(wires=[3]),
    qml.PauliY(wires=[0]) @ qml.PauliX(wires=[1]) @ qml.PauliX(wires=[2]) @ qml.PauliY(wires=[3]),
    qml.PauliY(wires=[0]) @ qml.PauliY(wires=[1]) @ qml.PauliX(wires=[2]) @ qml.PauliX(wires=[3]),
    qml.PauliX(wires=[0]) @ qml.PauliX(wires=[1]) @ qml.PauliY(wires=[2]) @ qml.PauliY(wires=[3]),
    qml.PauliX(wires=[0]) @ qml.PauliY(wires=[1]) @ qml.PauliY(wires=[2]) @ qml.PauliX(wires=[3]),
    qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[2]),
    qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[3]),
    qml.PauliZ(wires=[1]) @ qml.PauliZ(wires=[2]),
    qml.PauliZ(wires=[1]) @ qml.PauliZ(wires=[3]),
    qml.PauliZ(wires=[2]) @ qml.PauliZ(wires=[3]),
]


@pytest.mark.parametrize(
    ("i", "x", "delta", "coeffs", "ops"),
    [(0, x0, delta0, coeffs0, ops), (1, x1, delta1, coeffs1, ops), (2, x0, delta0, [], [])],
)
def test_deriv(i, x, delta, coeffs, ops, tol):
    r"""Tests the correctness of the nuclear derivative of the electronic Hamiltonian
    computed by the ``'derivative'`` function. 
    """

    def H(x):
        return qchem.molecular_hamiltonian(
            ["H", "O", "H"], x, active_electrons=2, active_orbitals=2
        )[0]

    deriv = qchem.derivative(H, x, i, delta=delta)

    shutil.rmtree("pyscf")

    exp_obs = qml.vqe.Hamiltonian(coeffs, ops)

    assert np.allclose(deriv.coeffs, exp_obs.coeffs, **tol)
    assert all(isinstance(o1, o2.__class__) for o1, o2 in zip(deriv.ops, exp_obs.ops))
    assert all(o1.wires == o2.wires for o1, o2 in zip(deriv.ops, exp_obs.ops))


def test_grad(tol):
    r"""Tests function `'gradient'` computing the gradient of the electronic Hamiltonian H(x)."""

    dev = qml.device("default.qubit", wires=4)

    def circuit(param, wires):
        qml.BasisState(np.array([1, 1, 0, 0]), wires=[0, 1, 2, 3])
        qml.RY(param, wires=2)
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[2, 0])
        qml.CNOT(wires=[3, 1])

    def H(x):
        return qchem.molecular_hamiltonian(["H", "H"], x)[0]

    x = np.array([0.0, 0.0, 0.35, 0.0, 0.0, -0.35])

    grad_h = qchem.gradient(H, x)

    shutil.rmtree("pyscf")

    assert len(grad_h) == x.size

    param = 6.07230111451844
    exp_grad_E = np.array([0.0, 0.0, -0.03540190344912991, 0.0, 0.0, 0.03540190344912991])

    calc_grad_E = np.array([qml.ExpvalCost(circuit, grad, dev)(param) for grad in grad_h])

    assert np.allclose(exp_grad_E, calc_grad_E, **tol)
