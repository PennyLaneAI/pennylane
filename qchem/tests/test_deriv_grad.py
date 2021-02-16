import numpy as np
import pytest

import pennylane as qml
from pennylane import qchem

import shutil


non_zero_coeffs = np.array(
    [
        -0.7763135743293005,
        -0.08534360840293387,
        -0.08534360840293387,
        0.2669341092545041,
        0.26693410925450134,
        -0.025233628744274508,
        0.0072162443961340415,
        -0.0072162443961340415,
        -0.0072162443961340415,
        0.0072162443961340415,
        -0.030654287745411964,
        -0.023438043349280003,
        -0.023438043349280003,
        -0.030654287745411964,
        -0.02494407786332001,
    ]
)
non_zero_ops = [
    qml.Identity(wires=[0]),
    qml.PauliZ(wires=[0]),
    qml.PauliZ(wires=[1]),
    qml.PauliZ(wires=[2]),
    qml.PauliZ(wires=[3]),
    qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[1]),
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


x = np.array([0.0, 0.0, 0.35, 0.0, 0.0, -0.35])


def H(x):
    return qchem.molecular_hamiltonian(["H", "H"], x)[0]


@pytest.mark.parametrize(
    ("i", "coeffs", "ops"),
    [(0, [], []), (2, non_zero_coeffs, non_zero_ops), (5, -non_zero_coeffs, non_zero_ops)],
)
def test_deriv(i, coeffs, ops):
    r"""Tests the correctness of the nuclear derivative of the electronic Hamiltonian
    computed by the ``'derivative'`` function. 
    """

    deriv = qchem.derivative(H, x, i)

    shutil.rmtree("pyscf")

    exp_obs = qml.Hamiltonian(coeffs, ops)

    assert exp_obs.compare(deriv)


def test_grad(tol):
    r"""Tests function `'gradient'` computing the gradient of the electronic Hamiltonian H(x)."""

    dev = qml.device("default.qubit", wires=4)

    def circuit(param, wires):
        qml.BasisState(np.array([1, 1, 0, 0]), wires=[0, 1, 2, 3])
        qml.RY(param, wires=2)
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[2, 0])
        qml.CNOT(wires=[3, 1])

    grad_h = qchem.gradient(H, x)

    shutil.rmtree("pyscf")

    assert len(grad_h) == x.size

    param = 6.07230111451844
    exp_grad_E = np.array([0.0, 0.0, -0.03540190344912991, 0.0, 0.0, 0.03540190344912991])

    calc_grad_E = np.array([qml.ExpvalCost(circuit, grad, dev)(param) for grad in grad_h])

    assert np.allclose(exp_grad_E, calc_grad_E, **tol)
