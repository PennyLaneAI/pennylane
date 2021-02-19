import numpy as np
import pytest

import pennylane as qml
from pennylane import qchem

import shutil


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

delta_diag = 0.005291772
coeffs_diag = np.array(
    [
        0.34772577929044984,
        -0.030524310020001295,
        -0.030524310019168627,
        -0.17711904124472824,
        -0.17711904124389558,
        -0.031476241863026465,
        0.031476241863026465,
        0.031476241863026465,
        -0.031476241863026465,
        0.0456184398783165,
        0.014142198015151252,
        0.014142198015151252,
        0.0456184398783165,
        0.6647460642390949,
    ]
)

delta_off_diag = 0.008
coeffs_off_diag = np.array(
    [
        -0.04445503691300096,
        0.08566509919512479,
        0.08566509919536767,
        0.28996324336433565,
        0.2899632433644571,
        0.05413053432861373,
        -0.05413053432861373,
        -0.05413053432861373,
        0.05413053432861373,
        -0.08536187184617826,
        -0.03123133751738237,
        -0.03123133751738237,
        -0.08536187184617826,
        -1.1678041893026774,
    ]
)

x = np.array([-0.0211, -0.002, 0.0, 0.8345, 0.4519, 0.0, 1.4769, -0.273, 0.0])


def H(x):
    return qml.qchem.molecular_hamiltonian(
        ["H", "O", "H"], x, active_electrons=2, active_orbitals=2
    )[0]


@pytest.mark.parametrize(
    ("i", "j", "delta", "coeffs", "ops"),
    [
        (0, 0, delta_diag, coeffs_diag, ops),
        (0, 3, delta_off_diag, coeffs_off_diag, ops),
        (0, 2, delta_diag, [], []),
    ],
)
def test_deriv2(i, j, delta, coeffs, ops, tol):
    r"""Tests the correctness of the second-order nuclear derivative of the electronic Hamiltonian
    computed by the ``'second_derivative'`` function. 
    """

    deriv2 = qchem.second_derivative(H, x, i, j, delta=delta)

    shutil.rmtree("pyscf")

    exp_obs = qml.vqe.Hamiltonian(coeffs, ops)

    assert np.allclose(deriv2.coeffs, exp_obs.coeffs, **tol)
    assert all(isinstance(o1, o2.__class__) for o1, o2 in zip(deriv2.ops, exp_obs.ops))
    assert all(o1.wires == o2.wires for o1, o2 in zip(deriv2.ops, exp_obs.ops))


def test_integration(tol):
    r"""Tests integration with PennyLane."""

    dev = qml.device("default.qubit", wires=4)

    def circuit(param, wires):
        qml.BasisState(np.array([1, 1, 0, 0]), wires=[0, 1, 2, 3])
        qml.RY(param, wires=2)
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[2, 0])
        qml.CNOT(wires=[3, 1])

    deriv2 = qchem.second_derivative(H, x, i=0, j=0)

    shutil.rmtree("pyscf")

    param = 6.0723
    exp_res = 0.632611572251488

    assert np.allclose(qml.ExpvalCost(circuit, deriv2, dev)(param), exp_res, **tol)
