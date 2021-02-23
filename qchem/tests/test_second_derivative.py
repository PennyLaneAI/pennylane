import numpy as np
import pytest

import pennylane as qml
from pennylane import qchem

import shutil


delta_diag = 0.00529
x_diag = np.array([-0.0211, -0.002, 0.0, 0.8345, 0.4519, 0.0, 1.4769, -0.273, 0.0])
coeffs_diag = np.array(
    [
        0.3477265623545358,
        -0.030523411304217995,
        -0.030523411304495738,
        -0.17712002790405676,
        -0.177120027903779,
        -0.03147619925860665,
        0.03147619925860665,
        0.03147619925860665,
        -0.03147619925860665,
        0.04561772636250726,
        0.014141527103779092,
        0.014141527103779092,
        0.04561772636250726,
        0.6647462687400493,
    ]
)


delta_off_diag = 0.08
x_off_diag = np.array([-0.028, -0.001, 0.0, 0.79, 0.43, 0.0, 1.5, -0.3, 0.0])
coeffs_off_diag = np.array(
    [
        -0.144934802574563,
        -0.01818706265356182,
        -0.018187062653564248,
        -0.17206776782849864,
        -0.17206776782850228,
        0.0009507863667557693,
        -0.0009507863667557693,
        -0.0009507863667557693,
        0.0009507863667557693,
        0.003285934907832377,
        0.004236721274589892,
        0.004236721274589892,
        0.003285934907832377,
        0.03200398467630317,
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
    ("i", "j", "x", "delta", "coeffs", "ops"),
    [
        (0, 0, x_diag, delta_diag, coeffs_diag, ops),
        (1, 3, x_off_diag, delta_off_diag, coeffs_off_diag, ops),
        (0, 2, x_off_diag, delta_off_diag, [], []),
    ],
)
def test_deriv2(i, j, x, delta, coeffs, ops, tol):
    r"""Tests the ``second_derivative`` function computing the second-order derivative
    of the electronic Hamiltonian ``H(x)`` at the nuclear coordinates ``x``.
    """

    def H(x):
        return qml.qchem.molecular_hamiltonian(
            ["H", "O", "H"], x, active_electrons=2, active_orbitals=2
        )[0]

    deriv2 = qchem.second_derivative(H, x, i, j, delta=delta)

    shutil.rmtree("pyscf")

    exp_obs = qml.vqe.Hamiltonian(coeffs, ops)

    assert np.allclose(deriv2.coeffs, exp_obs.coeffs, **tol)
    assert all(isinstance(o1, o2.__class__) for o1, o2 in zip(deriv2.ops, exp_obs.ops))
    assert all(o1.wires == o2.wires for o1, o2 in zip(deriv2.ops, exp_obs.ops))


def test_not_callable_h():
    r"""Test that an error is raised if the input Hamiltonian 'H' is not a callable
    object"""

    with pytest.raises(
        TypeError, match="should be a callable function to build the electronic Hamiltonian"
    ):
        qchem.second_derivative(ops, x_diag, 0, 0)
