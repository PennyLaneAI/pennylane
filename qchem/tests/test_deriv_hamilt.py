import numpy as np
import pytest

import pennylane as qml


coeffs1 = np.array(
    [
        0.2777044490017033,
        -0.0017650045675554615,
        -0.0017650045675499104,
        -0.1223647641789366,
        -0.1223647641789366,
        0.0016333664189948152,
        -0.0016333664189948152,
        -0.0016333664189948152,
        0.0016333664189948152,
        0.006294159636022201,
        0.007927526055015455,
        0.007927526055015455,
        0.006294159636022201,
        0.008903332246626428,
    ]
)

coeffs2 = np.array(
    [
        0.0908363115881059,
        0.0024997938516189144,
        0.0024997938516186924,
        -0.037411957658542605,
        -0.03741195765854255,
        0.0015388146024749305,
        -0.0015388146024749305,
        -0.0015388146024749305,
        0.0015388146024749305,
        -0.0054494665220388705,
        -0.003910651919563968,
        -0.003910651919563968,
        -0.0054494665220388705,
        -0.013008346319230257,
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
    ("i", "delta", "coeffs", "ops"),
    [
        (0, 0.01, coeffs1, ops),
        (1, 0.5, coeffs2, ops),
        (2, 0.5, [], []),
    ],
)
def test_finit_diff_hamilt(i, delta, coeffs, ops, tol, tmpdir):
    r"""Tests the correctness of the derivative of a molecular Hamiltonian calculated
    by the 'finite_diff' function."""

    # parametrized Hamiltonian of the water molecule
    def hamilt(x):
        return qml.qchem.molecular_hamiltonian(
            ["H", "O", "H"], x, active_electrons=2, active_orbitals=2, outpath=tmpdir.strpath
        )[0]

    x = np.array(
        [-0.03987322, -0.00377945, 0.0, 1.57697645, 0.85396724, 0.0, 2.79093651, -0.51589523, 0.0]
    )

    deriv = qml.finite_diff(hamilt, x, i, delta)

    assert np.allclose(deriv.coeffs, coeffs, **tol)
    assert all(isinstance(o1, o2.__class__) for o1, o2 in zip(deriv.ops, ops))
    assert all(o1.wires == o2.wires for o1, o2 in zip(deriv.ops, ops))
