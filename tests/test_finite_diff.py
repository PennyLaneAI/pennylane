import numpy as np
import pytest

import pennylane as qml


def f(x):
    return np.sin(x)


x1 = 0.5
delta1 = 0.001
df1 = 0.877582525324383

x2 = np.array([0.5, -0.3])
delta2 = 0.002
df2 = np.array([0.87758242, 0.0])


def g(x):
    return np.array([np.sin(x), 1 / x])


x3 = -0.25
delta3 = 0.003
dg3 = np.array([0.96891206, -16.00057602])

x4 = np.array([-0.25, 1.975])
delta4 = 0.004
dg4 = np.array([[0.0, -0.39328647], [0.0, -0.25636943]])


# parametrized Hamiltonian of the water molecule
def hamilt(x):
    return qml.qchem.molecular_hamiltonian(
        ["H", "O", "H"], x, active_electrons=2, active_orbitals=2, outpath=tmpdir.strpath
    )[0]


x5 = np.array(
    [-0.03987322, -0.00377945, 0.0, 1.57697645, 0.85396724, 0.0, 2.79093651, -0.51589523, 0.0]
)
delta5 = 0.01
coeffs5 = np.array(
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

ops5 = [
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
    ("func", "x", "i", "delta", "deriv_ref"),
    [
        (f, x1, None, delta1, df1),
        (f, x2, 0, delta2, df2),
        (g, x3, None, delta3, dg3),
        (g, x4, 1, delta4, dg4),
        # (hamilt, x5, 0, delta5, [coeffs5, ops5]),
        # (hamilt, x5, 2, delta5, [[], []]),
    ],
)
def test_finit_diff(func, x, i, delta, deriv_ref, fd_tol):
    r"""Tests the correctness of the derivative evaluated by the 'finite_diff' function."""

    deriv = qml.finite_diff(func, x, i, delta)

    if isinstance(deriv, qml.vqe.vqe.Hamiltonian):
        assert np.allclose(deriv.coeffs, deriv_ref[0], **fd_tol)
        assert all(isinstance(o1, o2.__class__) for o1, o2 in zip(deriv.ops, deriv_ref[1]))
        assert all(o1.wires == o2.wires for o1, o2 in zip(deriv.ops, deriv_ref[1]))
    else:
        assert np.allclose(deriv_ref, deriv, **fd_tol)


# def test_not_callable_func():
#     r"""Test that an error is raised if the function to be differentiated
#     is not a callable object"""

#     with pytest.raises(TypeError, match="'F' should be a callable function"):
#         qml.finite_diff(hamilt(x5), x5, 0)


@pytest.mark.parametrize(
    ("x", "i"), [(x2, None), (x2, 4)],
)
def test_exceptions(x, i):
    r"""Test that an error is raised if the index 'i' of the variable we are differentiating
    with respect to is not given or is out of bounds for multivariable functions"""

    with pytest.raises(ValueError, match="'i' must be an integer between"):
        qml.finite_diff(f, x2, i)
