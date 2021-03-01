import numpy as np
import pytest

import pennylane as qml
from pennylane import qchem


x = np.array(
    [-0.03987322, -0.00377945, 0.0, 1.57697645, 0.85396724, 0.0, 2.79093651, -0.51589523, 0.0]
)

coeffs0 = np.array(
    [
        0.2777044490144931,
        -0.001765004568932138,
        -0.001765004568932138,
        -0.1223647641780512,
        -0.12236476417804565,
        0.001633366418892987,
        -0.001633366418892987,
        -0.001633366418892987,
        0.001633366418892987,
        0.006294159637185159,
        0.007927526056078493,
        0.007927526056078493,
        0.006294159637185159,
        0.00890333224733697,
    ]
)

coeffs1 = np.array(
    [
        0.09224972635024642,
        0.0024980374700173114,
        0.002498037470020642,
        -0.041507719134558085,
        -0.041507719134558085,
        0.001097571018512239,
        -0.001097571018512239,
        -0.001097571018512239,
        0.001097571018512239,
        -0.005103621783507095,
        -0.0040060507649947175,
        -0.0040060507649947175,
        -0.005103621783507095,
        -0.004156005798662266,
    ]
)

nonzero_ops = [
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
    [(0, 0.01, coeffs0, nonzero_ops), (1, 0.05, coeffs1, nonzero_ops), (2, 0.01, [], [])],
)
def test_deriv(i, delta, coeffs, ops, tol, tmpdir):
    r"""Tests the correctness of the nuclear derivative of the electronic Hamiltonian
    computed by the ``'derivative'`` function. 
    """

    def H(x):
        return qchem.molecular_hamiltonian(
            ["H", "O", "H"], x, active_electrons=2, active_orbitals=2, outpath=tmpdir.strpath
        )[0]

    deriv = qchem.derivative(H, x, i, delta=delta)

    exp_obs = qml.vqe.Hamiltonian(coeffs, ops)

    assert np.allclose(deriv.coeffs, exp_obs.coeffs, **tol)
    assert all(isinstance(o1, o2.__class__) for o1, o2 in zip(deriv.ops, exp_obs.ops))
    assert all(o1.wires == o2.wires for o1, o2 in zip(deriv.ops, exp_obs.ops))


def test_grad(tol, tmpdir):
    r"""Tests function `'gradient'` computing the gradient of the electronic Hamiltonian H(x)."""

    dev = qml.device("default.qubit", wires=4)

    def circuit(param, wires):
        qml.BasisState(np.array([1, 1, 0, 0]), wires=[0, 1, 2, 3])
        qml.RY(param, wires=2)
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[2, 0])
        qml.CNOT(wires=[3, 1])

    def H(x):
        return qchem.molecular_hamiltonian(["H", "H"], x, outpath=tmpdir.strpath)[0]

    x = np.array([0.0, 0.0, -0.66140414, 0.0, 0.0, 0.66140414])

    grad_h = qchem.gradient(H, x)

    assert len(grad_h) == len(x)

    param = 6.07230111451844
    exp_grad_E = np.array([0.0, 0.0, 0.035401908032693274, 0.0, 0.0, -0.035401908032788365])

    calc_grad_E = np.array([qml.ExpvalCost(circuit, grad, dev)(param) for grad in grad_h])

    assert np.allclose(exp_grad_E, calc_grad_E, **tol)


def test_not_callable_h():
    r"""Test that an error is raised if the input Hamiltonian 'H' is not a callable
    object"""

    with pytest.raises(TypeError, match="'F' should be a callable function"):
        qchem.derivative(nonzero_ops, x, 0)
