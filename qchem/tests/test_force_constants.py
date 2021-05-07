import numpy as np
import pytest

import pennylane as qml


h2 = ["H", "H"]
x_h2 = np.array([0.0, 0.0, -0.62846244, 0.0, 0.0, -2.01715414])
hessian_h2 = np.array(
    [
        [-4.85689668e-18, -3.03626501e-17, 1.66817003e-17],
        [1.22996247e-17, 7.17888633e-01, 1.46097459e-01],
        [-2.13220779e-17, 1.46097459e-01, -3.13920491e-02],
    ]
)

h2o = ["H", "O", "H"]
x_h2o = np.array(
    [-0.03987322, -0.00377945, 0.0, 1.57697645, 0.85396724, 0.0, 2.79093651, -0.51589523, 0.0]
)
hessian_h2o = np.array(
    [
        [5.99588172e-16, 1.38654534e-16, 4.78966786e-15],
        [7.83125526e-16, 5.45215003e-01, 3.08814261e-02],
        [6.13265568e-16, 3.08814261e-02, -6.63551063e-03],
    ]
)


@pytest.mark.parametrize(
    ("symbols", "x", "idx", "hessian", "delta", "exp_deriv2"),
    [
        (h2, x_h2, [2, 2], hessian_h2, 0.01, 0.5233469267239744),
        (h2o, x_h2o, [0, 3], hessian_h2o, 0.02, -0.5542671797678137),
    ],
)
def test_force_constants(symbols, x, idx, hessian, delta, exp_deriv2, tol, tmpdir):
    r"""Tests the correctness of the second derivative of the energy calculated
    by the 'force_constants' function."""

    def H(x):
        return qml.qchem.molecular_hamiltonian(
            symbols, x, active_electrons=2, active_orbitals=2, outpath=tmpdir.strpath
        )[0]

    dev = qml.device("default.qubit", wires=4)

    def circuit(params, wires):
        qml.BasisState(np.array([1, 1, 0, 0]), wires=[0, 1, 2, 3])
        qml.Rot(*params, wires=2)
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[2, 0])
        qml.CNOT(wires=[3, 1])

    params = np.array([-3.49563464, 0.34446748, -1.03118105])

    deriv2 = qml.qchem.force_constants(H, x, idx, hessian, circuit, params, dev, delta=delta)

    assert np.allclose(deriv2, exp_deriv2, **tol)


@pytest.mark.parametrize(
    ("idx", "hessian", "msg_match"),
    [
        (
            [0, 1, 2],
            np.ones((3, 3)),
            "The number of indices given in 'idx' can not be greater than two",
        ),
        (
            [0, 1],
            np.ones((3, 3, 3)),
            "The argument 'hessian' must be an array of shape",
        ),
        (
            [0, 1],
            np.ones((4, 4)),
            "The argument 'hessian' must be an array of shape",
        ),
    ],
)
def test_exceptions_force_constants(idx, hessian, msg_match):
    r"""Test that the 'force_constants' function raises an exception if the length of 'idx' or
    the shape of the 'hessian' are incorrect"""

    def H(x):
        obs = [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliX(0)]
        return qml.Hamiltonian(x, obs)

    x = np.array([1, 2])

    dev = qml.device("default.qubit", wires=4)

    def circuit(params, wires):
        qml.Rot(*params, wires=2)

    params = np.array([-3.49563464, 0.34446748, -1.03118105])

    with pytest.raises(ValueError, match=msg_match):
        qml.qchem.force_constants(H, x, idx, hessian, circuit, params, dev)
