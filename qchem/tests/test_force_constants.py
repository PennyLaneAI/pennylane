import numpy as np
import pytest

import pennylane as qml


h2 = ["H", "H"]
x_h2 = np.array([0.0, 0.0, -0.62846244, 0.0, 0.0, -2.01715414])

h2o = ["H", "O", "H"]
x_h2o = np.array(
    [-0.03987322, -0.00377945, 0.0, 1.57697645, 0.85396724, 0.0, 2.79093651, -0.51589523, 0.0]
)


@pytest.mark.parametrize(
    ("symbols", "x", "idx", "delta", "exp_deriv2"),
    [
        (h2, x_h2, [2, 2], 0.01, 0.5233469267239744),
        (h2, x_h2, [2, 5], 0.02, -0.5234751074837963),
    ],
)
def test_force_constants(symbols, x, idx, delta, exp_deriv2, tol, tmpdir):
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

    hessian = np.array(
        [
            [-4.85689668e-18, -3.03626501e-17, 1.66817003e-17],
            [1.22996247e-17, 7.17888633e-01, 1.46097459e-01],
            [-2.13220779e-17, 1.46097459e-01, -3.13920491e-02],
        ]
    )

    deriv2 = qml.qchem.force_constants(H, x, idx, circuit, params, dev, hessian, delta=delta)

    assert np.allclose(deriv2, exp_deriv2)
