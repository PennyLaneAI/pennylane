import os
import sys

import numpy as np
import pytest

import pennylane as qml

# TODO: Bring pytest skip to relevant tests.
openfermion = pytest.importorskip("openfermion")
openfermionpyscf = pytest.importorskip("openfermionpyscf")

h2 = ["H", "H"]
x_h2 = np.array([0.0, 0.0, -0.661, 0.0, 0.0, 0.661])
coeffs_h2 = []
coeffs_h2.append([0.0])
coeffs_h2.append([0.0])
coeffs_h2.append([0.45445016, 0.45445016, 0.45445016, 0.45445016])


ops_h2 = []
ops_h2.append([qml.Identity(wires=[0])])
ops_h2.append([qml.Identity(wires=[0])])
ops_h2.append(
    [
        qml.PauliY(wires=[0]) @ qml.PauliZ(wires=[1]) @ qml.PauliY(wires=[2]),
        qml.PauliX(wires=[0]) @ qml.PauliZ(wires=[1]) @ qml.PauliX(wires=[2]),
        qml.PauliY(wires=[1]) @ qml.PauliZ(wires=[2]) @ qml.PauliY(wires=[3]),
        qml.PauliX(wires=[1]) @ qml.PauliZ(wires=[2]) @ qml.PauliX(wires=[3]),
    ]
)


h3p = ["H", "H", "H"]
x_h3p = np.array([0.028, 0.054, 0.0, 0.986, 1.610, 0.0, 1.855, 0.002, 0.0])
coeffs_h3p = []
coeffs_h3p.append(
    [
        0.47811232,
        0.47811232,
        -0.39136385,
        -0.39136385,
        -0.39136385,
        -0.39136385,
        0.26611147,
        0.26611147,
        0.26611147,
        0.26611147,
        0.71447791,
        0.71447791,
        -0.11734959,
        -0.11734959,
        -0.11734959,
        -0.11734959,
        0.24190978,
        0.24190978,
    ]
)
coeffs_h3p.append(
    [
        0.27769368,
        0.27769368,
        0.26614699,
        0.26614699,
        0.26614699,
        0.26614699,
        0.39131162,
        0.39131162,
        0.39131162,
        0.39131162,
        0.16019825,
        0.16019825,
        -0.23616713,
        -0.23616713,
        -0.23616713,
        -0.23616713,
        0.39510807,
        0.39510807,
    ]
)
coeffs_h3p.append([0.0])

ops_h3p = []
ops_h3p.append(
    [
        qml.PauliZ(wires=[0]),
        qml.PauliZ(wires=[1]),
        qml.PauliY(wires=[0]) @ qml.PauliZ(wires=[1]) @ qml.PauliY(wires=[2]),
        qml.PauliX(wires=[0]) @ qml.PauliZ(wires=[1]) @ qml.PauliX(wires=[2]),
        qml.PauliY(wires=[1]) @ qml.PauliZ(wires=[2]) @ qml.PauliY(wires=[3]),
        qml.PauliX(wires=[1]) @ qml.PauliZ(wires=[2]) @ qml.PauliX(wires=[3]),
        qml.PauliY(wires=[0])
        @ qml.PauliZ(wires=[1])
        @ qml.PauliZ(wires=[2])
        @ qml.PauliZ(wires=[3])
        @ qml.PauliY(wires=[4]),
        qml.PauliX(wires=[0])
        @ qml.PauliZ(wires=[1])
        @ qml.PauliZ(wires=[2])
        @ qml.PauliZ(wires=[3])
        @ qml.PauliX(wires=[4]),
        qml.PauliY(wires=[1])
        @ qml.PauliZ(wires=[2])
        @ qml.PauliZ(wires=[3])
        @ qml.PauliZ(wires=[4])
        @ qml.PauliY(wires=[5]),
        qml.PauliX(wires=[1])
        @ qml.PauliZ(wires=[2])
        @ qml.PauliZ(wires=[3])
        @ qml.PauliZ(wires=[4])
        @ qml.PauliX(wires=[5]),
        qml.PauliZ(wires=[2]),
        qml.PauliZ(wires=[3]),
        qml.PauliY(wires=[2]) @ qml.PauliZ(wires=[3]) @ qml.PauliY(wires=[4]),
        qml.PauliX(wires=[2]) @ qml.PauliZ(wires=[3]) @ qml.PauliX(wires=[4]),
        qml.PauliY(wires=[3]) @ qml.PauliZ(wires=[4]) @ qml.PauliY(wires=[5]),
        qml.PauliX(wires=[3]) @ qml.PauliZ(wires=[4]) @ qml.PauliX(wires=[5]),
        qml.PauliZ(wires=[4]),
        qml.PauliZ(wires=[5]),
    ]
)
ops_h3p.append(
    [
        qml.PauliZ(wires=[0]),
        qml.PauliZ(wires=[1]),
        qml.PauliY(wires=[0]) @ qml.PauliZ(wires=[1]) @ qml.PauliY(wires=[2]),
        qml.PauliX(wires=[0]) @ qml.PauliZ(wires=[1]) @ qml.PauliX(wires=[2]),
        qml.PauliY(wires=[1]) @ qml.PauliZ(wires=[2]) @ qml.PauliY(wires=[3]),
        qml.PauliX(wires=[1]) @ qml.PauliZ(wires=[2]) @ qml.PauliX(wires=[3]),
        qml.PauliY(wires=[0])
        @ qml.PauliZ(wires=[1])
        @ qml.PauliZ(wires=[2])
        @ qml.PauliZ(wires=[3])
        @ qml.PauliY(wires=[4]),
        qml.PauliX(wires=[0])
        @ qml.PauliZ(wires=[1])
        @ qml.PauliZ(wires=[2])
        @ qml.PauliZ(wires=[3])
        @ qml.PauliX(wires=[4]),
        qml.PauliY(wires=[1])
        @ qml.PauliZ(wires=[2])
        @ qml.PauliZ(wires=[3])
        @ qml.PauliZ(wires=[4])
        @ qml.PauliY(wires=[5]),
        qml.PauliX(wires=[1])
        @ qml.PauliZ(wires=[2])
        @ qml.PauliZ(wires=[3])
        @ qml.PauliZ(wires=[4])
        @ qml.PauliX(wires=[5]),
        qml.PauliZ(wires=[2]),
        qml.PauliZ(wires=[3]),
        qml.PauliY(wires=[2]) @ qml.PauliZ(wires=[3]) @ qml.PauliY(wires=[4]),
        qml.PauliX(wires=[2]) @ qml.PauliZ(wires=[3]) @ qml.PauliX(wires=[4]),
        qml.PauliY(wires=[3]) @ qml.PauliZ(wires=[4]) @ qml.PauliY(wires=[5]),
        qml.PauliX(wires=[3]) @ qml.PauliZ(wires=[4]) @ qml.PauliX(wires=[5]),
        qml.PauliZ(wires=[4]),
        qml.PauliZ(wires=[5]),
    ]
)
ops_h3p.append([qml.Identity(wires=[0])])


h2o = ["H", "H", "O"]
x_h2o = np.array([0.0, 1.431, -0.887, 0.0, -1.431, -0.887, 0.0, 0.0, 0.222])

coeffs_h2o = []
coeffs_h2o.append([-0.03700799, 0.03700799, 0.03700799, -0.03700799])
coeffs_h2o.append([0.0])
coeffs_h2o.append([0.28530454, 0.111, 0.111, -0.37101744, -0.37101744])

ops_h2o = []
ops_h2o.append(
    [
        qml.PauliX(wires=[0]) @ qml.PauliY(wires=[1]) @ qml.PauliY(wires=[2]),
        qml.PauliY(wires=[0]) @ qml.PauliY(wires=[1]) @ qml.PauliX(wires=[2]),
        qml.PauliZ(wires=[0]) @ qml.PauliX(wires=[1]) @ qml.PauliZ(wires=[3]),
        qml.PauliX(wires=[1]) @ qml.PauliZ(wires=[2]),
    ]
)
ops_h2o.append([qml.Identity(wires=[0])])
ops_h2o.append(
    [
        qml.Identity(wires=[0]),
        qml.PauliZ(wires=[0]),
        qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[1]),
        qml.PauliZ(wires=[2]),
        qml.PauliZ(wires=[1]) @ qml.PauliZ(wires=[2]) @ qml.PauliZ(wires=[3]),
    ]
)


@pytest.mark.parametrize(
    ("symbols", "coords", "charge", "core", "active", "mapping", "coeffs", "ops"),
    [
        (h2, x_h2, 0, None, None, "jordan_wigner", coeffs_h2, ops_h2),
        (h3p, x_h3p, 1, None, None, "jordan_wigner", coeffs_h3p, ops_h3p),
        (h2o, x_h2o, 0, range(4), [4, 5], "bravyi_kitaev", coeffs_h2o, ops_h2o),
    ],
)
def test_dipole_obs(symbols, coords, charge, core, active, mapping, coeffs, ops, tol, tmpdir):
    r"""Tests the correctness of the dipole observable computed by the ``dipole`` function."""

    dip = qml.qchem.dipole_of(
        symbols,
        coords,
        charge=charge,
        core=core,
        active=active,
        mapping=mapping,
        outpath=tmpdir.strpath,
    )

    assert len(dip) == len(ops)

    for i, _dip in enumerate(dip):
        calc_coeffs = np.array(_dip.coeffs)
        exp_coeffs = np.array(coeffs[i])
        assert np.allclose(calc_coeffs, exp_coeffs, **tol)
        assert all(isinstance(o1, o2.__class__) for o1, o2 in zip(_dip.ops, ops[i]))
        assert all(o1.wires == o2.wires for o1, o2 in zip(_dip.ops, ops[i]))


@pytest.mark.parametrize(
    ("symbols", "coords", "charge", "hf_state", "exp_dipole"),
    [
        (h2, x_h2, 0, np.array([1, 1, 0, 0]), np.array([0.0, 0.0, 0.0])),
        (h3p, x_h3p, 1, np.array([1, 1, 0, 0, 0, 0]), np.array([0.95655073, 0.55522528, 0.0])),
    ],
)
def test_dipole(symbols, coords, charge, hf_state, exp_dipole, tol, tmpdir):
    r"""Tests the correctness of the computed dipole moment."""

    n_qubits = len(hf_state)

    dev = qml.device("default.qubit", wires=n_qubits)

    dip_obs = qml.qchem.dipole_of(symbols, coords, charge=charge, outpath=tmpdir.strpath)

    def circuit(param, wires):
        qml.BasisState(hf_state, wires=wires)

    with pytest.warns(UserWarning, match="is deprecated,"):
        dipole = np.array([qml.ExpvalCost(circuit, obs, dev)(0.0) for obs in dip_obs])

    assert np.allclose(dipole, exp_dipole, **tol)


@pytest.mark.parametrize(
    ("symbols", "coords", "mult", "msg_match"),
    [
        (["H", "H"], x_h2, 2, "this functionality is constrained to Hartree-Fock states"),
        (["H", "Ca"], x_h2, 1, "only first- or second-row elements of the periodic table"),
    ],
)
def test_exceptions_dipole(symbols, coords, mult, msg_match):
    """Test exceptions of the ``dipole`` function."""

    with pytest.raises(ValueError, match=msg_match):
        qml.qchem.dipole_of(symbols, coords, mult=mult)
