import os

import numpy as np
import pytest

import pennylane as qml


h2 = ["H", "H"]
x_h2 = np.array([0.0, 0.0, -0.66140414, 0.0, 0.0, 0.66140414])
coeffs_h2 = []
coeffs_h2.append([0.0])
coeffs_h2.append([0.0])
coeffs_h2.append(
    [
        1.71471176,
        -0.42867794,
        -0.42867794,
        -0.62481348,
        -0.62481348,
        -0.62481348,
        -0.62481348,
        -0.42867794,
        -0.42867794,
    ]
)


ops_h2 = []
ops_h2.append([qml.Identity(wires=[0])])
ops_h2.append([qml.Identity(wires=[0])])
ops_h2.append(
    [
        qml.Identity(wires=[0]),
        qml.PauliZ(wires=[0]),
        qml.PauliZ(wires=[1]),
        qml.PauliY(wires=[0]) @ qml.PauliZ(wires=[1]) @ qml.PauliY(wires=[2]),
        qml.PauliX(wires=[0]) @ qml.PauliZ(wires=[1]) @ qml.PauliX(wires=[2]),
        qml.PauliY(wires=[1]) @ qml.PauliZ(wires=[2]) @ qml.PauliY(wires=[3]),
        qml.PauliX(wires=[1]) @ qml.PauliZ(wires=[2]) @ qml.PauliX(wires=[3]),
        qml.PauliZ(wires=[2]),
        qml.PauliZ(wires=[3]),
    ]
)


h3p = ["H", "H", "H"]
x_h3p = np.array([0.028, 0.054, 0.0, 0.986, 1.610, 0.0, 1.855, 0.002, 0.0])
coeffs_h3p = []
coeffs_h3p.append(
    [
        3.01569149,
        -0.88863892,
        -0.88863892,
        -0.88783782,
        -0.88783782,
        -0.88783782,
        -0.88783782,
        0.22780641,
        0.22780641,
        0.22780641,
        0.22780641,
        -1.30305728,
        -1.30305728,
        -0.80078948,
        -0.80078948,
        -0.80078948,
        -0.80078948,
        -0.75064955,
        -0.75064955,
    ]
)
coeffs_h3p.append(
    [
        1.6907817,
        0.08159501,
        0.08159501,
        0.15052334,
        0.15052334,
        0.15052334,
        0.15052334,
        -0.18741661,
        -0.18741661,
        -0.18741661,
        -0.18741661,
        -0.47139071,
        -0.47139071,
        -0.78233294,
        -0.78233294,
        -0.78233294,
        -0.78233294,
        -1.28859515,
        -1.28859515,
    ]
)
coeffs_h3p.append([0.0])

ops_h3p = []
ops_h3p.append(
    [
        qml.Identity(wires=[0]),
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
        qml.Identity(wires=[0]),
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
coeffs_h2o.append([0.01540292, -0.01540292, -0.01540292, 0.01540292])
coeffs_h2o.append([3.21808936])
coeffs_h2o.append([-0.57923888, -0.111, -0.111, -0.18010003, -0.18010003])

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
def test_dipole(symbols, coords, charge, core, active, mapping, coeffs, ops, tol, tmpdir):
    r"""Tests the correctness of the dipole observable computed by the ``dipole`` function."""

    dip = qml.qchem.dipole(
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
    ("symbols", "coords", "mult", "msg_match"),
    [
        (["H", "H"], x_h2, 2, "this functionality is constrained to Hartree-Fock states"),
        (["H", "Ca"], x_h2, 1, "only first- or second-row elements of the periodic table"),
    ],
)
def test_exceptions_dipole(symbols, coords, mult, msg_match):
    """Test exceptions of the ``dipole`` function."""

    with pytest.raises(ValueError, match=msg_match):
        qml.qchem.dipole(symbols, coords, mult=mult)
