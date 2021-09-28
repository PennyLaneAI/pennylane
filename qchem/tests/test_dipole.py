import os

import numpy as np
import pytest

import pennylane as qml


ref_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_ref_files")

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


coeffs_h2o_jw = []
coeffs_h2o_jw.append(
    [-0.26800371, -0.26800371, -0.26800371, -0.26800371, 0.0154021, 0.0154021, 0.0154021, 0.0154021]
)

coeffs_h2o_jw.append(
    [1.13391414, 0.45813039, 0.45813039, 0.02155415, 0.02155415, 0.02155415, 0.02155415]
)
coeffs_h2o_jw.append(
    [
        -0.5443051,
        -0.10090594,
        -0.10090594,
        0.25657967,
        0.25657967,
        0.25657967,
        0.25657967,
        -0.11083244,
        -0.11083244,
        -0.11892573,
        -0.11892573,
    ]
)


ops_h2o_jw = []
ops_h2o_jw.append(
    [
        qml.PauliY(wires=[0]) @ qml.PauliZ(wires=[1]) @ qml.PauliY(wires=[2]),
        qml.PauliX(wires=[0]) @ qml.PauliZ(wires=[1]) @ qml.PauliX(wires=[2]),
        qml.PauliY(wires=[1]) @ qml.PauliZ(wires=[2]) @ qml.PauliY(wires=[3]),
        qml.PauliX(wires=[1]) @ qml.PauliZ(wires=[2]) @ qml.PauliX(wires=[3]),
        qml.PauliY(wires=[2]) @ qml.PauliZ(wires=[3]) @ qml.PauliY(wires=[4]),
        qml.PauliX(wires=[2]) @ qml.PauliZ(wires=[3]) @ qml.PauliX(wires=[4]),
        qml.PauliY(wires=[3]) @ qml.PauliZ(wires=[4]) @ qml.PauliY(wires=[5]),
        qml.PauliX(wires=[3]) @ qml.PauliZ(wires=[4]) @ qml.PauliX(wires=[5]),
    ]
)
ops_h2o_jw.append(
    [
        qml.Identity(wires=[0]),
        qml.PauliZ(wires=[0]),
        qml.PauliZ(wires=[1]),
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
    ]
)
ops_h2o_jw.append(
    [
        qml.Identity(wires=[0]),
        qml.PauliZ(wires=[0]),
        qml.PauliZ(wires=[1]),
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
        qml.PauliZ(wires=[4]),
        qml.PauliZ(wires=[5]),
    ]
)


coeffs_h2o_bk = []
coeffs_h2o_bk.append([-0.26800371, 0.26800371, 0.26800371, -0.26800371])
coeffs_h2o_bk.append([1.13391414, 0.45813039, 0.45813039])
coeffs_h2o_bk.append([-0.78215657, -0.10090594, -0.10090594, -0.11083244, -0.11083244])

ops_h2o_bk = []
ops_h2o_bk.append(
    [
        qml.PauliX(wires=[0]) @ qml.PauliY(wires=[1]) @ qml.PauliY(wires=[2]),
        qml.PauliY(wires=[0]) @ qml.PauliY(wires=[1]) @ qml.PauliX(wires=[2]),
        qml.PauliZ(wires=[0]) @ qml.PauliX(wires=[1]) @ qml.PauliZ(wires=[3]),
        qml.PauliX(wires=[1]) @ qml.PauliZ(wires=[2]),
    ]
)
ops_h2o_bk.append(
    [qml.Identity(wires=[0]), qml.PauliZ(wires=[0]), qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[1])]
)
ops_h2o_bk.append(
    [
        qml.Identity(wires=[0]),
        qml.PauliZ(wires=[0]),
        qml.PauliZ(wires=[0]) @ qml.PauliZ(wires=[1]),
        qml.PauliZ(wires=[2]),
        qml.PauliZ(wires=[1]) @ qml.PauliZ(wires=[2]) @ qml.PauliZ(wires=[3]),
    ]
)


@pytest.mark.parametrize(
    ("name", "core", "active", "mapping", "coeffs", "ops"),
    [
        ("h2_pyscf.hdf5", None, None, "jordan_wigner", coeffs_h2, ops_h2),
        ("h2o_psi4.hdf5", range(4), [4, 5, 6], "jordan_wigner", coeffs_h2o_jw, ops_h2o_jw),
        ("h2o_psi4.hdf5", range(4), [4, 5], "bravyi_kitaev", coeffs_h2o_bk, ops_h2o_bk),
    ],
)
def test_dipole(name, core, active, mapping, coeffs, ops, tol):
    r"""Tests the correctness of the dipole observable computed by the ``dipole`` function."""

    hf_file = os.path.join(ref_dir, name)
    dip = qml.qchem.dipole(hf_file, core=core, active=active, mapping=mapping)

    assert len(dip) == len(ops)

    for i, _dip in enumerate(dip):
        calc_coeffs = np.array(_dip.coeffs)
        exp_coeffs = np.array(coeffs[i])
        assert np.allclose(calc_coeffs, exp_coeffs, **tol)
        assert all(isinstance(o1, o2.__class__) for o1, o2 in zip(_dip.ops, ops[i]))
        assert all(o1.wires == o2.wires for o1, o2 in zip(_dip.ops, ops[i]))


@pytest.mark.parametrize(
    ("name", "msg_match"),
    [
        ("lih_anion.hdf5", "this functionality is constrained to closed-shell Hartree-Fock states"),
        ("Mg.hdf5", "only first- or second-row elements of the periodic table"),
    ],
)
def test_exceptions_dipole(name, msg_match):
    """Test exceptions of the ``dipole`` function."""

    hf_file = os.path.join(ref_dir, name)

    with pytest.raises(ValueError, match=msg_match):
        qml.qchem.dipole(hf_file)
