import os

import numpy as np
import pytest

from pennylane import qchem

from openfermion import QubitOperator


terms_1_jw = {
    ((0, "Z"),): (-0.25 + 0j),
    ((1, "Z"),): (0.25 + 0j),
    ((2, "Z"),): (-0.25 + 0j),
    ((3, "Z"),): (0.25 + 0j),
}

terms_2_bk = {
    ((0, "Z"),): (-0.25 + 0j),
    ((0, "Z"), (1, "Z")): (0.25 + 0j),
    ((2, "Z"),): (-0.25 + 0j),
    ((1, "Z"), (2, "Z"), (3, "Z")): (0.25 + 0j),
    ((4, "Z"),): (-0.25 + 0j),
    ((4, "Z"), (5, "Z")): (0.25 + 0j),
}


@pytest.mark.parametrize(
    ("orbitals", "mapping", "terms_exp"),
    [
        (4, "JORDAN_wigner", terms_1_jw),
        (6, "bravyi_KITAEV", terms_2_bk),
    ],
)
def test_spin_z(orbitals, mapping, terms_exp, monkeypatch):
    r"""Tests the correctness of the :math:`\hat{S}_z` observable built by the
    function `'spin_z'`.

    The parametrized inputs are `.terms()` attribute of the `QubitOperator. The equality
    checking is implemented in the `qchem` module itself as it could be something
    useful to the users as well.
    """

    Sz = qchem.spin_z(orbitals, mapping=mapping)

    Sz_qubit_op = QubitOperator()
    monkeypatch.setattr(Sz_qubit_op, "terms", terms_exp)

    assert qchem._qubit_operators_equivalent(Sz_qubit_op, Sz)


@pytest.mark.parametrize(
    ("orbitals", "msg_match"),
    [
        (-3, "'orbitals' must be greater than 0"),
        (0, "'orbitals' must be greater than 0"),
    ],
)
def test_exception_spin_z(orbitals, msg_match):
    """Test that the function `'spin_z'` throws an exception if the
    number of orbitals is less than zero."""

    with pytest.raises(ValueError, match=msg_match):
        qchem.spin_z(orbitals)
