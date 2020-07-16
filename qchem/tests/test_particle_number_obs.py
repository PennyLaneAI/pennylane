import os

import numpy as np
import pytest

from pennylane import qchem

from openfermion.ops._qubit_operator import QubitOperator

terms_h20_jw_full = {
    (): (7 + 0j),
    ((0, "Z"),): (-0.5 + 0j),
    ((1, "Z"),): (-0.5 + 0j),
    ((2, "Z"),): (-0.5 + 0j),
    ((3, "Z"),): (-0.5 + 0j),
    ((4, "Z"),): (-0.5 + 0j),
    ((5, "Z"),): (-0.5 + 0j),
    ((6, "Z"),): (-0.5 + 0j),
    ((7, "Z"),): (-0.5 + 0j),
    ((8, "Z"),): (-0.5 + 0j),
    ((9, "Z"),): (-0.5 + 0j),
    ((10, "Z"),): (-0.5 + 0j),
    ((11, "Z"),): (-0.5 + 0j),
    ((12, "Z"),): (-0.5 + 0j),
    ((13, "Z"),): (-0.5 + 0j),
}

terms_h20_jw_23 = {
    (): (3 + 0j),
    ((0, "Z"),): (-0.5 + 0j),
    ((1, "Z"),): (-0.5 + 0j),
    ((2, "Z"),): (-0.5 + 0j),
    ((3, "Z"),): (-0.5 + 0j),
    ((4, "Z"),): (-0.5 + 0j),
    ((5, "Z"),): (-0.5 + 0j),
}

terms_h20_bk_44 = {
    (): (4 + 0j),
    ((0, "Z"),): (-0.5 + 0j),
    ((0, "Z"), (1, "Z")): (-0.5 + 0j),
    ((2, "Z"),): (-0.5 + 0j),
    ((1, "Z"), (2, "Z"), (3, "Z")): (-0.5 + 0j),
    ((4, "Z"),): (-0.5 + 0j),
    ((4, "Z"), (5, "Z")): (-0.5 + 0j),
    ((6, "Z"),): (-0.5 + 0j),
    ((3, "Z"), (5, "Z"), (6, "Z"), (7, "Z")): (-0.5 + 0j),
}

terms_lih_anion_bk = {
    (): (5 + 0j),
    ((0, "Z"),): (-0.5 + 0j),
    ((0, "Z"), (1, "Z")): (-0.5 + 0j),
    ((2, "Z"),): (-0.5 + 0j),
    ((1, "Z"), (2, "Z"), (3, "Z")): (-0.5 + 0j),
    ((4, "Z"),): (-0.5 + 0j),
    ((4, "Z"), (5, "Z")): (-0.5 + 0j),
    ((6, "Z"),): (-0.5 + 0j),
    ((3, "Z"), (5, "Z"), (6, "Z"), (7, "Z")): (-0.5 + 0j),
    ((8, "Z"),): (-0.5 + 0j),
    ((8, "Z"), (9, "Z")): (-0.5 + 0j),
}


@pytest.mark.parametrize(
    ("n_act_orb", "mapping", "terms_exp"),
    [
        (7, "JORDAN_wigner", terms_h20_jw_full),
        (3, "JORDAN_wigner", terms_h20_jw_23),
        (4, "bravyi_KITAEV", terms_h20_bk_44),
        (5, "bravyi_KITAEV", terms_lih_anion_bk),
    ],
)
def test_particle_number_observable(n_act_orb, mapping, terms_exp, monkeypatch):
    r"""Tests the correctness of the particle number observable :math:`\hat{N}` generated
    by the ``'particle_number'`` function.

    The parametrized inputs are `.terms` attribute of the particle number `QubitOperator`.
    The equality checking is implemented in the `qchem` module itself as it could be
    something useful to the users as well.
    """

    pn_obs = qchem.particle_number(n_act_orb, mapping=mapping)

    particle_number_qubit_op = QubitOperator()
    monkeypatch.setattr(particle_number_qubit_op, "terms", terms_exp)

    assert qchem._qubit_operators_equivalent(particle_number_qubit_op, pn_obs)
