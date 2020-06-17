import os

import numpy as np
import pytest

from pennylane import qchem

from openfermion.ops._qubit_operator import QubitOperator


@pytest.mark.parametrize(
    ("me_table", "init_term", "mapping", "terms_exp"),
    [
        (np.array([[0., 0., 0.5], [1., 1., -0.5], [1., 0., 0.0]]),
         1/2,
         "JORDAN_wigner",
         {(): (0.5+0j), ((0, 'Z'),): (-0.25+0j), ((1, 'Z'),): (0.25+0j)}),

        (np.array([[0., 0., 0.5], [1., 1., -0.5], [1., 0., 0.0]]),
         0,
         "JORDAN_wigner",
         {((0, 'Z'),): (-0.25+0j), ((1, 'Z'),): (0.25+0j)}),

        (np.array([[0., 0., 0.2], [1., 1., -0.5], [1., 0., 0.0]]),
         1/2,
         "bravyi_KITAEV",
         {(): (0.35+0j), ((0, 'Z'),): (-0.1+0j), ((0, 'Z'), (1, 'Z')): (0.25+0j)}),

        (np.array([[0., 0., 0., 0.,  0.25], [0., 1., 1., 0., -0.25], [ 0., 2., 2., 0., 0.25],
                   [0., 3., 3., 0., -0.25], [1., 0., 0., 1., -0.25], [ 2., 0., 0., 2., 0.25]]),
         1/2,
         "JORDAN_wigner",
         {(): (0.4375+0j),
          ((1, 'Z'),): (0.125+0j),
          ((0, 'Z'), (1, 'Z')): (-0.125+0j),
          ((2, 'Z'),): (-0.125+0j),
          ((0, 'Z'), (2, 'Z')): (0.125+0j),
          ((0, 'Z'),): (0.0625+0j),
          ((3, 'Z'),): (0.0625+0j),
          ((0, 'Z'), (3, 'Z')): (-0.0625+0j)}),

        (np.array([[0., 0., 0., 0.,  0.5], [0.,  1., 1., 0., -0.25], [ 0., 2., 2., 0.,  0.5],
                   [0., 3., 3., 0., -0.25], [1., 0., 0., 1., -0.25], [ 2., 0., 0., 2., -0.25]]),
         1/4,
         "bravyi_KITAEV",
         {(): (0.125+0j),
          ((0, 'Z'), (1, 'Z')): (0.125+0j),
          ((1, 'Z'),): (-0.125+0j), ((2, 'Z'),): (-0.0625+0j),
          ((0, 'Z'), (2, 'Z')): (0.0625+0j),
          ((1, 'Z'), (2, 'Z'), (3, 'Z')): (0.0625+0j),
          ((0, 'Z'), (1, 'Z'), (2, 'Z'), (3, 'Z')): (-0.0625+0j),
          ((0, 'Z'),): (0.125+0j)})
    ],
)
def test_observable(me_table, init_term, mapping, terms_exp, monkeypatch):
    r"""Tests the correctness of the 'observable' used to build many-body observables.

    The parametrized inputs `terms_exp` are `.terms` attribute of the corresponding
    `QubitOperator. The equality checking is implemented in the `qchem` module itself
    as it could be something useful to the users as well.
    """

    res_obs = qchem.obs.observable(me_table, init_term=init_term, mapping=mapping)

    qubit_op = QubitOperator()
    monkeypatch.setattr(qubit_op, "terms", terms_exp)

    assert qchem._qubit_operators_equivalent(qubit_op, res_obs)
