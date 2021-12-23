import os

import numpy as np
import pennylane as qml
import pytest
from openfermion import QubitOperator

from pennylane import qchem


@pytest.mark.parametrize(
    ("mol_name", "terms_ref"),
    [
        ("empty", None),
        (
            "lih [jordan_WIGNER]",
            {
                (): (-7.50915719389077 + 0j),
                ((0, "Z"),): (0.155924093421341 + 0j),
                ((0, "Y"), (1, "Z"), (2, "Y")): (0.01401593800246412 + 0j),
                ((0, "X"), (1, "Z"), (2, "X")): (0.01401593800246412 + 0j),
                ((1, "Z"),): (0.1559240934213409 + 0j),
                ((1, "Y"), (2, "Z"), (3, "Y")): (0.014015938002464118 + 0j),
                ((1, "X"), (2, "Z"), (3, "X")): (0.014015938002464118 + 0j),
                ((2, "Z"),): (-0.01503982573626933 + 0j),
                ((3, "Z"),): (-0.015039825736269333 + 0j),
                ((0, "Z"), (1, "Z")): (0.12182774218528421 + 0j),
                ((0, "Y"), (2, "Y")): (0.012144893851836855 + 0j),
                ((0, "X"), (2, "X")): (0.012144893851836855 + 0j),
                ((0, "Z"), (1, "Y"), (2, "Z"), (3, "Y")): (0.012144893851836855 + 0j),
                ((0, "Z"), (1, "X"), (2, "Z"), (3, "X")): (0.012144893851836855 + 0j),
                ((0, "Y"), (1, "X"), (2, "X"), (3, "Y")): (0.00326599398593671 + 0j),
                ((0, "Y"), (1, "Y"), (2, "X"), (3, "X")): (-0.00326599398593671 + 0j),
                ((0, "X"), (1, "X"), (2, "Y"), (3, "Y")): (-0.00326599398593671 + 0j),
                ((0, "X"), (1, "Y"), (2, "Y"), (3, "X")): (0.00326599398593671 + 0j),
                ((0, "Z"), (2, "Z")): (0.052636515240899254 + 0j),
                ((0, "Z"), (3, "Z")): (0.05590250922683597 + 0j),
                ((0, "Y"), (1, "Z"), (2, "Y"), (3, "Z")): (-0.0018710418360866883 + 0j),
                ((0, "X"), (1, "Z"), (2, "X"), (3, "Z")): (-0.0018710418360866883 + 0j),
                ((1, "Z"), (2, "Z")): (0.05590250922683597 + 0j),
                ((1, "Y"), (3, "Y")): (-0.0018710418360866883 + 0j),
                ((1, "X"), (3, "X")): (-0.0018710418360866883 + 0j),
                ((1, "Z"), (3, "Z")): (0.052636515240899254 + 0j),
                ((2, "Z"), (3, "Z")): (0.08447056917218312 + 0j),
            },
        ),
        (
            "lih [BRAVYI_kitaev]",
            {
                (): (-7.50915719389077 + 0j),
                ((0, "Z"),): (0.155924093421341 + 0j),
                ((0, "X"), (1, "Y"), (2, "Y")): (0.01401593800246412 + 0j),
                ((0, "Y"), (1, "Y"), (2, "X")): (-0.01401593800246412 + 0j),
                ((0, "Z"), (1, "Z")): (0.1559240934213409 + 0j),
                ((0, "Z"), (1, "X"), (3, "Z")): (-0.014015938002464118 + 0j),
                ((1, "X"), (2, "Z")): (0.014015938002464118 + 0j),
                ((2, "Z"),): (-0.01503982573626933 + 0j),
                ((1, "Z"), (2, "Z"), (3, "Z")): (-0.015039825736269333 + 0j),
                ((1, "Z"),): (0.12182774218528421 + 0j),
                ((0, "Y"), (1, "X"), (2, "Y")): (0.012144893851836855 + 0j),
                ((0, "X"), (1, "X"), (2, "X")): (0.012144893851836855 + 0j),
                ((1, "X"), (3, "Z")): (-0.012144893851836855 + 0j),
                ((0, "Z"), (1, "X"), (2, "Z")): (0.012144893851836855 + 0j),
                ((0, "Y"), (1, "Z"), (2, "Y"), (3, "Z")): (0.00326599398593671 + 0j),
                ((0, "X"), (1, "Z"), (2, "X")): (0.00326599398593671 + 0j),
                ((0, "X"), (1, "Z"), (2, "X"), (3, "Z")): (0.00326599398593671 + 0j),
                ((0, "Y"), (1, "Z"), (2, "Y")): (0.00326599398593671 + 0j),
                ((0, "Z"), (2, "Z")): (0.052636515240899254 + 0j),
                ((0, "Z"), (1, "Z"), (2, "Z"), (3, "Z")): (0.05590250922683597 + 0j),
                ((0, "X"), (1, "X"), (2, "X"), (3, "Z")): (0.0018710418360866883 + 0j),
                ((0, "Y"), (1, "X"), (2, "Y"), (3, "Z")): (0.0018710418360866883 + 0j),
                ((0, "Z"), (1, "Z"), (2, "Z")): (0.05590250922683597 + 0j),
                ((0, "Z"), (1, "X"), (2, "Z"), (3, "Z")): (0.0018710418360866883 + 0j),
                ((1, "X"),): (-0.0018710418360866883 + 0j),
                ((0, "Z"), (2, "Z"), (3, "Z")): (0.052636515240899254 + 0j),
                ((1, "Z"), (3, "Z")): (0.08447056917218312 + 0j),
            },
        ),
        (
            "h2_psycf [jordan_WIGNER]",
            {
                (): (-0.04207897647782188 + 0j),
                ((0, "Z"),): (0.17771287465139934 + 0j),
                ((1, "Z"),): (0.1777128746513993 + 0j),
                ((2, "Z"),): (-0.24274280513140484 + 0j),
                ((3, "Z"),): (-0.24274280513140484 + 0j),
                ((0, "Z"), (1, "Z")): (0.17059738328801055 + 0j),
                ((0, "Y"), (1, "X"), (2, "X"), (3, "Y")): (0.04475014401535161 + 0j),
                ((0, "Y"), (1, "Y"), (2, "X"), (3, "X")): (-0.04475014401535161 + 0j),
                ((0, "X"), (1, "X"), (2, "Y"), (3, "Y")): (-0.04475014401535161 + 0j),
                ((0, "X"), (1, "Y"), (2, "Y"), (3, "X")): (0.04475014401535161 + 0j),
                ((0, "Z"), (2, "Z")): (0.12293305056183801 + 0j),
                ((0, "Z"), (3, "Z")): (0.1676831945771896 + 0j),
                ((1, "Z"), (2, "Z")): (0.1676831945771896 + 0j),
                ((1, "Z"), (3, "Z")): (0.12293305056183801 + 0j),
                ((2, "Z"), (3, "Z")): (0.176276408043196 + 0j),
            },
        ),
        (
            "h2_psycf [BRAVYI_kitaev]",
            {
                (): (-0.04207897647782188 + 0j),
                ((0, "Z"),): (0.17771287465139934 + 0j),
                ((0, "Z"), (1, "Z")): (0.1777128746513993 + 0j),
                ((2, "Z"),): (-0.24274280513140484 + 0j),
                ((1, "Z"), (2, "Z"), (3, "Z")): (-0.24274280513140484 + 0j),
                ((1, "Z"),): (0.17059738328801055 + 0j),
                ((0, "Y"), (1, "Z"), (2, "Y"), (3, "Z")): (0.04475014401535161 + 0j),
                ((0, "X"), (1, "Z"), (2, "X")): (0.04475014401535161 + 0j),
                ((0, "X"), (1, "Z"), (2, "X"), (3, "Z")): (0.04475014401535161 + 0j),
                ((0, "Y"), (1, "Z"), (2, "Y")): (0.04475014401535161 + 0j),
                ((0, "Z"), (2, "Z")): (0.12293305056183801 + 0j),
                ((0, "Z"), (1, "Z"), (2, "Z"), (3, "Z")): (0.1676831945771896 + 0j),
                ((0, "Z"), (1, "Z"), (2, "Z")): (0.1676831945771896 + 0j),
                ((0, "Z"), (2, "Z"), (3, "Z")): (0.12293305056183801 + 0j),
                ((1, "Z"), (3, "Z")): (0.176276408043196 + 0j),
            },
        ),
        (
            "h2o_psi4 [jordan_WIGNER]",
            {
                (): (-73.3320453921657 + 0j),
                ((0, "Z"),): (0.5152794751801038 + 0j),
                ((0, "Y"), (1, "Z"), (2, "Z"), (3, "Z"), (4, "Y")): (0.07778754984633934 + 0j),
                ((0, "X"), (1, "Z"), (2, "Z"), (3, "Z"), (4, "X")): (0.07778754984633934 + 0j),
                ((1, "Z"),): (0.515279475180104 + 0j),
                ((1, "Y"), (2, "Z"), (3, "Z"), (4, "Z"), (5, "Y")): (0.07778754984633934 + 0j),
                ((1, "X"), (2, "Z"), (3, "Z"), (4, "Z"), (5, "X")): (0.07778754984633934 + 0j),
                ((2, "Z"),): (0.4812925883672432 + 0j),
                ((3, "Z"),): (0.48129258836724326 + 0j),
                ((4, "Z"),): (0.09030949181042286 + 0j),
                ((5, "Z"),): (0.09030949181042283 + 0j),
                ((0, "Z"), (1, "Z")): (0.1956590715408106 + 0j),
                ((0, "Y"), (2, "Z"), (3, "Z"), (4, "Y")): (0.030346614024840804 + 0j),
                ((0, "X"), (2, "Z"), (3, "Z"), (4, "X")): (0.030346614024840804 + 0j),
                ((0, "Y"), (1, "X"), (2, "X"), (3, "Y")): (0.013977596555816168 + 0j),
                ((0, "Y"), (1, "Y"), (2, "X"), (3, "X")): (-0.013977596555816168 + 0j),
                ((0, "X"), (1, "X"), (2, "Y"), (3, "Y")): (-0.013977596555816168 + 0j),
                ((0, "X"), (1, "Y"), (2, "Y"), (3, "X")): (0.013977596555816168 + 0j),
                ((0, "Z"), (1, "Y"), (2, "Z"), (3, "Z"), (4, "Z"), (5, "Y")): (
                    0.030346614024840804 + 0j
                ),
                ((0, "Z"), (1, "X"), (2, "Z"), (3, "Z"), (4, "Z"), (5, "X")): (
                    0.030346614024840804 + 0j
                ),
                ((0, "Y"), (1, "X"), (4, "X"), (5, "Y")): (0.01718525123891425 + 0j),
                ((0, "Y"), (1, "Y"), (4, "X"), (5, "X")): (-0.01718525123891425 + 0j),
                ((0, "X"), (1, "X"), (4, "Y"), (5, "Y")): (-0.01718525123891425 + 0j),
                ((0, "X"), (1, "Y"), (4, "Y"), (5, "X")): (0.01718525123891425 + 0j),
                ((0, "Z"), (2, "Z")): (0.16824174504299702 + 0j),
                ((0, "Y"), (1, "Z"), (3, "Z"), (4, "Y")): (0.029512711807110188 + 0j),
                ((0, "X"), (1, "Z"), (3, "Z"), (4, "X")): (0.029512711807110188 + 0j),
                ((0, "Z"), (3, "Z")): (0.18221934159881317 + 0j),
                ((0, "Y"), (1, "Z"), (2, "Z"), (4, "Y")): (0.029077593893863385 + 0j),
                ((0, "X"), (1, "Z"), (2, "Z"), (4, "X")): (0.029077593893863385 + 0j),
                ((0, "Y"), (1, "Z"), (2, "Y"), (3, "Y"), (4, "Z"), (5, "Y")): (
                    0.00043511791324680473 + 0j
                ),
                ((0, "Y"), (1, "Z"), (2, "Y"), (3, "X"), (4, "Z"), (5, "X")): (
                    0.00043511791324680473 + 0j
                ),
                ((0, "X"), (1, "Z"), (2, "X"), (3, "Y"), (4, "Z"), (5, "Y")): (
                    0.00043511791324680473 + 0j
                ),
                ((0, "X"), (1, "Z"), (2, "X"), (3, "X"), (4, "Z"), (5, "X")): (
                    0.00043511791324680473 + 0j
                ),
                ((0, "Z"), (4, "Z")): (0.12008313883007578 + 0j),
                ((0, "Z"), (5, "Z")): (0.13726839006899005 + 0j),
                ((0, "Y"), (1, "Z"), (2, "Z"), (3, "Z"), (4, "Y"), (5, "Z")): (
                    0.011149373109704066 + 0j
                ),
                ((0, "X"), (1, "Z"), (2, "Z"), (3, "Z"), (4, "X"), (5, "Z")): (
                    0.011149373109704066 + 0j
                ),
                ((1, "Z"), (2, "Z")): (0.18221934159881317 + 0j),
                ((1, "Y"), (3, "Z"), (4, "Z"), (5, "Y")): (0.029077593893863385 + 0j),
                ((1, "X"), (3, "Z"), (4, "Z"), (5, "X")): (0.029077593893863385 + 0j),
                ((1, "Y"), (2, "X"), (3, "X"), (4, "Y")): (0.00043511791324680484 + 0j),
                ((1, "Y"), (2, "Y"), (3, "X"), (4, "X")): (-0.00043511791324680484 + 0j),
                ((1, "X"), (2, "X"), (3, "Y"), (4, "Y")): (-0.00043511791324680484 + 0j),
                ((1, "X"), (2, "Y"), (3, "Y"), (4, "X")): (0.00043511791324680484 + 0j),
                ((1, "Z"), (3, "Z")): (0.16824174504299702 + 0j),
                ((1, "Y"), (2, "Z"), (4, "Z"), (5, "Y")): (0.029512711807110188 + 0j),
                ((1, "X"), (2, "Z"), (4, "Z"), (5, "X")): (0.029512711807110188 + 0j),
                ((1, "Z"), (4, "Z")): (0.13726839006899005 + 0j),
                ((1, "Y"), (2, "Z"), (3, "Z"), (5, "Y")): (0.011149373109704066 + 0j),
                ((1, "X"), (2, "Z"), (3, "Z"), (5, "X")): (0.011149373109704066 + 0j),
                ((1, "Z"), (5, "Z")): (0.12008313883007578 + 0j),
                ((2, "Z"), (3, "Z")): (0.22003977334376118 + 0j),
                ((2, "Y"), (3, "X"), (4, "X"), (5, "Y")): (0.009647475282106617 + 0j),
                ((2, "Y"), (3, "Y"), (4, "X"), (5, "X")): (-0.009647475282106617 + 0j),
                ((2, "X"), (3, "X"), (4, "Y"), (5, "Y")): (-0.009647475282106617 + 0j),
                ((2, "X"), (3, "Y"), (4, "Y"), (5, "X")): (0.009647475282106617 + 0j),
                ((2, "Z"), (4, "Z")): (0.13758959215600186 + 0j),
                ((2, "Z"), (5, "Z")): (0.1472370674381085 + 0j),
                ((3, "Z"), (4, "Z")): (0.1472370674381085 + 0j),
                ((3, "Z"), (5, "Z")): (0.13758959215600186 + 0j),
                ((4, "Z"), (5, "Z")): (0.1492827559305538 + 0j),
            },
        ),
        (
            "h2o_psi4 [BRAVYI_kitaev]",
            {
                (): (-73.3320453921657 + 0j),
                ((0, "Z"),): (0.5152794751801038 + 0j),
                ((0, "X"), (1, "X"), (3, "Y"), (4, "Y"), (5, "X")): (0.07778754984633934 + 0j),
                ((0, "Y"), (1, "X"), (3, "Y"), (4, "X"), (5, "X")): (-0.07778754984633934 + 0j),
                ((0, "Z"), (1, "Z")): (0.515279475180104 + 0j),
                ((0, "Z"), (1, "X"), (3, "Y"), (5, "Y")): (0.07778754984633934 + 0j),
                ((1, "Y"), (3, "Y"), (4, "Z"), (5, "X")): (-0.07778754984633934 + 0j),
                ((2, "Z"),): (0.4812925883672432 + 0j),
                ((1, "Z"), (2, "Z"), (3, "Z")): (0.48129258836724326 + 0j),
                ((4, "Z"),): (0.09030949181042286 + 0j),
                ((4, "Z"), (5, "Z")): (0.09030949181042283 + 0j),
                ((1, "Z"),): (0.1956590715408106 + 0j),
                ((0, "Y"), (1, "Y"), (3, "Y"), (4, "Y"), (5, "X")): (-0.030346614024840804 + 0j),
                ((0, "X"), (1, "Y"), (3, "Y"), (4, "X"), (5, "X")): (-0.030346614024840804 + 0j),
                ((0, "Y"), (1, "Z"), (2, "Y"), (3, "Z")): (0.013977596555816168 + 0j),
                ((0, "X"), (1, "Z"), (2, "X")): (0.013977596555816168 + 0j),
                ((0, "X"), (1, "Z"), (2, "X"), (3, "Z")): (0.013977596555816168 + 0j),
                ((0, "Y"), (1, "Z"), (2, "Y")): (0.013977596555816168 + 0j),
                ((1, "X"), (3, "Y"), (5, "Y")): (0.030346614024840804 + 0j),
                ((0, "Z"), (1, "Y"), (3, "Y"), (4, "Z"), (5, "X")): (-0.030346614024840804 + 0j),
                ((0, "Y"), (4, "Y"), (5, "Z")): (0.01718525123891425 + 0j),
                ((0, "X"), (1, "Z"), (4, "X")): (0.01718525123891425 + 0j),
                ((0, "X"), (4, "X"), (5, "Z")): (0.01718525123891425 + 0j),
                ((0, "Y"), (1, "Z"), (4, "Y")): (0.01718525123891425 + 0j),
                ((0, "Z"), (2, "Z")): (0.16824174504299702 + 0j),
                ((0, "X"), (1, "X"), (2, "Z"), (3, "Y"), (4, "Y"), (5, "X")): (
                    0.029512711807110188 + 0j
                ),
                ((0, "Y"), (1, "X"), (2, "Z"), (3, "Y"), (4, "X"), (5, "X")): (
                    -0.029512711807110188 + 0j
                ),
                ((0, "Z"), (1, "Z"), (2, "Z"), (3, "Z")): (0.18221934159881317 + 0j),
                ((0, "X"), (1, "Y"), (2, "Z"), (3, "X"), (4, "Y"), (5, "X")): (
                    0.029077593893863385 + 0j
                ),
                ((0, "Y"), (1, "Y"), (2, "Z"), (3, "X"), (4, "X"), (5, "X")): (
                    -0.029077593893863385 + 0j
                ),
                ((0, "X"), (1, "X"), (2, "X"), (3, "Y"), (5, "Y")): (-0.00043511791324680473 + 0j),
                ((0, "X"), (1, "Y"), (2, "Y"), (3, "X"), (4, "Z"), (5, "X")): (
                    0.00043511791324680473 + 0j
                ),
                ((0, "Y"), (1, "X"), (2, "Y"), (3, "Y"), (5, "Y")): (-0.00043511791324680473 + 0j),
                ((0, "Y"), (1, "Y"), (2, "X"), (3, "X"), (4, "Z"), (5, "X")): (
                    -0.00043511791324680473 + 0j
                ),
                ((0, "Z"), (4, "Z")): (0.12008313883007578 + 0j),
                ((0, "Z"), (4, "Z"), (5, "Z")): (0.13726839006899005 + 0j),
                ((0, "X"), (1, "X"), (3, "Y"), (4, "X"), (5, "Y")): (0.011149373109704066 + 0j),
                ((0, "Y"), (1, "X"), (3, "Y"), (4, "Y"), (5, "Y")): (0.011149373109704066 + 0j),
                ((0, "Z"), (1, "Z"), (2, "Z")): (0.18221934159881317 + 0j),
                ((0, "Z"), (1, "X"), (2, "Z"), (3, "Y"), (5, "Y")): (0.029077593893863385 + 0j),
                ((1, "Y"), (2, "Z"), (3, "Y"), (4, "Z"), (5, "X")): (-0.029077593893863385 + 0j),
                ((0, "Z"), (1, "Y"), (2, "X"), (3, "X"), (4, "Y"), (5, "X")): (
                    0.00043511791324680484 + 0j
                ),
                ((0, "Z"), (1, "Y"), (2, "Y"), (3, "X"), (4, "X"), (5, "X")): (
                    -0.00043511791324680484 + 0j
                ),
                ((1, "Y"), (2, "Y"), (3, "Y"), (4, "Y"), (5, "X")): (0.00043511791324680484 + 0j),
                ((1, "Y"), (2, "X"), (3, "Y"), (4, "X"), (5, "X")): (0.00043511791324680484 + 0j),
                ((0, "Z"), (2, "Z"), (3, "Z")): (0.16824174504299702 + 0j),
                ((0, "Z"), (1, "Y"), (2, "Z"), (3, "X"), (5, "Y")): (0.029512711807110188 + 0j),
                ((1, "X"), (2, "Z"), (3, "X"), (4, "Z"), (5, "X")): (0.029512711807110188 + 0j),
                ((0, "Z"), (1, "Z"), (4, "Z")): (0.13726839006899005 + 0j),
                ((0, "Z"), (1, "X"), (3, "Y"), (4, "Z"), (5, "Y")): (0.011149373109704066 + 0j),
                ((1, "Y"), (3, "Y"), (5, "X")): (-0.011149373109704066 + 0j),
                ((0, "Z"), (1, "Z"), (4, "Z"), (5, "Z")): (0.12008313883007578 + 0j),
                ((1, "Z"), (3, "Z")): (0.22003977334376118 + 0j),
                ((2, "Y"), (4, "Y"), (5, "Z")): (0.009647475282106617 + 0j),
                ((1, "Z"), (2, "X"), (3, "Z"), (4, "X")): (0.009647475282106617 + 0j),
                ((2, "X"), (4, "X"), (5, "Z")): (0.009647475282106617 + 0j),
                ((1, "Z"), (2, "Y"), (3, "Z"), (4, "Y")): (0.009647475282106617 + 0j),
                ((2, "Z"), (4, "Z")): (0.13758959215600186 + 0j),
                ((2, "Z"), (4, "Z"), (5, "Z")): (0.1472370674381085 + 0j),
                ((1, "Z"), (2, "Z"), (3, "Z"), (4, "Z")): (0.1472370674381085 + 0j),
                ((1, "Z"), (2, "Z"), (3, "Z"), (4, "Z"), (5, "Z")): (0.13758959215600186 + 0j),
                ((5, "Z"),): (0.1492827559305538 + 0j),
            },
        ),
    ],
)
def test_observable_conversion(mol_name, terms_ref, custom_wires, monkeypatch):
    r"""Test the correctness of the QubitOperator observable conversion from
    OpenFermion to Pennylane.

    The parametrized inputs are `.terms` attribute of the output `QubitOperator`s based on
    the same set of test molecules as `test_gen_hamiltonian_pauli_basis`.

    The equality checking is implemented in the `qchem` module itself as it could be
    something useful to the users as well.
    """
    qOp = QubitOperator()
    if terms_ref is not None:
        monkeypatch.setattr(qOp, "terms", terms_ref)

    vqe_observable = qchem.convert_observable(qOp, custom_wires)

    if isinstance(custom_wires, dict):
        custom_wires = {v: k for k, v in custom_wires.items()}

    assert qchem._qubit_operators_equivalent(qOp, vqe_observable, custom_wires)


def test_not_xyz_terms_to_qubit_operator():
    r"""Test if the conversion complains about non Pauli matrix observables"""
    with pytest.raises(
        ValueError,
        match="Expected only PennyLane observables PauliX/Y/Z or Identity, but also got {"
        "'QuadOperator'}.",
    ):
        qchem._terms_to_qubit_operator(
            np.array([0.1 + 0.0j, 0.0]),
            [
                qml.operation.Tensor(qml.PauliX(0)),
                qml.operation.Tensor(qml.PauliZ(0), qml.QuadOperator(0.1, wires=1)),
            ],
        )


def test_types_consistency():
    r"""Test the type consistency of the qubit Hamiltonian constructed by 'convert_observable' from
    an OpenFermion QubitOperator with respect to the same observable built directly using PennyLane
    operations"""

    # Reference PL operator
    pl_ref = 1 * qml.Identity(0) + 2 * qml.PauliZ(0) @ qml.PauliX(1)

    # Corresponding OpenFermion QubitOperator
    of = QubitOperator("", 1) + QubitOperator("Z0 X1", 2)

    # Build PL operator using 'convert_observable'
    pl = qchem.convert_observable(of)

    ops = pl.terms()[1]
    ops_ref = pl_ref.terms()[1]

    for i, op in enumerate(ops):
        assert op.name == ops_ref[i].name
        assert type(op) == type(ops_ref[i])


op_1 = QubitOperator("Y0 Y1", 1 + 0j) + QubitOperator("Y0 X1", 2) + QubitOperator("Z0 Y1", 2.3e-08j)
op_2 = QubitOperator("Z0 Y1", 2.23e-10j)


@pytest.mark.parametrize(
    ("qubit_op", "tol"),
    [
        (op_1, 1e08),
        (op_2, 1e06),
    ],
)
def test_exception_convert_observable(qubit_op, tol):
    r"""Test that an error is raised if the QubitOperator contains complex coefficients.
    Currently the Hamiltonian class does not support complex coefficients.
    """
    with pytest.raises(TypeError, match="The coefficients entering the QubitOperator must be real"):
        qchem.convert_observable(qubit_op, tol=tol)


def test_identities_terms_to_qubit_operator():
    """Test that tensor products that contain Identity instances are handled
    correctly by the _terms_to_qubit_operator function.

    A decomposition of the following observable was used:
    [[1 0 0 0]
     [0 2 0 0]
     [0 0 3 0]
     [0 0 0 4]]
    """
    coeffs = [2.5, -0.5, -1.0]
    obs_list = [
        qml.Identity(wires=[0]) @ qml.Identity(wires=[1]),
        qml.Identity(wires=[0]) @ qml.PauliZ(wires=[1]),
        qml.PauliZ(wires=[0]) @ qml.Identity(wires=[1]),
    ]

    op_str = str(qchem._terms_to_qubit_operator(coeffs, obs_list))

    # Remove new line characters
    op_str = op_str.replace("\n", "")
    assert op_str == "2.5 [] +-1.0 [Z0] +-0.5 [Z1]"


def test_terms_to_qubit_operator_no_decomp():
    """Test the _terms_to_qubit_operator function with custom wires."""
    coeffs = np.array([0.1, 0.2])
    ops = [
        qml.operation.Tensor(qml.PauliX(wires=["w0"])),
        qml.operation.Tensor(qml.PauliY(wires=["w0"]), qml.PauliZ(wires=["w2"])),
    ]
    op_str = str(
        qchem._terms_to_qubit_operator(coeffs, ops, wires=qml.wires.Wires(["w0", "w1", "w2"]))
    )

    # Remove new line characters
    op_str = op_str.replace("\n", "")
    expected = "0.1 [X0] +0.2 [Y0 Z2]"
    assert op_str == expected


@pytest.mark.parametrize(
    ("mol_name", "terms_ref", "expected_cost"),
    [
        ("empty", None, 0),
        (
            "h2_psycf [jordan_WIGNER]",
            {
                (): (-0.04207897647782188 + 0j),
                ((0, "Z"),): (0.17771287465139934 + 0j),
                ((1, "Z"),): (0.1777128746513993 + 0j),
                ((2, "Z"),): (-0.24274280513140484 + 0j),
                ((3, "Z"),): (-0.24274280513140484 + 0j),
                ((0, "Z"), (1, "Z")): (0.17059738328801055 + 0j),
                ((0, "Y"), (1, "X"), (2, "X"), (3, "Y")): (0.04475014401535161 + 0j),
                ((0, "Y"), (1, "Y"), (2, "X"), (3, "X")): (-0.04475014401535161 + 0j),
                ((0, "X"), (1, "X"), (2, "Y"), (3, "Y")): (-0.04475014401535161 + 0j),
                ((0, "X"), (1, "Y"), (2, "Y"), (3, "X")): (0.04475014401535161 + 0j),
                ((0, "Z"), (2, "Z")): (0.12293305056183801 + 0j),
                ((0, "Z"), (3, "Z")): (0.1676831945771896 + 0j),
                ((1, "Z"), (2, "Z")): (0.1676831945771896 + 0j),
                ((1, "Z"), (3, "Z")): (0.12293305056183801 + 0j),
                ((2, "Z"), (3, "Z")): (0.176276408043196 + 0j),
            },
            (0.7384971473437577 + 0j),
        ),
    ],
)
def test_integration_observable_to_vqe_cost(
    monkeypatch, mol_name, terms_ref, expected_cost, custom_wires, tol
):
    r"""Test if `convert_observable()` in qchem integrates with `ExpvalCost()` in pennylane"""

    qOp = QubitOperator()
    if terms_ref is not None:
        monkeypatch.setattr(qOp, "terms", terms_ref)
    vqe_observable = qchem.convert_observable(qOp, custom_wires)

    num_qubits = len(vqe_observable.wires)
    assert vqe_observable.terms().__repr__()  # just to satisfy codecov

    if custom_wires is None:
        wires = num_qubits
    elif isinstance(custom_wires, dict):
        wires = qchem.structure._process_wires(custom_wires)
    else:
        wires = custom_wires[:num_qubits]
    dev = qml.device("default.qubit", wires=wires)

    # can replace the ansatz with more suitable ones later.
    def dummy_ansatz(phis, wires):
        for phi, w in zip(phis, wires):
            qml.RX(phi, wires=w)

    dummy_cost = qml.ExpvalCost(dummy_ansatz, vqe_observable, dev)
    params = [0.1 * i for i in range(num_qubits)]
    res = dummy_cost(params)

    assert np.allclose(res, expected_cost, **tol)


@pytest.mark.parametrize(
    ("name", "core", "active", "mapping", "expected_cost"),
    [
        ("lih", [0], [1, 2], "jordan_WIGNER", -7.255500051039507),
        ("lih", [0], [1, 2], "BRAVYI_kitaev", -7.246409364088741),
        ("h2_pyscf", list(range(0)), list(range(2)), "jordan_WIGNER", 0.19364907363263958),
        ("h2_pyscf", list(range(0)), list(range(2)), "BRAVYI_kitaev", 0.16518000728327564),
        ("gdb3", list(range(11)), [11, 12], "jordan_WIGNER", -130.59816885313248),
        ("gdb3", list(range(11)), [11, 12], "BRAVYI_kitaev", -130.6156540164148),
    ],
)
def test_integration_mol_file_to_vqe_cost(
    name, core, active, mapping, expected_cost, custom_wires, tol
):
    r"""Test if the output of `decompose()` works with `convert_observable()`
    to generate `ExpvalCost()`"""

    ref_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_ref_files")
    hf_file = os.path.join(ref_dir, name)
    qubit_hamiltonian = qchem.decompose(
        hf_file,
        mapping=mapping,
        core=core,
        active=active,
    )

    vqe_hamiltonian = qchem.convert_observable(qubit_hamiltonian, custom_wires)
    assert len(vqe_hamiltonian.ops) > 1  # just to check if this runs

    num_qubits = len(vqe_hamiltonian.wires)
    assert num_qubits == 2 * len(active)

    if custom_wires is None:
        wires = num_qubits
    elif isinstance(custom_wires, dict):
        wires = qchem.structure._process_wires(custom_wires)
    else:
        wires = custom_wires[:num_qubits]
    dev = qml.device("default.qubit", wires=wires)

    # can replace the ansatz with more suitable ones later.
    def dummy_ansatz(phis, wires):
        for phi, w in zip(phis, wires):
            qml.RX(phi, wires=w)

    phis = np.load(os.path.join(ref_dir, "dummy_ansatz_parameters.npy"))

    dummy_cost = qml.ExpvalCost(dummy_ansatz, vqe_hamiltonian, dev)
    res = dummy_cost(phis)

    assert np.abs(res - expected_cost) < tol["atol"]


@pytest.mark.parametrize("n_wires", [None, 6])
def test_process_wires(custom_wires, n_wires):
    r"""Test if _process_wires handels different combinations of input types correctly."""

    wires = qchem.structure._process_wires(custom_wires, n_wires)

    assert isinstance(wires, qml.wires.Wires)

    expected_length = (
        n_wires if n_wires is not None else len(custom_wires) if custom_wires is not None else 1
    )
    if len(wires) > expected_length:
        assert isinstance(custom_wires, dict)
        assert len(wires) == max(custom_wires) + 1
    else:
        assert len(wires) == expected_length

    if custom_wires is not None and n_wires is not None:
        if not isinstance(custom_wires, dict):
            assert wires == qchem.structure._process_wires(custom_wires[:n_wires], n_wires)
        else:
            assert wires == qchem.structure._process_wires(
                {k: v for k, v in custom_wires.items()}, n_wires
            )


def test_process_wires_raises():
    """Test if exceptions are raised for _wire_proc()"""

    with pytest.raises(ValueError, match="Expected only int-keyed or consecutive int-valued dict"):
        qchem.structure._process_wires({"a": "b"})

    with pytest.raises(ValueError, match="Expected type Wires, list, tuple, or dict"):
        qchem.structure._process_wires(1.2)

    with pytest.raises(ValueError, match="Length of `wires`"):
        qchem.structure._process_wires([3, 4], 3)
