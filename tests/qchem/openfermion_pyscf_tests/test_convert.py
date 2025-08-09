# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unit tests for functions needed for converting objects obtained from external libraries to a
PennyLane object.
"""
# pylint: disable=too-many-arguments,protected-access
import os
import sys

import pytest

import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem

openfermion = pytest.importorskip("openfermion")
openfermionpyscf = pytest.importorskip("openfermionpyscf")
pyscf = pytest.importorskip("pyscf")

pauli_ops_and_prod = (qml.PauliX, qml.PauliY, qml.PauliZ, qml.Identity, qml.ops.Prod)


@pytest.fixture(
    scope="module",
    params=[
        None,
        qml.wires.Wires(
            list("ab") + [-3, 42] + ["xyz", "23", "wireX"] + [f"w{i}" for i in range(20)]
        ),
        list(range(100, 120)),
        {13 - i: "abcdefghijklmn"[i] for i in range(14)},
    ],
    name="custom_wires",
)
def custom_wires_fixture(request):
    """Custom wire mapping for Pennylane<->OpenFermion conversion"""
    return request.param


@pytest.fixture(scope="session", name="tol")
def tol_fixture():
    """Numerical tolerance for equality tests."""
    return {"rtol": 0, "atol": 1e-8}


@pytest.mark.parametrize(
    ("_", "terms_ref"),
    [
        ("empty", None),
        ("singlewire", {((0, "Z"),): (0.155924093421341 + 0j)}),
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
def test_observable_conversion(_, terms_ref, custom_wires, monkeypatch):
    r"""Test the correctness of the QubitOperator observable conversion from
    OpenFermion to Pennylane.

    The parametrized inputs are `.terms` attribute of the output `QubitOperator`s based on
    the same set of test molecules as `test_gen_hamiltonian_pauli_basis`.

    The equality checking is implemented in the `convert` module itself as it could be
    something useful to the users as well.
    """
    qOp = openfermion.QubitOperator()

    if terms_ref is not None:
        monkeypatch.setattr(qOp, "terms", terms_ref)

    vqe_observable = qml.qchem.convert.import_operator(qOp, "openfermion", custom_wires)

    if isinstance(custom_wires, dict):
        custom_wires = {v: k for k, v in custom_wires.items()}

    assert qml.qchem.convert._openfermion_pennylane_equivalent(qOp, vqe_observable, custom_wires)


ops_wires = (
    (
        ([0.1, 0.2], [qml.PauliZ(0), qml.Identity(1)]),
        (0.1 * openfermion.QubitOperator("Z0") + 0.2 * openfermion.QubitOperator("")),
        [0, 1],
    ),
    (
        ([0.1, 0.2, 0.3], [qml.PauliX(0), qml.PauliY(1), qml.PauliZ(2)]),
        (
            0.1 * openfermion.QubitOperator("X0")
            + 0.2 * openfermion.QubitOperator("Y1")
            + 0.3 * openfermion.QubitOperator("Z2")
        ),
        [0, 1, 2],
    ),
    (
        ([0.1], [qml.s_prod(0.5, qml.PauliZ(0))]),
        0.05 * openfermion.QubitOperator("Z0"),
        [0],
    ),
    (
        ([0.1, 0.2], [qml.PauliX(0), qml.prod(qml.PauliY(1), qml.PauliZ(2))]),
        (0.1 * openfermion.QubitOperator("X0") + 0.2 * openfermion.QubitOperator("Y1 Z2")),
        [0, 1, 2],
    ),
    (
        ([0.1, 0.2, 0.3], [qml.PauliX(0), qml.sum(qml.PauliY(1), qml.PauliZ(2)), qml.PauliY(1)]),
        (
            0.1 * openfermion.QubitOperator("X0")
            + 0.5 * openfermion.QubitOperator("Y1")
            + 0.2 * openfermion.QubitOperator("Z2")
        ),
        [0, 1, 2],
    ),
)


@pytest.mark.parametrize("pl_op, of_op, wire_order", ops_wires)
def test_operation_conversion(pl_op, of_op, wire_order):
    """Assert the conversion between pennylane and openfermion operators"""
    converted_pl_op = qml.qchem.convert._pennylane_to_openfermion(*pl_op)  # coeffs, ops lists
    assert of_op == converted_pl_op

    converted_of_op = qml.qchem.convert._openfermion_to_pennylane(of_op)
    _, converted_of_op_terms = converted_of_op

    assert all(isinstance(term, pauli_ops_and_prod) for term in converted_of_op_terms)

    assert np.allclose(
        qml.matrix(qml.dot(*pl_op), wire_order=wire_order),
        qml.matrix(qml.dot(*converted_of_op), wire_order=wire_order),
    )


@pytest.mark.parametrize(
    ("terms_ref", "lib_name"),
    [
        ({((0, "Z"),): (0.155924093421341 + 0j)}, "qiskit"),
    ],
)
def test_convert_format_not_supported(terms_ref, lib_name, monkeypatch):
    """Test if an ImportError is raised when openfermion is requested but not installed"""

    qOp = openfermion.QubitOperator()
    if terms_ref is not None:
        monkeypatch.setattr(qOp, "terms", terms_ref)

    with pytest.raises(TypeError, match="Converter does not exist for"):
        qml.qchem.convert.import_operator(qOp, format=lib_name)


invalid_ops = (
    qml.prod(qml.PauliZ(0), qml.QuadOperator(0.1, wires=1)),
    qml.prod(qml.PauliX(0), qml.Hadamard(1)),
    qml.sum(qml.PauliZ(0), qml.Hadamard(1)),
)


@pytest.mark.parametrize("op", invalid_ops)
def test_not_xyz_pennylane_to_openfermion(op):
    r"""Test if the conversion complains about non Pauli matrix observables"""
    _match = "Expected a Pennylane operator with a valid Pauli word representation,"
    with pytest.raises(ValueError, match=_match):
        qml.qchem.convert._pennylane_to_openfermion(
            np.array([0.1 + 0.0j, 0.0]),
            [
                qml.prod(qml.PauliX(0)),
                op,
            ],
        )


def test_wires_not_covered_pennylane_to_openfermion():
    r"""Test if the conversion complains about Supplied wires not covering ops wires"""
    with pytest.raises(
        ValueError,
        match="Supplied `wires` does not cover all wires defined in `ops`.",
    ):
        qml.qchem.convert._pennylane_to_openfermion(
            np.array([0.1, 0.2]),
            [
                qml.prod(qml.PauliX(wires=["w0"])),
                qml.prod(qml.PauliY(wires=["w0"]), qml.PauliZ(wires=["w2"])),
            ],
            wires=qml.wires.Wires(["w0", "w1"]),
        )


def test_types_consistency():
    r"""Test the type consistency of the qubit Hamiltonian constructed by 'import_operator' from
    an OpenFermion QubitOperator with respect to the same observable built directly using PennyLane
    operations"""

    # Reference PL operator
    pl_ref = 1 * qml.Identity(0) + 2 * qml.PauliZ(0) @ qml.PauliX(1)

    # Corresponding OpenFermion QubitOperator
    of = openfermion.QubitOperator("", 1) + openfermion.QubitOperator("Z0 X1", 2)

    # Build PL operator using 'import_operator'
    pl = qml.qchem.convert.import_operator(of, "openfermion")

    ops = pl.terms()[1]
    ops_ref = pl_ref.terms()[1]

    for i, op in enumerate(ops):
        assert op.name == ops_ref[i].name
        assert isinstance(op, type(ops_ref[i]))


of_pl_ops = (
    (
        (
            0.1 * openfermion.QubitOperator("X0")
            + 0.2 * openfermion.QubitOperator("Y1")
            + 0.3 * openfermion.QubitOperator("Z2")
            + 0.4 * openfermion.QubitOperator("")
        ),
        qml.Hamiltonian(
            [0.1, 0.2, 0.3, 0.4],
            [qml.PauliX("w0"), qml.PauliY("w1"), qml.PauliZ("w2"), qml.Identity("w0")],
        ),
        qml.sum(
            qml.s_prod(0.1, qml.PauliX("w0")),
            qml.s_prod(0.2, qml.PauliY("w1")),
            qml.s_prod(0.3, qml.PauliZ("w2")),
            qml.s_prod(0.4, qml.Identity("w0")),
        ),
        ["w0", "w1", "w2"],
    ),
    (
        (0.1 * openfermion.QubitOperator("X0 Y1") + 0.2 * openfermion.QubitOperator("Z2")),
        qml.Hamiltonian([0.1, 0.2], [qml.PauliX("w0") @ qml.PauliY("w1"), qml.PauliZ("w2")]),
        qml.sum(
            qml.s_prod(0.1, qml.prod(qml.PauliX("w0"), qml.PauliY("w1"))),
            qml.s_prod(0.2, qml.PauliZ("w2")),
        ),
        ["w0", "w1", "w2"],
    ),
    (
        (0.1 * openfermion.QubitOperator("X0 Y1")),
        qml.Hamiltonian([0.1], [qml.PauliX("w0") @ qml.PauliY("w1")]),
        qml.s_prod(0.1, qml.prod(qml.PauliX("w0"), qml.PauliY("w1"))),
        ["w0", "w1"],
    ),
)


@pytest.mark.parametrize("of_op, pl_h, pl_op, wires", of_pl_ops)
def test_import_operator(of_op, pl_h, pl_op, wires):
    """Test the import_operator function correctly imports an OpenFermion operator into a PL one."""
    of_h = qml.qchem.convert.import_operator(of_op, "openfermion", wires=wires)
    assert qml.pauli.pauli_sentence(pl_h) == qml.pauli.pauli_sentence(of_h)

    assert isinstance(of_h, type(pl_op))

    if isinstance(of_h, qml.ops.Sum):
        assert all(
            isinstance(term, qml.ops.SProd) and isinstance(term.base, pauli_ops_and_prod)
            for term in of_h.operands
        )
    assert np.allclose(qml.matrix(of_h, wire_order=wires), qml.matrix(pl_op, wire_order=wires))


op_1 = (
    openfermion.QubitOperator("Y0 Y1", 1 + 0j)
    + openfermion.QubitOperator("Y0 X1", 2)
    + openfermion.QubitOperator("Z0 Y1", 2.3e-08j)
)
op_2 = openfermion.QubitOperator("Z0 Y1", 2.3e-6j)


@pytest.mark.parametrize(
    ("qubit_op", "tol"),
    [
        (op_1, 1e-8),
        (op_2, 1e-10),
    ],
)
def test_exception_import_operator(qubit_op, tol):
    r"""Test that a warning is raised if the QubitOperator contains complex coefficients.
    Currently, the Hamiltonian class does not support complex coefficients.
    """
    with pytest.warns(
        UserWarning, match="The coefficients entering the QubitOperator must be real"
    ):
        qml.qchem.convert.import_operator(qubit_op, "openfermion", tol=tol)


def test_identities_pennylane_to_openfermion():
    """Test that tensor products that contain Identity instances are handled
    correctly by the _pennylane_to_openfermion function.

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

    op_str = str(qml.qchem.convert._pennylane_to_openfermion(coeffs, obs_list))

    # Remove new line characters
    op_str = op_str.replace("\n", "")

    assert op_str == "(2.5+0j) [] +(-1+0j) [Z0] +(-0.5+0j) [Z1]"


def test_singlewire_pennylane_to_openfermion():
    """Test that _pennylane_to_openfermion function returns the correct Hamiltonian for a
    single-wire case.
    """
    coeffs = np.array([0.5])
    obs_list = [qml.PauliZ(wires=[0])]

    op_str = str(qml.qchem.convert._pennylane_to_openfermion(coeffs, obs_list))

    # Remove new line characters
    op_str = op_str.replace("\n", "")
    assert op_str == "(0.5+0j) [Z0]"


def test_pennylane_to_openfermion_no_decomp():
    """Test the _pennylane_to_openfermion function with custom wires."""
    coeffs = np.array([0.1, 0.2])
    ops = [
        qml.prod(qml.PauliX(wires=["w0"])),
        qml.prod(qml.PauliY(wires=["w0"]), qml.PauliZ(wires=["w2"])),
    ]
    op_str = str(
        qml.qchem.convert._pennylane_to_openfermion(
            coeffs, ops, wires=qml.wires.Wires(["w0", "w1", "w2"])
        )
    )

    # Remove new line characters
    op_str = op_str.replace("\n", "")
    expected = "(0.1+0j) [X0] +(0.2+0j) [Y0 Z2]"
    assert op_str == expected


@pytest.mark.parametrize(
    ("_", "terms_ref", "expected_cost"),
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
    monkeypatch, _, terms_ref, expected_cost, custom_wires, tol
):
    r"""Test if `import_operator()` integrates with `QNode` in pennylane"""

    qOp = openfermion.QubitOperator()
    if terms_ref is not None:
        monkeypatch.setattr(qOp, "terms", terms_ref)
    vqe_observable = qml.qchem.convert.import_operator(qOp, "openfermion", custom_wires)

    num_qubits = len(vqe_observable.wires)

    if custom_wires is None:
        wires = num_qubits
    elif isinstance(custom_wires, dict):
        wires = qml.qchem.convert._process_wires(custom_wires)
    else:
        wires = custom_wires[:num_qubits]

    dev = qml.device("default.qubit", wires=wires)

    @qml.qnode(dev)
    def dummy_cost(params):
        for phi, w in zip(params, dev.wires):
            qml.RX(phi, wires=w)
        return qml.expval(vqe_observable)

    params = [0.1 * i for i in range(num_qubits)]
    res = dummy_cost(params)

    assert np.allclose(res, expected_cost, **tol)


@pytest.mark.parametrize("n_wires", [None, 6])
def test_process_wires(custom_wires, n_wires):
    r"""Test if _process_wires handels different combinations of input types correctly."""

    wires = qml.qchem.convert._process_wires(custom_wires, n_wires)

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
            assert wires == qml.qchem.convert._process_wires(custom_wires[:n_wires], n_wires)
        else:
            assert wires == qml.qchem.convert._process_wires(custom_wires, n_wires)


def test_process_wires_raises():
    """Test if exceptions are raised for _wire_proc()"""

    with pytest.raises(ValueError, match="Expected only int-keyed or consecutive int-valued dict"):
        qml.qchem.convert._process_wires({"a": "b"})

    with pytest.raises(ValueError, match="Expected type Wires, list, tuple, or dict"):
        qml.qchem.convert._process_wires(1.2)

    with pytest.raises(ValueError, match="Length of `wires`"):
        qml.qchem.convert._process_wires([3, 4], 3)


def test_fail_import_openfermion(monkeypatch):
    """Test if an ImportError is raised when openfermion is requested but not installed"""

    with monkeypatch.context() as m:
        m.setitem(sys.modules, "openfermion", None)

        with pytest.raises(ImportError, match="This feature requires openfermion"):
            qml.qchem.convert._pennylane_to_openfermion(
                np.array([0.1 + 0.0j, 0.0]),
                [
                    qml.prod(qml.PauliX(0)),
                    qml.prod(qml.PauliZ(0), qml.QuadOperator(0.1, wires=1)),
                ],
            )


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
    r"""Test if the output of `decompose()` works with `import_operator()`
    to generate `QNode`"""
    ref_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "test_ref_files")
    hf_file = os.path.join(ref_dir, name)
    qubit_hamiltonian = qchem.decompose(
        hf_file,
        mapping=mapping,
        core=core,
        active=active,
    )

    vqe_hamiltonian = qml.qchem.convert.import_operator(
        qubit_hamiltonian, wires=custom_wires, format="openfermion"
    )
    assert len(vqe_hamiltonian.terms()[1]) > 1  # just to check if this runs

    num_qubits = len(vqe_hamiltonian.wires)
    assert num_qubits == 2 * len(active)

    if custom_wires is None:
        wires = num_qubits
    elif isinstance(custom_wires, dict):
        wires = qml.qchem.convert._process_wires(custom_wires)
    else:
        wires = custom_wires[:num_qubits]

    dev = qml.device("default.qubit", wires=wires)
    phis = np.load(os.path.join(ref_dir, "dummy_ansatz_parameters.npy"))

    @qml.qnode(dev)
    def dummy_cost(params):
        for phi, w in zip(params, dev.wires):
            qml.RX(phi, wires=w)
        return qml.expval(vqe_hamiltonian)

    res = dummy_cost(phis)

    assert np.abs(res - expected_cost) < tol["atol"]


@pytest.mark.parametrize(
    ("electrons", "orbitals", "singles_ref", "doubles_ref"),
    [
        # trivial case, e.g., H2/STO-3G
        (2, 4, [[0, 2], [0, 3], [1, 2], [1, 3]], [[0, 1, 2, 3]]),
    ],
)
def test_excitations(electrons, orbitals, singles_ref, doubles_ref):
    r"""Test if the _excitations function returns correct single and double excitations."""
    singles, doubles = qchem.convert._excitations(electrons, orbitals)
    assert singles == singles_ref
    assert doubles == doubles_ref


@pytest.mark.parametrize(
    ("electrons", "orbitals", "excitation", "states_ref", "signs_ref"),
    [
        # reference data computed with pyscf:
        # pyscf_addrs, pyscf_signs = pyscf.ci.cisd.tn_addrs_signs(orbitals, electrons, excitation)
        # pyscf_state = pyscf.fci.cistring.addrs2str(orbitals, electrons, pyscf_addrs)
        # pyscf_state, pyscf_signs
        (
            3,
            8,
            1,
            np.array([14, 22, 38, 70, 134, 13, 21, 37, 69, 133, 11, 19, 35, 67, 131]),
            np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1]),
        ),
        (
            3,
            6,
            2,
            np.array([28, 44, 52, 26, 42, 50, 25, 41, 49]),
            np.array([1, 1, 1, -1, -1, -1, 1, 1, 1]),
        ),
    ],
)
def test_excited_configurations(electrons, orbitals, excitation, states_ref, signs_ref):
    r"""Test if the _excited_configurations function returns correct states and signs."""
    states, signs = qchem.convert._excited_configurations(electrons, orbitals, excitation)
    assert np.allclose(states, states_ref)
    assert np.allclose(signs, signs_ref)


@pytest.mark.parametrize(
    ("wf_dict", "n_orbitals", "string_ref", "coeff_ref"),
    [  # reference data were obtained manually
        (  #  0.87006284 |1100> + 0.3866946 |1001> + 0.29002095 |0110> + 0.09667365 |0011>
            {(1, 1): 0.87006284, (1, 2): 0.3866946, (2, 1): 0.29002095, (2, 2): 0.09667365},
            2,
            ["1100", "1001", "0110", "0011"],
            [0.87006284, 0.3866946, 0.29002095, 0.09667365],
        ),
        (  # 0.80448616 |110000> + 0.53976564 |001100> + 0.22350293 |000011> + 0.10724511 |100100>
            {(1, 1): 0.80448616, (2, 2): 0.53976564, (4, 4): 0.22350293, (1, 2): 0.10724511},
            3,
            ["110000", "001100", "000011", "100100"],
            [0.80448616, 0.53976564, 0.22350293, 0.10724511],
        ),
    ],
)
def test_wfdict_to_statevector(wf_dict, n_orbitals, string_ref, coeff_ref):
    r"""Test that _wfdict_to_statevector returns the correct state vector."""
    wf_ref = np.zeros(2 ** (n_orbitals * 2))
    idx_nonzero = [int(s, 2) for s in string_ref]
    wf_ref[idx_nonzero] = coeff_ref

    wf_comp = qchem.convert._wfdict_to_statevector(wf_dict, n_orbitals)

    assert np.allclose(wf_comp, wf_ref)


@pytest.mark.parametrize(
    ("molecule", "basis", "symm", "wf_ref"),
    [
        (
            [["H", (0, 0, 0)], ["H", (0, 0, 0.71)]],
            "sto6g",
            "d2h",
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    -0.1066467,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.99429698,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        ),
    ],
)
@pytest.mark.parametrize("method", ["rcisd", "ucisd", "rccsd", "uccsd"])
def test_import_state_pyscf(molecule, basis, symm, method, wf_ref):
    r"""Test that import_state returns the correct state vector."""

    mol = pyscf.gto.M(atom=molecule, basis=basis, symmetry=symm)

    if method == "rcisd":
        myhf = pyscf.scf.RHF(mol).run()
        solver = pyscf.ci.cisd.RCISD(myhf).run()
    elif method == "ucisd":
        myhf = pyscf.scf.UHF(mol).run()
        solver = pyscf.ci.ucisd.UCISD(myhf).run()
    elif method == "rccsd":
        myhf = pyscf.scf.RHF(mol).run()
        solver = pyscf.cc.rccsd.RCCSD(myhf).run()
    elif method == "uccsd":
        myhf = pyscf.scf.UHF(mol).run()
        solver = pyscf.cc.uccsd.UCCSD(myhf).run()
    else:
        assert False, "Invalid method"

    wf_comp = qchem.convert.import_state(solver)

    # overall sign could be different in each PySCF run
    assert np.allclose(wf_comp, wf_ref) or np.allclose(wf_comp, -wf_ref)


@pytest.mark.parametrize(
    ("detscoeffs", "wf_ref"),
    [
        (
            # dmrg
            ([[0, 3], [3, 0]], np.array([-0.10660077, 0.9943019])),
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    -0.10660077,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.9943019,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        ),
        (
            # shci
            (["02", "20"], np.array([-0.1066467, 0.99429698])),
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    -0.1066467,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.99429698,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        ),
    ],
)
def test_import_state_nonpyscf(detscoeffs, wf_ref):
    r"""Test that import_state returns the correct state vector."""

    wf_comp = qchem.convert.import_state(detscoeffs)

    # overall sign could be different in each PySCF run
    assert np.allclose(wf_comp, wf_ref) or np.allclose(wf_comp, -wf_ref)


def test_import_state_error():
    r"""Test that an error is raised by import_state if a wrong object is entered."""

    myci = "wrongobject"

    with pytest.raises(ValueError, match="The supported objects"):
        qchem.convert.import_state(myci)

    mytuple = (np.array([[3, 0], [0, 3]]), np.array([0]))

    with pytest.raises(ValueError, match="For tuple input"):
        qchem.convert.import_state(mytuple)

    mytuple = ([[3, 0], [0, 3]], [0], [0])

    with pytest.raises(ValueError, match="The supported objects"):
        qchem.convert.import_state(mytuple)


@pytest.mark.parametrize(("excitation"), [-1, 0, 3])
def test_excited_configurations_error(excitation):
    r"""Test that an error is raised by _excited_configurations if a wrong excitation is entered."""
    with pytest.raises(ValueError, match="excitations are supported"):
        _ = qchem.convert._excited_configurations(2, 4, excitation)


h2_molecule = [["H", (0, 0, 0)], ["H", (0, 0, 0.71)]]
h3_molecule = [["H", (0, 0, 0)], ["H", (0, 1.2, 1.2)], ["H", (0, 0, 2.4)]]
lih_molecule = [["Li", (0, 0, 0)], ["H", (0, 0, 1.2)]]
beh_molecule = [["Be", (0, 0, 0)], ["H", (0, 0, 1.2)]]
h2_wf_sto6g = {(1, 1): -0.9942969785398778, (2, 2): 0.10664669927602179}  # tol = 1e-1
h2_wf_ccpvdz = {  # tol = 4e-2
    (1, 1): 0.9919704795977625,
    (2, 2): -0.048530356564386895,
    (2, 8): 0.044523330850078625,
    (4, 4): -0.050035945684911876,
    (8, 2): 0.04452333085007864,
    (8, 8): -0.052262303220437775,
    (16, 16): -0.040475973747662694,
    (32, 32): -0.040475973747662694,
}

li2_molecule = [["Li", (0, 0, 0)], ["Li", (0, 0, 0.71)]]
li2_wf_sto6g = {  # tol = 1e-1
    (7, 7): 0.8886970081919591,
    (11, 11): -0.3058459002168582,
    (19, 19): -0.30584590021685887,
    (35, 35): -0.14507552387854625,
}


# shci
h3p_shci_dets_coeffs = (
    ["200", "020", "002", "b0a", "a0b"],
    np.array([0.9389761486, -0.278704004, -0.1838389711, -0.0585282123, -0.0585282123]),
)
h3p_shci_e = -1.0862426041366735
lihpp_shci_dets_coeffs = (
    [
        "200000",
        "020000",
        "000020",
        "0b00a0",
        "0a00b0",
        "002000",
        "000200",
        "000002",
        "0b000a",
        "0a000b",
        "0000ba",
        "0000ab",
    ],
    np.array(
        [
            0.9999782547,
            -0.0035481253,
            -0.0033237037,
            -0.0025347719,
            -0.0025347719,
            -0.0015751526,
            -0.0015751526,
            -0.0008477427,
            -0.0007073151,
            -0.0007073151,
            0.000401425,
            0.000401425,
        ]
    ),
)
lihpp_shci_e = -6.781988968692473
h3_shci_dets_coeffs = (
    ["2a0", "aba", "0a2", "aab", "baa"],
    np.array([0.8614786698, 0.322024811, 0.319505036, 0.1715258542, 0.1504989567]),
)
h3_shci_e = -1.454861277373169
lih_shci_dets_coeffs = (
    [
        "220000",
        "200002",
        "20b00a",
        "20a00b",
        "200200",
        "200020",
        "2ba000",
        "2ab000",
        "202000",
        "022000",
        "ba0200",
        "ab0200",
        "ba0020",
        "ab0020",
        "abb00a",
        "baa00b",
        "bba00a",
        "aab00b",
        "020200",
        "020020",
        "ba0002",
        "ab0002",
        "2b000a",
        "2a000b",
        "02b00a",
        "02a00b",
        "b0a002",
        "a0b002",
        "002002",
        "b0200a",
        "a0200b",
        "020002",
        "0ba002",
        "0ab002",
    ],
    np.array(
        [
            -0.9911126427,
            0.0971352826,
            -0.0441230599,
            -0.0441230599,
            0.031774824,
            0.031774824,
            -0.0299817085,
            -0.0299817085,
            0.0217765126,
            0.003770931,
            -0.0029408247,
            -0.0029408247,
            -0.0029408247,
            -0.0029408247,
            0.0019487434,
            0.0019487434,
            -0.0017840878,
            -0.0017840878,
            0.0017812537,
            0.0017812537,
            -0.001350467,
            -0.001350467,
            -0.0007362362,
            -0.0007362362,
            0.0006455811,
            0.0006455811,
            -0.0004461604,
            -0.0004461604,
            -0.000422693,
            0.0002477636,
            0.0002477636,
            0.0002293939,
            -0.0001387897,
            -0.0001387897,
        ]
    ),
)
lih_shci_e = -7.943187881293274
behp_shci_dets_coeffs = (
    [
        "220000",
        "200002",
        "20b00a",
        "20a00b",
        "200200",
        "200020",
        "202000",
        "2ba000",
        "2ab000",
        "2b000a",
        "2a000b",
        "022000",
        "ba0200",
        "ab0200",
        "ba0020",
        "ab0020",
        "020200",
        "020020",
        "020002",
        "bab00a",
        "aba00b",
        "abb00a",
        "baa00b",
        "b0020a",
        "b0002a",
        "a0020b",
        "a0002b",
        "bba00a",
        "aab00b",
        "ba2000",
        "ab2000",
        "b0200a",
        "a0200b",
        "002002",
        "002200",
        "002020",
        "000202",
        "000022",
        "02b00a",
        "02a00b",
        "b0a002",
        "a0b002",
        "0ba002",
        "0ab002",
    ],
    np.array(
        [
            0.9916223775,
            -0.0701573082,
            0.0457383365,
            0.0457383365,
            -0.0439638102,
            -0.0439638102,
            -0.0429077189,
            0.0300723377,
            0.0300723377,
            0.0027956902,
            0.0027956902,
            -0.0026612596,
            0.0025805103,
            0.0025805103,
            0.0025805103,
            0.0025805103,
            -0.0014738523,
            -0.0014738523,
            -0.0014321532,
            0.0013339653,
            0.0013339653,
            0.0008826921,
            0.0008826921,
            -0.0005961969,
            -0.0005961969,
            -0.0005961969,
            -0.0005961969,
            0.0004513426,
            0.0004513426,
            0.0004152687,
            0.0004152687,
            -0.0004127526,
            -0.0004127526,
            0.0002445009,
            0.0001889348,
            0.0001889348,
            0.000186323,
            0.000186323,
            0.00012713,
            0.00012713,
            7.81899e-05,
            7.81899e-05,
            -3.28429e-05,
            -3.28429e-05,
        ]
    ),
)
behp_shci_e = -14.834784634050825
beh_shci_dets_coeffs = (
    [
        "22a000",
        "20a002",
        "2a0200",
        "2a0020",
        "2a2000",
        "20a200",
        "20a020",
        "20200a",
        "2a0002",
        "2ab00a",
        "2aa00b",
        "20020a",
        "20002a",
        "22000a",
        "2ba00a",
        "a20200",
        "a20020",
        "aba200",
        "aba020",
        "baa200",
        "baa020",
        "02a002",
        "02a200",
        "02a020",
        "a2b00a",
        "a20002",
        "ba200a",
        "a2a00b",
        "ab200a",
        "b0a20a",
        "b0a02a",
        "b2a00a",
        "a0a20b",
        "a0a02b",
        "02200a",
        "a00202",
        "a00022",
        "aa200b",
        "a02002",
        "0a0220",
        "00a202",
        "00a022",
        "0a2200",
        "0a2020",
        "0a0202",
        "0a0022",
        "0a2002",
        "02020a",
        "02002a",
        "aab200",
        "aab020",
        "aa020b",
        "aa002b",
    ],
    np.array(
        [
            -0.9884697344,
            -0.0746546627,
            0.0683966601,
            0.0683966601,
            -0.0417752874,
            -0.0376712952,
            -0.0376712952,
            0.0370141999,
            -0.0247112865,
            0.0245969841,
            0.0159033543,
            -0.0128140165,
            -0.0128140165,
            0.0106079865,
            0.0086970692,
            -0.0032951676,
            -0.0032951676,
            0.0023815863,
            0.0023815863,
            0.0023463095,
            0.0023463095,
            -0.0016910066,
            -0.0014915787,
            -0.0014915787,
            -0.0014179848,
            -0.001181045,
            0.0011338562,
            -0.0009487498,
            0.0008708464,
            0.0005297078,
            0.0005297078,
            -0.000466079,
            -0.0003691234,
            -0.0003691234,
            0.0003391998,
            0.0003245304,
            0.0003245304,
            -0.0002595431,
            0.000228621,
            0.0002006272,
            -0.0001946553,
            -0.0001946553,
            0.0001543147,
            0.0001543147,
            0.0001027719,
            0.0001027719,
            -9.55056e-05,
            8.77035e-05,
            8.77035e-05,
            3.77675e-05,
            3.77675e-05,
            1.61517e-05,
            1.61517e-05,
        ]
    ),
)
beh_shci_e = -15.11001775498381

# dmrg
h3p_dmrg_dets_coeffs = (
    [[0, 0, 3], [0, 3, 0], [1, 0, 2], [2, 0, 1], [3, 0, 0]],
    np.array(
        [
            0.1838436556640729,
            0.27869890041510864,
            0.058518639425201226,
            -0.05853696090499803,
            -0.9389767958793727,
        ]
    ),
)
h3p_dmrg_e = -1.0862426034107344
lihpp_dmrg_dets_coeffs = (
    [
        [0, 1, 0, 0, 0, 2],
        [0, 1, 0, 0, 2, 0],
        [0, 2, 0, 0, 0, 1],
        [0, 2, 0, 0, 1, 0],
        [0, 3, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 2],
        [1, 0, 0, 0, 2, 0],
        [1, 2, 0, 0, 0, 0],
        [2, 0, 0, 0, 0, 1],
        [2, 0, 0, 0, 1, 0],
        [2, 1, 0, 0, 0, 0],
        [3, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 3],
        [0, 0, 0, 0, 1, 2],
        [0, 0, 0, 0, 2, 1],
        [0, 0, 0, 0, 3, 0],
        [0, 0, 0, 3, 0, 0],
        [0, 0, 3, 0, 0, 0],
    ],
    np.array(
        [
            -0.0007085379968001104,
            -0.0025498996125903938,
            0.0007085582612973782,
            0.0025499022933380584,
            -0.0035720668764806264,
            8.25369038452123e-06,
            0.00031201802679808036,
            0.0004236898625428547,
            -8.388113240073638e-06,
            -0.00031199640545515007,
            -0.0004236856143996911,
            0.9999777636770568,
            -0.0008482894337553155,
            0.00040053659690504937,
            -0.0004005242398938148,
            -0.003337398668550216,
            -0.001576661418251104,
            -0.001576713155939749,
        ]
    ),
)
lihpp_dmrg_e = -6.781990152954243
h3_dmrg_dets_coeffs = (
    [[0, 1, 3], [1, 1, 2], [1, 2, 1], [2, 1, 1], [3, 1, 0]],
    np.array(
        [
            -0.319505629799915,
            -0.1715263012109641,
            0.3220243489995388,
            -0.15049933790616382,
            0.8614784666009474,
        ]
    ),
)
h3_dmrg_e = -1.4548612773471408
lih_dmrg_dets_coeffs = (
    [
        [3, 0, 1, 0, 0, 2],
        [3, 0, 1, 0, 2, 0],
        [3, 0, 2, 0, 0, 1],
        [3, 0, 2, 0, 1, 0],
        [3, 0, 3, 0, 0, 0],
        [3, 1, 0, 0, 0, 2],
        [3, 1, 2, 0, 0, 0],
        [3, 2, 0, 0, 0, 1],
        [3, 2, 1, 0, 0, 0],
        [3, 3, 0, 0, 0, 0],
        [2, 1, 3, 0, 0, 0],
        [2, 2, 1, 0, 0, 1],
        [2, 3, 0, 0, 0, 1],
        [2, 3, 1, 0, 0, 0],
        [3, 0, 0, 0, 0, 3],
        [3, 0, 0, 0, 2, 1],
        [3, 0, 0, 0, 3, 0],
        [3, 0, 0, 1, 2, 0],
        [3, 0, 0, 2, 1, 0],
        [3, 0, 0, 3, 0, 0],
        [2, 1, 2, 0, 0, 1],
        [2, 0, 1, 0, 0, 3],
        [2, 0, 1, 0, 3, 0],
        [2, 0, 1, 3, 0, 0],
        [2, 0, 3, 0, 0, 1],
        [2, 1, 0, 0, 0, 3],
        [2, 1, 0, 0, 3, 0],
        [2, 1, 0, 2, 1, 0],
        [2, 1, 0, 3, 0, 0],
        [2, 1, 1, 0, 0, 2],
        [2, 0, 0, 0, 3, 1],
        [2, 0, 0, 3, 0, 1],
        [1, 2, 0, 0, 0, 3],
        [1, 2, 0, 0, 3, 0],
        [1, 2, 0, 3, 0, 0],
        [1, 2, 1, 0, 0, 2],
        [1, 2, 2, 0, 0, 1],
        [1, 2, 3, 0, 0, 0],
        [1, 3, 0, 0, 0, 2],
        [1, 3, 0, 0, 2, 0],
        [1, 3, 2, 0, 0, 0],
        [1, 1, 2, 0, 0, 2],
        [0, 3, 3, 0, 0, 0],
        [1, 0, 0, 0, 3, 2],
        [1, 0, 0, 3, 0, 2],
        [1, 0, 2, 0, 0, 3],
        [1, 0, 2, 0, 3, 0],
        [1, 0, 2, 3, 0, 0],
        [1, 0, 3, 0, 0, 2],
        [1, 1, 0, 2, 2, 0],
        [0, 3, 2, 0, 0, 1],
        [0, 3, 0, 0, 0, 3],
        [0, 3, 0, 0, 3, 0],
        [0, 3, 0, 3, 0, 0],
        [0, 3, 1, 0, 0, 2],
        [0, 2, 1, 0, 0, 3],
        [0, 2, 1, 0, 3, 0],
        [0, 2, 1, 3, 0, 0],
        [0, 2, 2, 0, 1, 1],
        [0, 2, 3, 0, 0, 1],
        [0, 2, 0, 0, 3, 1],
        [0, 2, 0, 3, 0, 1],
        [0, 1, 3, 0, 0, 2],
        [0, 0, 3, 0, 0, 3],
        [0, 0, 3, 0, 3, 0],
        [0, 0, 3, 3, 0, 0],
        [0, 1, 0, 0, 3, 2],
        [0, 1, 0, 3, 0, 2],
        [0, 1, 1, 0, 2, 2],
        [0, 1, 2, 0, 0, 3],
        [0, 1, 2, 0, 3, 0],
        [0, 1, 2, 3, 0, 0],
        [0, 0, 2, 0, 3, 1],
        [0, 0, 2, 3, 0, 1],
        [0, 0, 0, 0, 3, 3],
        [0, 0, 0, 3, 0, 3],
        [0, 0, 0, 3, 3, 0],
        [0, 0, 1, 0, 3, 2],
        [0, 0, 1, 3, 0, 2],
    ],
    np.array(
        [
            -0.044171641132296,
            3.1795384687324654e-06,
            0.0441471754744111,
            -1.1048529231780264e-06,
            0.021781337809214797,
            -0.0007541406168372599,
            -0.02998190748332192,
            0.0007480470015027966,
            0.03006472297329561,
            -0.991101592355426,
            0.0001727751091940219,
            0.0017986287021022045,
            -0.0004304931978993048,
            0.00037811354366125584,
            0.0971524925415711,
            -5.749313651592423e-06,
            0.03180156439692795,
            -7.793073571442597e-06,
            3.1687488514261055e-06,
            0.03180156432722111,
            0.00014505503777276387,
            0.0004327636819693626,
            7.486361667019857e-05,
            7.4863634268553e-05,
            0.00023832536332415937,
            0.001320301026083546,
            0.002981440283637068,
            -1.109514776770464e-06,
            0.002981440286862689,
            -0.0019437526671066635,
            0.0005965687970651896,
            0.0005965687973502533,
            -0.0013204446955014488,
            -0.0029819393186628364,
            -0.0029819393116002533,
            0.0001439850563675419,
            -0.0019431433608525308,
            -0.00017783529618577024,
            0.00042998230839144436,
            1.0803352048345973e-06,
            -0.0003785998883045307,
            0.0017983878856310921,
            0.0038007669882245756,
            -0.0005966987046913243,
            -0.0005966987011576368,
            -0.00043248586473046446,
            -7.460353135500195e-05,
            -7.460351979097924e-05,
            -0.0002383387791303137,
            -1.950966520722567e-06,
            -0.0006336288555314841,
            0.00022811261568425278,
            0.0017931215232747835,
            0.0017931215000451177,
            0.0006334759894128236,
            0.00013988966109576196,
            -6.461565284794182e-05,
            -6.46156451695468e-05,
            -1.1555156523550322e-06,
            -2.1813313136873242e-05,
            -5.230486766808598e-05,
            -5.230487111400336e-05,
            2.1522972249027938e-05,
            -0.00041956689083560257,
            -0.00015902010306485768,
            -0.00015902009503254204,
            5.232063180244838e-05,
            5.232063053242752e-05,
            1.2510889461358998e-06,
            -0.00013975755580558413,
            6.443074553771463e-05,
            6.443076599770788e-05,
            -7.28600906539348e-05,
            -7.286009825401676e-05,
            -0.0001744331681133361,
            -0.0001744331682637868,
            -9.99722324566731e-05,
            7.289491905274055e-05,
            7.289492292477615e-05,
        ]
    ),
)
lih_dmrg_e = -7.943195279734395
behp_dmrg_dets_coeffs = (
    [
        [3, 0, 1, 0, 0, 2],
        [3, 0, 2, 0, 0, 1],
        [3, 0, 3, 0, 0, 0],
        [3, 1, 0, 0, 0, 2],
        [3, 1, 2, 0, 0, 0],
        [3, 2, 0, 0, 0, 1],
        [3, 2, 1, 0, 0, 0],
        [3, 3, 0, 0, 0, 0],
        [2, 1, 2, 0, 0, 1],
        [2, 1, 3, 0, 0, 0],
        [2, 2, 0, 1, 1, 0],
        [2, 2, 1, 0, 0, 1],
        [2, 3, 0, 0, 0, 1],
        [2, 3, 0, 0, 1, 0],
        [2, 3, 1, 0, 0, 0],
        [3, 0, 0, 0, 0, 3],
        [3, 0, 0, 0, 3, 0],
        [3, 0, 0, 3, 0, 0],
        [2, 1, 1, 0, 0, 2],
        [2, 0, 0, 0, 3, 1],
        [2, 0, 0, 3, 0, 1],
        [2, 0, 1, 0, 0, 3],
        [2, 0, 1, 0, 3, 0],
        [2, 0, 1, 3, 0, 0],
        [2, 0, 3, 0, 0, 1],
        [2, 1, 0, 0, 0, 3],
        [2, 1, 0, 0, 3, 0],
        [2, 1, 0, 3, 0, 0],
        [1, 3, 0, 0, 0, 2],
        [1, 3, 2, 0, 0, 0],
        [1, 1, 2, 0, 0, 2],
        [1, 2, 0, 0, 0, 3],
        [1, 2, 0, 0, 3, 0],
        [1, 2, 0, 3, 0, 0],
        [1, 2, 1, 0, 0, 2],
        [1, 2, 2, 0, 0, 1],
        [1, 2, 3, 0, 0, 0],
        [1, 0, 3, 0, 0, 2],
        [0, 3, 0, 3, 0, 0],
        [0, 3, 1, 0, 0, 2],
        [0, 3, 2, 0, 0, 1],
        [0, 3, 3, 0, 0, 0],
        [1, 0, 0, 0, 3, 2],
        [1, 0, 0, 3, 0, 2],
        [1, 0, 2, 0, 0, 3],
        [1, 0, 2, 0, 3, 0],
        [1, 0, 2, 3, 0, 0],
        [0, 2, 3, 0, 0, 1],
        [0, 3, 0, 0, 0, 3],
        [0, 3, 0, 0, 3, 0],
        [0, 2, 0, 0, 3, 1],
        [0, 2, 0, 3, 0, 1],
        [0, 2, 1, 0, 0, 3],
        [0, 2, 1, 0, 3, 0],
        [0, 2, 1, 3, 0, 0],
        [0, 1, 3, 0, 0, 2],
        [0, 1, 0, 0, 3, 2],
        [0, 1, 0, 3, 0, 2],
        [0, 1, 2, 0, 0, 3],
        [0, 1, 2, 0, 3, 0],
        [0, 1, 2, 3, 0, 0],
        [0, 0, 2, 3, 0, 1],
        [0, 0, 3, 0, 0, 3],
        [0, 0, 3, 0, 3, 0],
        [0, 0, 3, 3, 0, 0],
        [0, 0, 1, 0, 3, 2],
        [0, 0, 1, 3, 0, 2],
        [0, 0, 2, 0, 3, 1],
        [0, 0, 0, 0, 3, 3],
        [0, 0, 0, 3, 0, 3],
        [0, 0, 0, 3, 3, 0],
    ],
    np.array(
        [
            -0.04574644026032889,
            0.04574537386192611,
            0.04291979725407384,
            -0.0028011537374583663,
            -0.03007781220719284,
            0.0028002641351943277,
            0.030074362830994565,
            -0.9916214085936498,
            -0.001344179255159954,
            0.00040994767000806457,
            1.0889425964770007e-06,
            0.0004530333735537102,
            -0.00032185119505419394,
            1.1244108685977887e-06,
            0.00014665385264667346,
            0.07015065398967743,
            0.04395952524672076,
            0.04395952524704585,
            0.0008907073828164922,
            0.0006042201058627647,
            0.0006042201063234662,
            7.931030012816218e-05,
            -0.00010036018486982971,
            -0.00010036018524266041,
            0.0004138836874252615,
            0.00016462742407944873,
            0.0025843163343285376,
            0.0025843163357712235,
            0.00032121788499602606,
            -0.00014662371324876012,
            0.00045345746915230613,
            -0.00016465869257110716,
            -0.002584322084504138,
            -0.0025843220855925448,
            -0.0013441695828928099,
            0.0008910494902545033,
            -0.0004111513370322702,
            -0.0004147876431848979,
            0.0014752895543995392,
            -0.00013371821622591742,
            0.00013361868173526148,
            0.0026660224315821343,
            -0.0006042486922325753,
            -0.0006042486937887367,
            -7.926606309602906e-05,
            0.00010034477417081034,
            0.00010034477405835584,
            8.85058847491531e-06,
            0.0014416236403467957,
            0.0014752895532286008,
            -4.6172356929958264e-05,
            -4.617235703735021e-05,
            -3.381772342001192e-05,
            -2.8745866647724918e-05,
            -2.8745863307208202e-05,
            -8.615520123995448e-06,
            4.6186326413730376e-05,
            4.6186327407198915e-05,
            3.386105493261175e-05,
            2.875495482614447e-05,
            2.8754956844198343e-05,
            -9.89261031640903e-05,
            -0.00024534982137603594,
            -0.00019390408459867852,
            -0.0001939040894728445,
            9.893126842166655e-05,
            9.893126744523666e-05,
            -9.892610315486499e-05,
            -0.00018675662702591196,
            -0.00018675662714923912,
            -0.00012426121855931997,
        ]
    ),
)
behp_dmrg_e = -14.834787198616041
beh_dmrg_dets_coeffs = (
    [
        [3, 1, 0, 0, 0, 3],
        [3, 1, 0, 0, 3, 0],
        [3, 1, 0, 3, 0, 0],
        [3, 1, 1, 0, 0, 2],
        [3, 1, 2, 0, 0, 1],
        [3, 1, 3, 0, 0, 0],
        [3, 2, 1, 0, 0, 1],
        [3, 3, 0, 0, 0, 1],
        [3, 3, 1, 0, 0, 0],
        [3, 0, 3, 0, 0, 1],
        [2, 1, 1, 3, 0, 0],
        [2, 1, 3, 0, 0, 1],
        [2, 3, 1, 0, 0, 1],
        [3, 0, 0, 0, 3, 1],
        [3, 0, 0, 3, 0, 1],
        [3, 0, 1, 0, 0, 3],
        [3, 0, 1, 0, 3, 0],
        [3, 0, 1, 3, 0, 0],
        [2, 1, 0, 0, 3, 1],
        [2, 1, 0, 3, 0, 1],
        [2, 1, 1, 0, 0, 3],
        [2, 1, 1, 0, 3, 0],
        [1, 3, 2, 0, 0, 1],
        [1, 3, 3, 0, 0, 0],
        [2, 0, 1, 0, 3, 1],
        [2, 0, 1, 3, 0, 1],
        [1, 3, 0, 0, 0, 3],
        [1, 3, 0, 0, 3, 0],
        [1, 3, 0, 3, 0, 0],
        [1, 3, 1, 0, 0, 2],
        [1, 2, 0, 0, 3, 1],
        [1, 2, 0, 3, 0, 1],
        [1, 2, 1, 0, 0, 3],
        [1, 2, 1, 0, 3, 0],
        [1, 2, 1, 3, 0, 0],
        [1, 2, 3, 0, 0, 1],
        [1, 1, 2, 3, 0, 0],
        [1, 1, 3, 0, 0, 2],
        [1, 1, 0, 0, 3, 2],
        [1, 1, 0, 3, 0, 2],
        [1, 1, 2, 0, 0, 3],
        [1, 1, 2, 0, 3, 0],
        [1, 0, 0, 0, 3, 3],
        [1, 0, 0, 3, 0, 3],
        [1, 0, 0, 3, 3, 0],
        [1, 0, 1, 0, 3, 2],
        [1, 0, 1, 3, 0, 2],
        [1, 0, 2, 0, 3, 1],
        [1, 0, 2, 3, 0, 1],
        [1, 0, 3, 0, 0, 3],
        [1, 0, 3, 0, 3, 0],
        [1, 0, 3, 3, 0, 0],
        [0, 2, 1, 0, 3, 1],
        [0, 2, 1, 3, 0, 1],
        [0, 3, 0, 0, 3, 1],
        [0, 3, 0, 3, 0, 1],
        [0, 3, 1, 0, 0, 3],
        [0, 3, 1, 0, 3, 0],
        [0, 3, 1, 3, 0, 0],
        [0, 3, 3, 0, 0, 1],
        [0, 1, 3, 0, 0, 3],
        [0, 1, 3, 0, 3, 0],
        [0, 1, 3, 3, 0, 0],
        [0, 1, 0, 3, 0, 3],
        [0, 1, 0, 3, 3, 0],
        [0, 1, 1, 0, 3, 2],
        [0, 1, 1, 3, 0, 2],
        [0, 1, 2, 0, 3, 1],
        [0, 1, 2, 3, 0, 1],
        [0, 0, 3, 3, 0, 1],
        [0, 1, 0, 0, 3, 3],
        [0, 0, 0, 3, 3, 1],
        [0, 0, 1, 0, 3, 3],
        [0, 0, 1, 3, 0, 3],
        [0, 0, 1, 3, 3, 0],
        [0, 0, 3, 0, 3, 1],
    ],
    np.array(
        [
            -0.024715466767529778,
            0.06840073088385241,
            0.06840073114002682,
            0.015906139503789715,
            -0.024604848890147215,
            -0.04179058308921919,
            0.008698709294383563,
            -0.01061502332170037,
            0.9884681843241176,
            -0.037018750518998944,
            -0.0023450206537299944,
            0.0011333651424117292,
            0.0004698778628639135,
            0.012812156739154595,
            0.012812156733123997,
            -0.07464913755537672,
            -0.037672406421505544,
            -0.037672406800031394,
            4.1732551720393404e-05,
            4.1732598062797094e-05,
            -8.52993058426425e-05,
            -0.0023450206424385145,
            -0.0014235445338209927,
            -0.00012355763169695304,
            0.000530664633450199,
            0.0005306647227062594,
            0.0011854977724366739,
            0.003296243082864487,
            0.003296243101602789,
            0.0009536668873782492,
            -5.927532073868568e-05,
            -5.927541952387782e-05,
            -0.00010837116800624228,
            0.0023829209908252966,
            0.002382921001774562,
            -0.0008762048228015052,
            -3.790023913302118e-05,
            -0.00025716013081403895,
            1.754259849678628e-05,
            1.7542596687487125e-05,
            0.00019367042291558277,
            -3.790023947523638e-05,
            -0.0003272481812481121,
            -0.0003272482030168279,
            2.841484439487961e-05,
            -0.0003713161497831783,
            -0.00037131611148657156,
            -0.00015934843004396658,
            -0.00015934809063613935,
            -0.00022495011778323045,
            -2.9523359670303098e-05,
            -2.9523360414717888e-05,
            -2.2932755579261792e-05,
            -2.2932759436476477e-05,
            -8.634130779546424e-05,
            -8.634131018386508e-05,
            -0.0016903782633922273,
            -0.0014915001782894518,
            -0.0014915001898725543,
            -0.0003403292448559,
            9.945990605960896e-05,
            -0.00015380754470109277,
            -0.00015380754515907855,
            -0.00010165062254639416,
            -0.000200316207131373,
            4.5730774571059186e-05,
            4.573076985450298e-05,
            -2.2798033408559164e-05,
            -2.2797884110822935e-05,
            5.8203547285469756e-05,
            -0.00010165076194042534,
            -3.409669367084149e-05,
            0.00019542636381644958,
            0.0001954264948706407,
            0.0001093736072676317,
            5.820358310562852e-05,
        ]
    ),
)
beh_dmrg_e = -15.110018770439684


@pytest.mark.parametrize(
    ("molecule", "basis", "symm", "charge", "spin", "tol"),
    [
        (h3_molecule, "sto6g", "C2v", 1, 0, 1e-6),
        (h3_molecule, "631g", "C2v", 1, 0, 1e-6),
        (lih_molecule, "sto6g", "Coov", 2, 0, 1e-6),
        (lih_molecule, "sto6g", "Coov", 0, 0, 1e-6),
        (beh_molecule, "sto6g", "Coov", 1, 0, 1e-6),
    ],
)
def test_rcisd_state_energy(molecule, basis, symm, charge, spin, tol):
    r"""Test that _rcisd_state returns the correct wavefunction, by comparing
    the energy of the method executed by PySCF with that evaluated in PennyLane
    directly as an expectation value for the state returned by the method."""

    mol = pyscf.gto.M(atom=molecule, basis=basis, symmetry=symm, charge=charge, spin=spin)
    myhf = pyscf.scf.RHF(mol).run(verbose=0)
    myci = pyscf.ci.CISD(myhf).run(verbose=0)
    wf_cisd = qchem.convert.import_state(myci, tol=tol)

    core_constant = np.array([mol.energy_nuc()])
    # molecular integrals in AO basis
    one_ao = mol.intor_symmetric("int1e_kin") + mol.intor_symmetric("int1e_nuc")
    two_ao = mol.intor("int2e_sph")
    # rotate to MO basis
    one_mo = np.einsum("pi,pq,qj->ij", myhf.mo_coeff, one_ao, myhf.mo_coeff)
    two_mo = pyscf.ao2mo.incore.full(two_ao, myhf.mo_coeff)
    # physicist ordering convention
    two_mo = np.swapaxes(two_mo, 1, 3)

    h_ferm = qchem.fermionic_observable(core_constant, one_mo, two_mo)
    H = qchem.qubit_observable(h_ferm)
    H_mat = H.sparse_matrix().toarray()
    energy_pl = np.conj(wf_cisd.T).dot(H_mat.dot(wf_cisd))

    assert np.allclose(energy_pl, myci.e_tot, atol=1e-6)


@pytest.mark.parametrize(
    ("molecule", "basis", "symm", "charge", "spin", "tol"),
    [
        (h3_molecule, "sto6g", None, 0, 1, 1e-6),
        (h3_molecule, "631g", None, 0, 1, 1e-6),
        (beh_molecule, "sto6g", "Coov", 0, 1, 1e-6),
    ],
)
def test_ucisd_state_energy(molecule, basis, symm, charge, spin, tol):
    r"""Test that _ucisd_state returns the correct wavefunction, by comparing
    the energy of the method executed by PySCF with that evaluated in PennyLane
    directly as an expectation value for the state returned by the method."""

    mol = pyscf.gto.M(atom=molecule, basis=basis, symmetry=symm, charge=charge, spin=spin)
    # don't use UHF -- leads to spin contamination
    myhf = pyscf.scf.ROHF(mol).run(verbose=0)
    myci = pyscf.ci.UCISD(myhf).run(verbose=0)
    wf_cisd = qchem.convert.import_state(myci, tol=tol)

    core_constant = np.array([mol.energy_nuc()])
    # molecular integrals in AO basis
    one_ao = mol.intor_symmetric("int1e_kin") + mol.intor_symmetric("int1e_nuc")
    two_ao = mol.intor("int2e_sph")
    # rotate to MO basis
    one_mo = np.einsum("pi,pq,qj->ij", myhf.mo_coeff, one_ao, myhf.mo_coeff)
    two_mo = pyscf.ao2mo.incore.full(two_ao, myhf.mo_coeff)
    # physicist ordering convention
    two_mo = np.swapaxes(two_mo, 1, 3)

    h_ferm = qchem.fermionic_observable(core_constant, one_mo, two_mo)
    H = qchem.qubit_observable(h_ferm)
    H_mat = H.sparse_matrix().toarray()
    energy_pl = np.conj(wf_cisd.T).dot(H_mat.dot(wf_cisd))

    assert np.allclose(energy_pl, myci.e_tot, atol=1e-6)


@pytest.mark.parametrize(
    ("molecule", "basis", "symm", "charge", "spin", "tol"),
    [
        (h3_molecule, "sto6g", "C2v", 1, 0, 1e-6),
        (h3_molecule, "631g", "C2v", 1, 0, 1e-6),
        (lih_molecule, "sto6g", "Coov", 2, 0, 1e-6),
        (lih_molecule, "sto6g", "Coov", 0, 0, 1e-6),
        (beh_molecule, "sto6g", "Coov", 1, 0, 1e-6),
    ],
)
def test_rccsd_state_energy(molecule, basis, symm, charge, spin, tol):
    r"""Test that _rccsd_state returns the correct wavefunction, by comparing
    the energy of the method executed by PySCF with that evaluated in PennyLane
    directly as an expectation value for the state returned by the method."""

    mol = pyscf.gto.M(atom=molecule, basis=basis, symmetry=symm, charge=charge, spin=spin)
    myhf = pyscf.scf.RHF(mol).run(verbose=0)
    mycc = pyscf.cc.CCSD(myhf).run(verbose=0)
    wf_ccsd = qchem.convert.import_state(mycc, tol=tol)

    core_constant = np.array([mol.energy_nuc()])
    # molecular integrals in AO basis
    one_ao = mol.intor_symmetric("int1e_kin") + mol.intor_symmetric("int1e_nuc")
    two_ao = mol.intor("int2e_sph")
    # rotate to MO basis
    one_mo = np.einsum("pi,pq,qj->ij", myhf.mo_coeff, one_ao, myhf.mo_coeff)
    two_mo = pyscf.ao2mo.incore.full(two_ao, myhf.mo_coeff)
    # physicist ordering convention
    two_mo = np.swapaxes(two_mo, 1, 3)

    h_ferm = qchem.fermionic_observable(core_constant, one_mo, two_mo)
    H = qchem.qubit_observable(h_ferm)
    H_mat = H.sparse_matrix().toarray()
    energy_pl = np.conj(wf_ccsd.T).dot(H_mat.dot(wf_ccsd))

    assert np.allclose(energy_pl, mycc.e_tot, atol=1e-4)


@pytest.mark.parametrize(
    ("molecule", "basis", "symm", "charge", "spin", "tol"),
    [
        (h3_molecule, "sto6g", None, 0, 1, 1e-6),
        (h3_molecule, "631g", None, 0, 1, 1e-6),
        (beh_molecule, "sto6g", "Coov", 0, 1, 1e-6),
    ],
)
def test_uccsd_state_energy(molecule, basis, symm, charge, spin, tol):
    r"""Test that _uccsd_state returns the correct wavefunction, by comparing
    the energy of the method executed by PySCF with that evaluated in PennyLane
    directly as an expectation value for the state returned by the method."""

    mol = pyscf.gto.M(atom=molecule, basis=basis, symmetry=symm, charge=charge, spin=spin)
    # don't use UHF -- leads to spin contamination
    myhf = pyscf.scf.ROHF(mol).run(verbose=0)
    mycc = pyscf.cc.UCCSD(myhf).run(verbose=0)
    wf_ccsd = qchem.convert.import_state(mycc, tol=tol)

    core_constant = np.array([mol.energy_nuc()])
    # molecular integrals in AO basis
    one_ao = mol.intor_symmetric("int1e_kin") + mol.intor_symmetric("int1e_nuc")
    two_ao = mol.intor("int2e_sph")
    # rotate to MO basis
    one_mo = np.einsum("pi,pq,qj->ij", myhf.mo_coeff, one_ao, myhf.mo_coeff)
    two_mo = pyscf.ao2mo.incore.full(two_ao, myhf.mo_coeff)
    # physicist ordering convention
    two_mo = np.swapaxes(two_mo, 1, 3)

    h_ferm = qchem.fermionic_observable(core_constant, one_mo, two_mo)
    H = qchem.qubit_observable(h_ferm)
    H_mat = H.sparse_matrix().toarray()
    energy_pl = np.conj(wf_ccsd.T).dot(H_mat.dot(wf_ccsd))

    assert np.allclose(energy_pl, mycc.e_tot, atol=1e-2)


@pytest.mark.parametrize(
    ("molecule", "basis", "charge", "spin", "dmrg_dets_coeffs", "dmrg_e", "tol"),
    [
        (h3_molecule, "sto6g", 1, 0, h3p_dmrg_dets_coeffs, h3p_dmrg_e, 1e-6),
        (lih_molecule, "sto6g", 2, 0, lihpp_dmrg_dets_coeffs, lihpp_dmrg_e, 1e-6),
        (h3_molecule, "sto6g", 0, 1, h3_dmrg_dets_coeffs, h3_dmrg_e, 1e-6),
        (lih_molecule, "sto6g", 0, 0, lih_dmrg_dets_coeffs, lih_dmrg_e, 1e-6),
        (beh_molecule, "sto6g", 1, 0, behp_dmrg_dets_coeffs, behp_dmrg_e, 1e-6),
        (beh_molecule, "sto6g", 0, 1, beh_dmrg_dets_coeffs, beh_dmrg_e, 1e-6),
    ],
)
def test_dmrg_state_energy(molecule, basis, charge, spin, dmrg_dets_coeffs, dmrg_e, tol):
    r"""Test that _dmrg_state returns the correct wavefunction, by comparing
    the energy of the method executed by PySCF with that evaluated in PennyLane
    directly as an expectation value for the state returned by the method."""

    wf_dmrg = qchem.convert.import_state(dmrg_dets_coeffs, tol=tol)

    mol = pyscf.gto.M(atom=molecule, basis=basis, charge=charge, spin=spin, symmetry=None)
    myhf = pyscf.scf.ROHF(mol).run(verbose=0)
    core_constant = np.array([mol.energy_nuc()])
    # molecular integrals in AO basis
    one_ao = mol.intor_symmetric("int1e_kin") + mol.intor_symmetric("int1e_nuc")
    two_ao = mol.intor("int2e_sph")
    # rotate to MO basis
    one_mo = np.einsum("pi,pq,qj->ij", myhf.mo_coeff, one_ao, myhf.mo_coeff)
    two_mo = pyscf.ao2mo.incore.full(two_ao, myhf.mo_coeff)
    # physicist ordering convention
    two_mo = np.swapaxes(two_mo, 1, 3)

    h_ferm = qchem.fermionic_observable(core_constant, one_mo, two_mo)
    H = qchem.qubit_observable(h_ferm)
    H_mat = H.sparse_matrix().toarray()
    energy_pl = np.conj(wf_dmrg.T).dot(H_mat.dot(wf_dmrg))

    assert np.allclose(energy_pl, dmrg_e, atol=1e-6)


@pytest.mark.parametrize(
    ("molecule", "basis", "charge", "spin", "shci_dets_coeffs", "shci_e", "tol"),
    [
        (h3_molecule, "sto6g", 1, 0, h3p_shci_dets_coeffs, h3p_shci_e, 1e-6),
        (lih_molecule, "sto6g", 2, 0, lihpp_shci_dets_coeffs, lihpp_shci_e, 1e-6),
        (h3_molecule, "sto6g", 0, 1, h3_shci_dets_coeffs, h3_shci_e, 1e-6),
        (lih_molecule, "sto6g", 0, 0, lih_shci_dets_coeffs, lih_shci_e, 1e-6),
        (beh_molecule, "sto6g", 1, 0, behp_shci_dets_coeffs, behp_shci_e, 1e-6),
        (beh_molecule, "sto6g", 0, 1, beh_shci_dets_coeffs, beh_shci_e, 1e-6),
    ],
)
def test_shci_state_energy(molecule, basis, charge, spin, shci_dets_coeffs, shci_e, tol):
    r"""Test that _uccsd_state returns the correct wavefunction, by comparing
    the energy of the method executed by PySCF with that evaluated in PennyLane
    directly as an expectation value for the state returned by the method."""

    wf_shci = qchem.convert.import_state(shci_dets_coeffs, tol=tol)

    mol = pyscf.gto.M(atom=molecule, basis=basis, charge=charge, spin=spin, symmetry=None)
    myhf = pyscf.scf.ROHF(mol).run(verbose=0)

    core_constant = np.array([mol.energy_nuc()])
    # molecular integrals in AO basis
    one_ao = mol.intor_symmetric("int1e_kin") + mol.intor_symmetric("int1e_nuc")
    two_ao = mol.intor("int2e_sph")
    # rotate to MO basis
    one_mo = np.einsum("pi,pq,qj->ij", myhf.mo_coeff, one_ao, myhf.mo_coeff)
    two_mo = pyscf.ao2mo.incore.full(two_ao, myhf.mo_coeff)
    # physicist ordering convention
    two_mo = np.swapaxes(two_mo, 1, 3)

    h_ferm = qchem.fermionic_observable(core_constant, one_mo, two_mo)
    H = qchem.qubit_observable(h_ferm)
    H_mat = H.sparse_matrix().toarray()
    energy_pl = np.conj(wf_shci.T).dot(H_mat.dot(wf_shci))

    assert np.allclose(energy_pl, shci_e, atol=1e-6)


@pytest.mark.parametrize(
    ("molecule", "basis", "symm", "tol", "wf_ref"),
    [
        (h2_molecule, "sto6g", "d2h", 1e-1, h2_wf_sto6g),
        (h2_molecule, "cc-pvdz", "d2h", 4e-2, h2_wf_ccpvdz),
    ],
)
def test_ucisd_state(molecule, basis, symm, tol, wf_ref):
    r"""Test that _ucisd_state returns the correct wavefunction."""

    mol = pyscf.gto.M(atom=molecule, basis=basis, symmetry=symm)
    myhf = pyscf.scf.UHF(mol).run()
    myci = pyscf.ci.UCISD(myhf).run()

    wf_cisd = qchem.convert._ucisd_state(myci, tol=tol)

    assert wf_cisd.keys() == wf_ref.keys()
    assert np.allclose(abs(np.array(list(wf_cisd.values()))), abs(np.array(list(wf_ref.values()))))


@pytest.mark.parametrize(
    ("molecule", "basis", "symm", "tol", "wf_ref"),
    [
        (h2_molecule, "sto6g", "d2h", 1e-1, h2_wf_sto6g),
        (h2_molecule, "cc-pvdz", "d2h", 4e-2, h2_wf_ccpvdz),
        (
            [["Be", (0, 0, 0)]],
            "sto6g",
            "d2h",
            1e-3,
            {
                (3, 3): 0.9446343496981953,
                (6, 5): 0.003359774446779245,
                (10, 9): 0.003359774446779244,
                (18, 17): 0.003359774446779245,
                (5, 6): 0.003359774446779244,
                (5, 5): -0.18938190575578503,
                (9, 10): 0.003359774446779243,
                (9, 9): -0.18938190575578523,
                (17, 18): 0.003359774446779244,
                (17, 17): -0.18938190575578503,
            },
        ),
    ],
)
def test_rcisd_state(molecule, basis, symm, tol, wf_ref):
    r"""Test that _rcisd_state returns the correct wavefunction."""

    mol = pyscf.gto.M(atom=molecule, basis=basis, symmetry=symm)
    myhf = pyscf.scf.RHF(mol).run()
    myci = pyscf.ci.CISD(myhf).run()

    wf_cisd = qchem.convert._rcisd_state(myci, tol=tol)

    assert wf_cisd.keys() == wf_ref.keys()
    assert np.allclose(abs(np.array(list(wf_cisd.values()))), abs(np.array(list(wf_ref.values()))))


@pytest.mark.parametrize(
    ("molecule", "basis", "symm", "tol", "wf_ref"),
    [
        (h2_molecule, "sto6g", "d2h", 1e-1, h2_wf_sto6g),
        (li2_molecule, "sto6g", "d2h", 1e-1, li2_wf_sto6g),
    ],
)
def test_uccsd_state(molecule, basis, symm, tol, wf_ref):
    r"""Test that _uccsd_state returns the correct wavefunction."""

    mol = pyscf.gto.M(atom=molecule, basis=basis, symmetry=symm)
    myhf = pyscf.scf.UHF(mol).run()
    mycc = pyscf.cc.UCCSD(myhf).run()

    wf_ccsd = qchem.convert._uccsd_state(mycc, tol=tol)

    assert wf_ccsd.keys() == wf_ref.keys()
    assert np.allclose(abs(np.array(list(wf_ccsd.values()))), abs(np.array(list(wf_ref.values()))))


@pytest.mark.parametrize(
    ("molecule", "basis", "symm", "tol", "wf_ref"),
    [
        (h2_molecule, "sto6g", "d2h", 1e-1, h2_wf_sto6g),
        (li2_molecule, "sto6g", "d2h", 1e-1, li2_wf_sto6g),
    ],
)
def test_rccsd_state(molecule, basis, symm, tol, wf_ref):
    r"""Test that _rccsd_state returns the correct wavefunction."""

    mol = pyscf.gto.M(atom=molecule, basis=basis, symmetry=symm)
    myhf = pyscf.scf.RHF(mol).run()
    mycc = pyscf.cc.CCSD(myhf).run()

    wf_ccsd = qchem.convert._rccsd_state(mycc, tol=tol)

    assert wf_ccsd.keys() == wf_ref.keys()
    assert np.allclose(abs(np.array(list(wf_ccsd.values()))), abs(np.array(list(wf_ref.values()))))


@pytest.mark.parametrize(
    ("wavefunction", "state_ref"),
    [
        # h2
        (
            ([[0, 3], [3, 0]], np.array([-0.10660077, 0.9943019])),
            {(2, 2): np.array([-0.10660077]), (1, 1): np.array([0.9943019])},
        ),
        # li2
        (
            (
                [
                    [3, 3, 3, 0, 0, 0, 0, 0, 0, 0],
                    [3, 3, 0, 3, 0, 0, 0, 0, 0, 0],
                    [3, 3, 0, 0, 3, 0, 0, 0, 0, 0],
                    [3, 3, 0, 0, 0, 3, 0, 0, 0, 0],
                ],
                np.array(
                    [-0.887277400314367, 0.308001203411555, 0.307470727263604, 0.145118175734375]
                ),
            ),
            {
                (7, 7): np.array([-0.887277400314367]),
                (11, 11): np.array([0.308001203411555]),
                (19, 19): np.array([0.307470727263604]),
                (35, 35): np.array([0.145118175734375]),
            },
        ),
    ],
)
def test_dmrg_state(wavefunction, state_ref):
    r"""Test that _dmrg_state returns the correct state."""

    state = qml.qchem.convert._dmrg_state(wavefunction)

    assert state == state_ref


@pytest.mark.parametrize(
    ("wavefunction", "state_ref"),
    [
        (
            (
                ["02", "20"],
                np.array([-0.10660077, 0.9943019]),
            ),
            {(2, 2): np.array([-0.10660077]), (1, 1): np.array([0.9943019])},
        ),
        (
            (["02", "ab", "20"], np.array([0.69958765, 0.70211014, 0.1327346])),
            {
                (2, 2): np.array([0.69958765]),
                (1, 2): np.array([0.70211014]),
                (1, 1): np.array([0.1327346]),
            },
        ),
        (
            (
                [
                    "2220000000",
                    "2202000000",
                    "2200200000",
                    "2200020000",
                    "22b00000a0",
                    "22a00000b0",
                ],
                np.array(
                    [
                        0.8874197325,
                        -0.3075732772,
                        -0.3075732772,
                        -0.1450493028,
                        -0.0226602105,
                        -0.0226602105,
                    ]
                ),
            ),
            {
                (7, 7): np.array([-0.8874197325]),
                (11, 11): np.array([0.3075732772]),
                (19, 19): np.array([0.3075732772]),
                (35, 35): np.array([0.1450493028]),
                (259, 7): np.array([-0.0226602105]),
                (7, 259): np.array([0.0226602105]),
            },
        ),
    ],
)
def test_shci_state(wavefunction, state_ref):
    r"""Test that _shci_state returns the correct state."""

    state = qml.qchem.convert._shci_state(wavefunction)

    assert state == state_ref


@pytest.mark.parametrize(
    ("sitevec", "format", "state_ref"),
    [([1, 2, 1, 0, 0, 2], "dmrg", (5, 34)), (["a", "b", "a", "0", "0", "b"], "shci", (5, 34))],
)
def test_sitevec_to_fock(sitevec, format, state_ref):
    r"""Test that _sitevec_to_fock returns the correct state."""

    state = qml.qchem.convert._sitevec_to_fock(sitevec, format)

    assert state == state_ref


@pytest.mark.parametrize(
    ("wf", "norb", "wf_ref"),
    [
        (
            {
                (3, 1): 0.9608586604821351,
                (6, 1): 0.09715813210829484,
                (3, 4): 0.09759050515462506,
                (6, 4): 0.14092548928282997,
                (5, 2): 0.19474863726291963,
            },
            3,
            {
                (3, 1): -0.9608586604821351,
                (6, 1): 0.09715813210829484,
                (3, 4): 0.09759050515462506,
                (6, 4): 0.14092548928282997,
                (5, 2): -0.19474863726291963,
            },
        ),
        (
            {
                (7, 3): 0.989277074123932,
                (38, 3): 0.00045505661217773674,
                (37, 3): -0.0084940777177162,
                (35, 3): -0.009193745907911288,
                (7, 6): -0.0001270090515374941,
                (7, 34): 0.0009164507372497628,
                (7, 5): 0.040547055978034256,
                (7, 33): -0.015939809663167964,
                (14, 10): 0.0014931298751716506,
                (14, 9): -0.0023851394997847376,
                (22, 18): 0.001493129875149863,
                (22, 17): -0.0023851394997471947,
                (38, 6): -0.00020368920130498351,
                (38, 34): 0.0015357755378035945,
                (38, 5): -0.001093202366666422,
                (38, 33): -0.00014202774296372779,
                (13, 10): -0.002433812323079705,
                (13, 9): 0.03686100643494343,
                (21, 18): -0.0024338123230964676,
                (21, 17): 0.03686100643580469,
                (37, 6): -0.000807153059800817,
                (37, 34): 4.586401163325758e-05,
                (37, 5): -0.036419040931801816,
                (37, 33): 0.07370277236189615,
                (11, 10): 0.0033256327616714215,
                (11, 9): -0.06530948644809573,
                (19, 18): 0.00332563276164104,
                (19, 17): -0.06530948627173455,
                (35, 6): 0.0013631853178684785,
                (35, 34): 0.0012748702665335689,
                (35, 5): -0.024439172035092348,
                (35, 33): 0.024769140616833586,
                (7, 36): 0.00028623147075180886,
            },
            5,
            {
                (7, 3): -0.989277074123932,
                (38, 3): -0.00045505661217773674,
                (37, 3): -0.0084940777177162,
                (35, 3): 0.009193745907911288,
                (7, 6): 0.0001270090515374941,
                (7, 34): -0.0009164507372497628,
                (7, 5): 0.040547055978034256,
                (7, 33): -0.015939809663167964,
                (14, 10): 0.0014931298751716506,
                (14, 9): 0.0023851394997847376,
                (22, 18): 0.001493129875149863,
                (22, 17): 0.0023851394997471947,
                (38, 6): 0.00020368920130498351,
                (38, 34): 0.0015357755378035945,
                (38, 5): -0.001093202366666422,
                (38, 33): 0.00014202774296372779,
                (13, 10): -0.002433812323079705,
                (13, 9): 0.03686100643494343,
                (21, 18): -0.0024338123230964676,
                (21, 17): 0.03686100643580469,
                (37, 6): 0.000807153059800817,
                (37, 34): 4.586401163325758e-05,
                (37, 5): 0.036419040931801816,
                (37, 33): 0.07370277236189615,
                (11, 10): -0.0033256327616714215,
                (11, 9): -0.06530948644809573,
                (19, 18): -0.00332563276164104,
                (19, 17): -0.06530948627173455,
                (35, 6): 0.0013631853178684785,
                (35, 34): -0.0012748702665335689,
                (35, 5): 0.024439172035092348,
                (35, 33): 0.024769140616833586,
                (7, 36): 0.00028623147075180886,
            },
        ),
    ],
)
def test_sign_chem_to_phys(wf, norb, wf_ref):
    r"""Test that _sign_chem_to_phys correctly executes the sign convention change."""
    signed_state = qchem.convert._sign_chem_to_phys(wf, norb)

    assert signed_state == wf_ref
