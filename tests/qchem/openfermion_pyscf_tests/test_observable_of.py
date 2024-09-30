# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

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
Unit tests for the ``observable`` function for openfermion operators.
"""
# pylint: disable=too-many-arguments
import sys

import pytest

import pennylane as qml
from pennylane import qchem

openfermion = pytest.importorskip("openfermion")
openfermionpyscf = pytest.importorskip("openfermionpyscf")

FermionOperator = openfermion.FermionOperator
QubitOperator = openfermion.QubitOperator

t = FermionOperator("0^ 0", 0.5) + FermionOperator("1^ 1", -0.5)

v = (
    FermionOperator("0^ 0^ 0 0", 0.25)
    + FermionOperator("0^ 1^ 1 0", -0.25)
    + FermionOperator("1^ 0^ 0 1", -0.5)
)

v1 = (
    FermionOperator("0^ 0^ 0 0", 0.25)
    + FermionOperator("0^ 1^ 1 0", -0.25)
    + FermionOperator("0^ 2^ 2 0", 0.25)
    + FermionOperator("0^ 3^ 3 0", -0.25)
    + FermionOperator("1^ 0^ 0 1", -0.25)
    + FermionOperator("2^ 0^ 0 2", 0.25)
)

v2 = (
    FermionOperator("0^ 0^ 0 0", 0.5)
    + FermionOperator("0^ 1^ 1 0", -0.25)
    + FermionOperator("0^ 2^ 2 0", 0.5)
    + FermionOperator("0^ 3^ 3 0", -0.25)
    + FermionOperator("1^ 0^ 0 1", -0.25)
    + FermionOperator("2^ 0^ 0 2", -0.25)
)


@pytest.mark.parametrize(
    ("fermion_ops", "init_term", "mapping", "terms_exp"),
    [
        (
            [t, v],
            1 / 4,
            "bravyi_KITAEV",
            {
                (): (0.0625 + 0j),
                ((0, "Z"),): (-0.0625 + 0j),
                ((0, "Z"), (1, "Z")): (0.4375 + 0j),
                ((1, "Z"),): (-0.1875 + 0j),
            },
        ),
        (
            [t, v],
            1 / 4,
            "JORDAN_wigner",
            {
                (): (0.0625 + 0j),
                ((0, "Z"),): (-0.0625 + 0j),
                ((1, "Z"),): (0.4375 + 0j),
                ((0, "Z"), (1, "Z")): (-0.1875 + 0j),
            },
        ),
        (
            [t],
            1 / 2,
            "JORDAN_wigner",
            {(): (0.5 + 0j), ((0, "Z"),): (-0.25 + 0j), ((1, "Z"),): (0.25 + 0j)},
        ),
        (
            [t],
            0,
            "JORDAN_wigner",
            {((0, "Z"),): (-0.25 + 0j), ((1, "Z"),): (0.25 + 0j)},
        ),
        (
            [v1],
            1 / 2,
            "JORDAN_wigner",
            {
                (): (0.4375 + 0j),
                ((1, "Z"),): (0.125 + 0j),
                ((0, "Z"), (1, "Z")): (-0.125 + 0j),
                ((2, "Z"),): (-0.125 + 0j),
                ((0, "Z"), (2, "Z")): (0.125 + 0j),
                ((0, "Z"),): (0.0625 + 0j),
                ((3, "Z"),): (0.0625 + 0j),
                ((0, "Z"), (3, "Z")): (-0.0625 + 0j),
            },
        ),
        (
            [v2],
            1 / 4,
            "bravyi_KITAEV",
            {
                (): (0.125 + 0j),
                ((0, "Z"), (1, "Z")): (0.125 + 0j),
                ((1, "Z"),): (-0.125 + 0j),
                ((2, "Z"),): (-0.0625 + 0j),
                ((0, "Z"), (2, "Z")): (0.0625 + 0j),
                ((1, "Z"), (2, "Z"), (3, "Z")): (0.0625 + 0j),
                ((0, "Z"), (1, "Z"), (2, "Z"), (3, "Z")): (-0.0625 + 0j),
                ((0, "Z"),): (0.125 + 0j),
            },
        ),
    ],
)
def test_observable(fermion_ops, init_term, mapping, terms_exp, custom_wires, monkeypatch):
    r"""Tests the correctness of the 'observable' function used to build many-body observables.

    The parametrized inputs `terms_exp` are `.terms` attribute of the corresponding
    `QubitOperator. The equality checking is implemented in the `qchem` module itself
    as it could be something useful to the users as well.
    """
    # pylint: disable=protected-access

    res_obs = qchem.observable(
        fermion_ops, init_term=init_term, mapping=mapping, wires=custom_wires
    )

    qubit_op = QubitOperator()
    monkeypatch.setattr(qubit_op, "terms", terms_exp)

    assert qml.qchem.convert._openfermion_pennylane_equivalent(
        qubit_op, res_obs, wires=custom_wires
    )


msg1 = "Elements in the lists are expected to be of type 'FermionOperator'"
msg2 = "Please set 'mapping' to 'jordan_wigner', 'parity', or 'bravyi_kitaev'"


@pytest.mark.parametrize(
    ("fermion_ops", "mapping", "msg_match"),
    [
        ([FermionOperator("0^ 0", 0.5), "notFermionOperator"], "JORDAN_wigner", msg1),
        ([FermionOperator("0^ 0", 0.5)], "no_valid_transformation", msg2),
    ],
)
def test_exceptions_observable(fermion_ops, mapping, msg_match):
    """Test that the 'observable' function throws an exception if any element
    in the list 'fermion_ops' is not a FermionOperator objector or if the
    fermionic-to-qubit transformation is not properly defined."""

    with pytest.raises(TypeError, match=msg_match):
        qchem.observable(fermion_ops, mapping=mapping)


def test_import_of(monkeypatch):
    """Test if an ImportError is raised by _import_of function."""
    # pylint: disable=protected-access

    with monkeypatch.context() as m:
        m.setitem(sys.modules, "openfermion", None)

        with pytest.raises(ImportError, match="This feature requires openfermion"):
            qml.qchem.openfermion_pyscf._import_of()

    with monkeypatch.context() as m:
        m.setitem(sys.modules, "openfermionpyscf", None)

        with pytest.raises(ImportError, match="This feature requires openfermion"):
            qml.qchem.openfermion_pyscf._import_of()


def test_import_pyscf(monkeypatch):
    """Test if an ImportError is raised by _import_pyscf function."""
    # pylint: disable=protected-access

    with monkeypatch.context() as m:
        m.setitem(sys.modules, "pyscf", None)

        with pytest.raises(ImportError, match="This feature requires pyscf"):
            qml.qchem.openfermion_pyscf._import_pyscf()
