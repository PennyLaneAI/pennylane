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
import sys

import pytest

import pennylane as qml
from pennylane import fermi

openfermion = pytest.importorskip("openfermion")


# PennyLane operators were obtained from openfermion operators manually
@pytest.mark.parametrize(
    ("of_op", "pl_op"),
    [
        (
            openfermion.FermionOperator("3^ 2", 0.5) + openfermion.FermionOperator("0 2^", 1.0),
            fermi.FermiSentence(
                {
                    fermi.FermiWord({(0, 3): "+", (1, 2): "-"}): 0.5,
                    fermi.FermiWord({(0, 0): "-", (1, 2): "+"}): 1.0,
                }
            ),
        ),
        (
            openfermion.FermionOperator("1^ 2^ 0 3", 1.2)
            + openfermion.FermionOperator("3^ 2^ 1 0", 4.5),
            fermi.FermiSentence(
                {
                    fermi.FermiWord({(0, 1): "+", (1, 2): "+", (2, 0): "-", (3, 3): "-"}): 1.2,
                    fermi.FermiWord({(0, 3): "+", (1, 2): "+", (2, 1): "-", (3, 0): "-"}): 4.5,
                }
            ),
        ),
        (
            openfermion.FermionOperator("2^ 2", 1.5),
            fermi.FermiSentence({fermi.FermiWord({(0, 2): "+", (1, 2): "-"}): 1.5}),
        ),
        (openfermion.FermionOperator("0^ 0"), fermi.FermiWord({(0, 0): "+", (1, 0): "-"})),
        (
            openfermion.FermionOperator("1^ 2", 1.5)
            + openfermion.FermionOperator("2^ 1", 1.5)
            + openfermion.FermionOperator("3^ 2", 0.8),
            fermi.FermiSentence(
                {
                    fermi.FermiWord({(0, 1): "+", (1, 2): "-"}): 1.5,
                    fermi.FermiWord({(0, 2): "+", (1, 1): "-"}): 1.5,
                    fermi.FermiWord({(0, 3): "+", (1, 2): "-"}): 0.8,
                }
            ),
        ),
        (
            openfermion.FermionOperator("3^ 4 5^ 6"),
            fermi.FermiWord({(0, 3): "+", (1, 4): "-", (2, 5): "+", (3, 6): "-"}),
        ),
    ],
)
def test_convert_from_openfermion(of_op, pl_op):
    r"""Test that conversion from openfermion fermionic operator to pennylane
    fermionic operator is correct"""

    converted_op = qml.qchem.from_openfermion(of_op)
    assert converted_op == pl_op


def test_convert_from_openfermion_type_fw():
    r"""Test that FermiWord object is returned when there is one term in the
    fermionic operator with coefficient equal to 1.0"""

    of_op = openfermion.FermionOperator("2^ 3")
    converted_op = qml.qchem.from_openfermion(of_op)

    assert isinstance(converted_op, qml.fermi.FermiWord)


def test_convert_from_openfermion_type_fs():
    r"""Test that FermiSentence object is returned when there are multiple
    terms in the fermionic operator"""

    of_op = openfermion.FermionOperator("2^ 3") + openfermion.FermionOperator("1^ 2")
    converted_op = qml.qchem.from_openfermion(of_op)

    assert isinstance(converted_op, qml.fermi.FermiSentence)


def test_convert_from_openfermion_tol():
    r"""Test that terms with coefficients larger than tolerance are discarded"""

    of_op = (
        openfermion.FermionOperator("2^ 3", 2.0)
        + openfermion.FermionOperator("1^ 2", 3.0)
        + openfermion.FermionOperator("2^ 1", 0.5)
    )
    truncated_op = fermi.FermiSentence(
        {
            fermi.FermiWord({(0, 2): "+", (1, 3): "-"}): 2.0,
            fermi.FermiWord({(0, 1): "+", (1, 2): "-"}): 3.0,
        }
    )

    converted_op = qml.qchem.from_openfermion(of_op, tol=0.6)

    assert converted_op == truncated_op


def test_fail_import_openfermion(monkeypatch):
    """Test if an ImportError is raised when openfermion is requested but not installed"""

    with monkeypatch.context() as m:
        m.setitem(sys.modules, "openfermion", None)

        with pytest.raises(ImportError, match="This feature requires openfermion"):
            qml.qchem.from_openfermion(openfermion.FermionOperator("0^ 1"))
