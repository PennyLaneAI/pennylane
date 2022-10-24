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
Unit tests for the :class:`pennylane.data.Dataset` class and its functions.
"""
import json
from multiprocessing.sharedctypes import Value

from pytest_mock import mocker
import pennylane as qml
import dill
import zstd
from unittest.mock import MagicMock, Mock, patch, mock_open
import pytest
import requests

from urllib.request import urlopen

_folder_map = {
    "qchem": {"H2": {"6-31G": ["0.46", "1.16"]}},
    "qspin": {"Heisenberg": {"closed": {"chain": ["1x4"]}}},
}
_data_struct = {
    "qchem": {
        "docstr": "Quantum chemistry dataset.",
        "params": ["molname", "basis", "bondlength"],
        "docstrings": [
            "Molecule object describing the chemical system",
            "Hamiltonian of the system using Jordan-Wigner mapping",
            "Sparse Hamiltonian of the system",
            "Hartree-Fock state for the system",
            "Contains all the attributes related to the quantum chemistry datatset",
        ],
        "attributes": ["molecule", "hamiltonian", "sparse_hamiltonian", "hf_state", "full"],
    },
    "qspin": {
        "docstr": "Quantum many-body spin system dataset.",
        "params": ["sysname", "periodicity", "lattice", "layout"],
        "docstrings": [
            "Parameters describing the spin system",
            "Hamiltonians for the spin systems with different parameter values",
            "Ground states for the spin systems with different parameter values",
            "Contains all the attributes related to the quantum spin datatset",
        ],
        "attributes": ["parameters", "hamiltonians", "ground_states", "full"],
    },
}


@pytest.fixture(scope="session")
def httpserver_listen_address():
    return ("localhost", 8888)


# ('qspn','Currently we have data hosted from types: {_data_struct.keys()}, but got {\'qspn\'}.'),


@patch.object(qml.data.s3, "_foldermap", _folder_map)
@patch.object(qml.data.s3, "_data_struct", _data_struct)
class TestLoad:
    @pytest.mark.parametrize(
        ("data_type", "molname", "basis", "bondlength", "error_message"),
        [
            ("qspn", "", "", "", "Currently we have data hosted from types"),
            ("qchem", "wrong", "", "", "Supported parameter values for qchem are"),
        ],
    )
    def test_load_error_data_type(self, data_type, molname, basis, bondlength, error_message):
        with pytest.raises(ValueError, match=error_message):
            qml.data.load(data_type, molname, basis, bondlength)

    @pytest.mark.parametrize(
        ("data_type", "molname", "basis", "bondlength", "error_message"),
        [
            (
                "qchem",
                "wrong",
                "",
                "",
                r"molname value of 'wrong' not available. Available values are \['H2'\]",
            ),
            (
                "qchem",
                "H2",
                "wrong",
                "",
                r"basis value of 'wrong' not available. Available values are \['6-31G'\]",
            ),
            (
                "qchem",
                "H2",
                "6-31G",
                "wrong",
                r"bondlength value of 'wrong' not available. Available values are \['0.46', '1.16'\]",
            ),
        ],
    )
    def test_load_error_data_unavailable_qchem(
        self, data_type, molname, basis, bondlength, error_message
    ):
        """Tests that the load function returns a list of available data when requesting nonexistent chemistry data"""
        with pytest.raises(ValueError, match=error_message):
            qml.data.load(data_type, molname=molname, basis=basis, bondlength=bondlength)

    @pytest.mark.parametrize(
        ("data_type", "sysname", "periodicity", "lattice", "layout", "error_message"),
        [
            (
                "qspin",
                "wrong",
                "",
                "",
                "",
                r"sysname value of 'wrong' not available. Available values are \['Heisenberg'\]",
            ),
            (
                "qspin",
                "Heisenberg",
                "wrong",
                "",
                "",
                r"periodicity value of 'wrong' not available. Available values are \['closed'\]",
            ),
            (
                "qspin",
                "Heisenberg",
                "closed",
                "wrong",
                "",
                r"lattice value of 'wrong' not available. Available values are \['chain'\]",
            ),
            (
                "qspin",
                "Heisenberg",
                "closed",
                "chain",
                "wrong",
                r"layout value of 'wrong' not available. Available values are \['1x4'\]",
            ),
        ],
    )
    def test_load_error_data_unavailable_qpsin(
        self, data_type, sysname, periodicity, lattice, layout, error_message
    ):
        """Tests that the load function returns a list of available data when requesting nonexistent spin data"""
        with pytest.raises(ValueError, match=error_message):
            qml.data.load(
                data_type, sysname=sysname, periodicity=periodicity, lattice=lattice, layout=layout
            )


# @pytest.mark.parametrize(

# )
# def test_load_correct_endpoints():
#     """Test that the endpoints built by the load function are correct"""


# def test_load():
#     """Test that the load function works correctly"""

# def test_load_informative_errors():
#     """Test that the load function gives clear errors"""
