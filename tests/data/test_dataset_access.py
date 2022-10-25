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


def get_mock(url, timeout=1.0):
    return


@patch.object(qml.data.data_manager, "_foldermap", _folder_map)
@patch.object(qml.data.data_manager, "_data_struct", _data_struct)
@patch.object(requests, "get", get_mock)
class TestValidateParams:
    def test_data_type_error(self):
        """Test that _validate_params fails when an unknown data_type is passed in."""
        with pytest.raises(ValueError, match="Currently we have data hosted from types"):
            qml.data.data_manager._validate_params("qspn", {}, [])

    @pytest.mark.parametrize(
        ("description", "error_message"),
        [
            (
                {"molname": ["H2"], "basis": ["6-31G"]},
                r"Supported parameter values for qchem are \['molname', 'basis', 'bondlength'\], but got \['molname', 'basis'\].",
            ),
            (
                {
                    "molname": ["H2"],
                    "basis": ["6-31G"],
                    "bondlength": ["0.46"],
                    "unexpected": ["foo"],
                },
                r"Supported parameter values for qchem are \['molname', 'basis', 'bondlength'\], but got \['molname', 'basis', 'bondlength', 'unexpected'\].",
            ),
        ],
    )
    def test_incorrect_set_of_params(self, description, error_message):
        """Test that _validate_params fails when the kwargs do not exactly match the dataset."""
        with pytest.raises(ValueError, match=error_message):
            qml.data.data_manager._validate_params("qchem", description, [])

    @pytest.mark.parametrize(
        ("description", "error_message"),
        [
            (
                {"molname": ["foo"], "basis": ["6-31G"], "bondlength": ["0.46"]},
                r"molname value of 'foo' not available. Available values are \['H2'\]",
            ),
            (
                {"molname": ["H2"], "basis": ["foo"], "bondlength": ["0.46"]},
                r"basis value of 'foo' not available. Available values are \['6-31G'\]",
            ),
            (
                {"molname": ["H2"], "basis": ["6-31G"], "bondlength": ["foo"]},
                r"bondlength value of 'foo' not available. Available values are \['0.46', '1.16'\]",
            ),
            (
                {"molname": ["H2"], "basis": ["6-31G", "foo"], "bondlength": ["0.46"]},
                r"basis value of 'foo' not available. Available values are \['6-31G'\]",
            ),
        ],
    )
    def test_incorrect_param_values(self, description, error_message):
        """Test that _validate_params fails when an unrecognized parameter value is given."""
        with pytest.raises(ValueError, match=error_message):
            qml.data.data_manager._validate_params("qchem", description, [])

    @pytest.mark.parametrize(
        ("attributes", "error_type", "error_message"),
        [
            (None, TypeError, "Arg 'attributes' should be a list, but got NoneType"),
            (
                ["molecule", "full", "foo"],
                ValueError,
                r"Supported key values for qchem are \['molecule', 'hamiltonian', 'sparse_hamiltonian', 'hf_state', 'full'], but got \['molecule', 'full', 'foo'\].",
            ),
            (
                ["foo"],
                ValueError,
                r"Supported key values for qchem are \['molecule', 'hamiltonian', 'sparse_hamiltonian', 'hf_state', 'full'\], but got \['foo'\].",
            ),
        ],
    )
    def test_attributes_must_be_list(self, attributes, error_type, error_message):
        with pytest.raises(error_type, match=error_message):
            qml.data.data_manager._validate_params(
                "qchem",
                {"molname": ["H2"], "basis": ["6-31G"], "bondlength": ["0.46"]},
                attributes,
            )

    @pytest.mark.parametrize(
        ("data_type", "description", "attributes"),
        [
            (
                "qchem",
                {"molname": ["H2"], "basis": ["6-31G"], "bondlength": ["0.46"]},
                ["full"],
            ),
            (
                "qchem",
                {"molname": ["full"], "basis": ["6-31G"], "bondlength": ["full"]},
                ["molecule", "hamiltonian", "sparse_hamiltonian"],
            ),
            (
                "qspin",
                {
                    "sysname": ["Heisenberg"],
                    "periodicity": ["closed"],
                    "lattice": ["chain"],
                    "layout": ["1x4"],
                },
                ["full", "ground_states"],
            ),
            (
                "qspin",
                {
                    "sysname": ["full"],
                    "periodicity": ["closed"],
                    "lattice": ["full"],
                    "layout": ["full"],
                },
                ["full", "ground_states"],
            ),
        ],
    )
    def test_validate_params_successes(self, data_type, description, attributes):
        """Test that the _validate_params method passes with valid parameters."""
        qml.data.data_manager._validate_params(data_type, description, attributes)
