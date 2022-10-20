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

_folder_map = {"qchem": {"H2": {"6-31G": ["0.46", "1.16"]}},"qspin": {"Heisenberg": {"closed": {"chain": ["1x4"]}}}}
_data_struct = {"qchem": {"docstr": "Quantum chemistry dataset.", "params": ["molname", "basis", "bondlength"], "docstrings": ["Molecule object describing the chemical system", "Hamiltonian of the system using Jordan-Wigner mapping", "Sparse Hamiltonian of the system", "Hartree-Fock state for the system","Contains all the attributes related to the quantum chemistry datatset"], "attributes":["molecule","hamiltonian", "sparse_hamiltonian", "hf_state", "full"]}, "qspin": {"docstr": "Quantum many-body spin system dataset.", "params": ["sysname", "periodicity", "lattice", "layout"], "docstrings": ["Parameters describing the spin system", "Hamiltonians for the spin systems with different parameter values", "Ground states for the spin systems with different parameter values", "Contains all the attributes related to the quantum spin datatset"], "attributes" : ["parameters", "hamiltonians", "ground_states","full"]}}


@pytest.fixture(scope="session")
def httpserver_listen_address():
    return ("localhost", 8888)

def get_replacement(url,timeout=5.0):
    response = Mock(spec=requests.Response)
    if url == "https://xanadu-quantum-datasets-test.s3.amazonaws.com/foldermap.json":
        response.json.return_value = {"qchem": {"H2": {"6-31G": ["0.46", "1.16"]}},"qspin": {"Heisenberg": {"closed": {"chain": ["1x4"]}}}}
    elif url =="https://xanadu-quantum-datasets-test.s3.amazonaws.com/data_struct.json":
        response.json.return_value ={"qchem": {"docstr": "Quantum chemistry dataset.", "params": ["molname", "basis", "bondlength"], "docstrings": ["Molecule object describing the chemical system", "Hamiltonian of the system using Jordan-Wigner mapping", "Sparse Hamiltonian of the system", "Hartree-Fock state for the system","Contains all the attributes related to the quantum chemistry datatset"], "attributes":["molecule","hamiltonian", "sparse_hamiltonian", "hf_state", "full"]}, "qspin": {"docstr": "Quantum many-body spin system dataset.", "params": ["sysname", "periodicity", "lattice", "layout"], "docstrings": ["Parameters describing the spin system", "Hamiltonians for the spin systems with different parameter values", "Ground states for the spin systems with different parameter values", "Contains all the attributes related to the quantum spin datatset"], "attributes" : ["parameters", "hamiltonians", "ground_states","full"]}}

    response.status_code = 200

    return response

# ('qspn','Currently we have data hosted from types: {_data_struct.keys()}, but got {\'qspn\'}.'),

@pytest.mark.parametrize(
    ('data_type','molname','basis','bondlength','error_message'),
    [
        ('qspn','','','', "Currently we have data hosted from types"),
        ('qchem','wrong','','',"Supported parameter values for qchem are"),
    ],
)
def test_load_error_data_type(data_type,molname, basis, bondlength, error_message):
    with pytest.raises(ValueError, match=error_message):
        qml.data.load(data_type,molname,basis,bondlength)

@pytest.mark.parametrize(
    ('data_type','molname','basis','bondlength','error_message'),
    [
        ('qchem','wrong','','',"'molname' value of 'wrong' not available. Available values are H2"),
        ('qchem','H2','wrong','',"'basis' value of 'wrong' not available. Available values are 6-31G"),
        ('qchem','H2','6-31G','wrong',"'bondlength' value of 'wrong' not available. Available values are [\"0.46\",\"1.16\"]"),
    ],
)
def test_load_error_data_unavailable_qchem(data_type,molname,basis,bondlength,error_message):
    """Tests that the load function returns a list of available data when requesting nonexistent chemistry data"""
    with patch("requests.get", get_replacement):
        with pytest.raises(
                ValueError, match=error_message
            ):
            qml.data.load(data_type,molname=molname,basis=basis,bondlength=bondlength)

@pytest.mark.parametrize(
    ('data_type','sysname','periodicity','lattice','layout','error_message'),
    [
        ('qspin','wrong','','','',"'sysname' value of 'wrong' not available. Available values are Heisenberg"),
        ('qspin','Heisenberg','wrong','','',"'periodicity' value of 'wrong' not available. Available values are closed"),
        ('qspin','Heisenberg','closed','wrong','',"'lattice' value of 'wrong' not available. Available values are chain"),
        ('qspin','Heisenberg','closed','chain','wrong',"'layout' value of 'wrong' not available. Available values are 1x4"),
    ],
    
)
def test_load_error_data_unavailable_qpsin(data_type, sysname, periodicity, lattice,layout,error_message):
    """Tests that the load function returns a list of available data when requesting nonexistent spin data"""
    with patch("requests.get", get_replacement):
        with pytest.raises(
                ValueError, match=error_message
            ):
            qml.data.load(data_type,sysname=sysname,periodicity=periodicity,lattice=lattice,layout=layout)

# @pytest.mark.parametrize(

# )
# def test_load_correct_endpoints():
#     """Test that the endpoints built by the load function are correct"""



# def test_load():
#     """Test that the load function works correctly"""

# def test_load_informative_errors():
#     """Test that the load function gives clear errors"""