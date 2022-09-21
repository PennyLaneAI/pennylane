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
Contains the Dataset utility functions.
"""
import os
import sys
import math
import json
import glob
import itertools
import zipfile
import pickle
import dill
import requests

from pennylane.data.dataset import Dataset
#from pennylane.qdata.qchem_dataset import ChemDataset
#from pennylane.qdata.qspin_dataset import SpinDataset

DATA_STRUCT = {
    "qchem": {
        "docstr": "Quantum chemistry dataset.",
        "params": ["molname", "basis", "bondlength"],
        "keys": [
            "molecule",
            "hamiltonian",
            "sparse_hamiltonian",
            "hf_state",
            "meas_groupings",
            "fci_energy",
            "dipole_op",
            "num_op",
            "spin2_op",
            "spinz_op",
            "symmetries",
            "paulix_ops",
            "optimal_sector",
            "ham_wire_map",
            "tapered_hamiltonian",
            "tapered_dipole_op",
            "tapered_num_op",
            "tapered_spin2_op",
            "tapered_spinz_op",
            "tapered_hf_state",
            "vqe_circuit",
            "vqe_params",
            "vqe_energy",
            "full",
        ],
        "docstrings": [
            "Molecule object describing the chemical system",
            "Hamiltonian of the system using Jordan-Wigner mapping",
            "Sparse Hamiltonian of the system",
            "Hartree-Fock state for the system",
            "Qubit-wise commuting measurement groupings",
            "FCI ground state energy",
            "Dipole moment operator of the system",
            "Number operator of the system",
            "Total spin operator of the system",
            "Spin projection operator of the system",
            "Symmetries of the hamiltonian used to taper qubits",
            "Pauli-X operators used in buiding Clifford unitary for tapering",
            "Eigensector containing ground state of the tapered Hamiltonian",
            "Hamiltonian wire map post tapering",
            "Tapered hamiltonian operator",
            "Tapered dipole moment operator",
            "Tapered number operator",
            "Tapered total spin operator",
            "Tapered spin projection operator",
            "Tapered Hartree-Fock state",
            "Variational circuit for obtaining ground-state",
            "Variational parameters for which ansatz evolve to ground state",
            "Variational energy given by the vqe_circuit for provided vqe_params",
            "Contains all the attributes related to the quantum chemistry datatset"
        ]
    },
    "qspin": {
        "docstr": "Quantum many-body spin system dataset.",
        "params": ["sysname", "periodicity", "lattice", "layout"],
        "keys": [
            "parameters",
            "hamiltonians",
            "ground_states",
            "ground_energies",
            "phase_labels",
            "order_parameters",
            "classical_shadows",
            "full",
        ],
        "docstrings" : [
            "Parameters describing the spin system",
            "Hamiltonians for the spin systems with different parameter values",
            "Ground states for the spin systems with different parameter values",
            "Ground state energies for the spin systems with different parameter values",
            "Phase labels for the spin systems with different parameter values",
            "Order parameters required to determine the phase labels",
            "Classical shadow description of the ground state of the spin systems",
            "Contains all the attributes related to the quantum spin datatset",
        ]
    },
}

URL = "https://pl-qd-flask-app.herokuapp.com"


def _convert_size(size_bytes):
    r"""Convert file size for the dataset into appropriate units from bytes

    Args:
        size_bytes(float): size of a file in bytes

    Returns:
        str: size of a file in the closes approximated units

    **Example:**

    .. code-block :: pycon

        >>> _convert_size(1024)
        1 KB

    """

    if not size_bytes:
        return "0 B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    indx = int(math.floor(math.log(size_bytes, 1024)))
    size = round(size_bytes / math.pow(1024, indx), 2)
    return f"{size} {size_name[indx]}"


def _write_prog_bar(progress, completed, barsize, barlength, total_length):
    r"""Helper function for printing progress bar for downloads

    Args:
        progress (float): File size in bytes of the file currently being downloaded
        completed (float): Bar size representing the file currently being downloaded
        barsize (float): Bar size representing the download bar
        barlength (float): Length of the bar printed for showing downloading progress
        total_length (float): Total size in bytes of the file currently being downloaded

    Returns:
        Prints the progressbar to the console
    """
    f = f"[{chr(9608)*barlength} {round(completed, 3)} %{'.'*(barsize-barlength)}] {_convert_size(progress)}/{_convert_size(total_length)}"
    sys.stdout.write("\r" + f)
    sys.stdout.flush()


def _validate_params(data_type, description, attributes=None):
    r"""Validate parameters for loading the data"""

    if data_type not in list(DATA_STRUCT.keys()):
        raise ValueError(
            f"Currently we have data hosted from types: qchem and qspin, but got {data_type}."
        )

    if not isinstance(description, dict):
        raise TypeError(f"Args 'description' should be a dict, but got {type(description)}.")

    if sorted(list(description.keys())) != sorted(DATA_STRUCT[data_type]["params"]):
        raise ValueError(
            f"Supported parameter values for {data_type} are {DATA_STRUCT[data_type]['params']}, but got {list(description.keys())}."
        )

    if attributes is not None and not set(attributes).issubset(
        DATA_STRUCT[data_type]["keys"]
    ):
        raise ValueError(
            f"Supported key values for {data_type} are {DATA_STRUCT[data_type]['keys']}, but got {attributes}."
        )

    if attributes is not None and not isinstance(attributes, list):
        raise TypeError(f"Args 'attributes' should be a list, but got {type(attributes)}.")


def _check_data_exist(data_type, description, directory_path):
    r"""Check if the data has to be redownloaded or not"""
    exist = False
    if "full" in description.values():
        exist = True
    else:
        subdirec_path = [description[param] for param in DATA_STRUCT[data_type]["params"]]
        for subdirec in itertools.product(*subdirec_path):
            path = os.path.join(directory_path, *subdirec)
            if not os.path.exists(path) or not glob.glob(
                os.path.join(path, "**", "*.dat"), recursive=True
            ):
                exist = True
                break
    return exist


def load(data_type, description, attributes=None, lazy=False, folder_path=None, force=True):
    r"""Downloads the data if it is not already present in the directory and return it to user as a Datset object

    Args:
        data_type (str):  A string representing the type of the data required - qchem or qspin
        description (dict): A dictionary with parameters for the required type of data
        attributes (list): An optional list to specify individual data element that are required
        folder_path (str): Path to the root folder where download takes place. By default dataset folder will be created in the working directory.
        force (Bool): Bool representing whether data has to be downloaded even if it is still present

    Returns:
        list[DatasetFile]

    """

    _validate_params(data_type, description, attributes)

    description = {
        key: (val if isinstance(val, list) else [val]) for (key, val) in description.items()
    }

    directory_path = f"datasets/{data_type}"
    if folder_path is not None:
        if folder_path[-1] == "/":
            folder_path = folder_path[:-1]
        directory_path = f"/{folder_path}/{directory_path}"

    if not force:
        force = _check_data_exist(data_type, description, directory_path)

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    with open(f"{directory_path}/data.zip", "wb") as file:
        request_data = {
            "dparams": description,
            "filters": attributes if attributes is not None else ["full"],
        }
        try:
            response = requests.post(f"{URL}/download/{data_type}", json=request_data, stream=True)
            response.raise_for_status()

            print(f"Downloading data to {directory_path}")
            total_length = response.headers.get("Content-Length")
            if total_length is None:
                file.write(response.content)
            else:
                total_length, barsize, progress = int(total_length), 60, 0
                for idx, chunk in enumerate(response.iter_content(chunk_size=4096)):
                    if chunk:
                        file.write(chunk)
                        progress += len(chunk)
                        completed = min(round(progress / total_length, 3) * 100, 100)
                        barlength = int(progress / total_length * barsize)
                        if not idx % 1000:
                            _write_prog_bar(progress, completed, barsize, barlength, total_length)
                _write_prog_bar(progress, completed, barsize, barlength, total_length)
        except requests.exceptions.HTTPError as err:
            os.remove(f"{directory_path}/data.zip")
            raise Exception(f"HTTP Error: {err}") from err
        except requests.exceptions.ConnectionError as err:
            os.remove(f"{directory_path}/data.zip")
            raise Exception(f"Connection Error: {err}") from err
        except requests.exceptions.Timeout as err:
            os.remove(f"{directory_path}/data.zip")
            raise Exception(f"Timeout Error: {err}") from err
        except requests.exceptions.TooManyRedirects as err:
            os.remove(f"{directory_path}/data.zip")
            raise Exception(f"Redirection Error: {err}") from err
        except requests.exceptions.RequestException as err:
            os.remove(f"{directory_path}/data.zip")
            raise Exception(f"Fatal Error: {err}") from err

    data_files = []
    with zipfile.ZipFile(f"{directory_path}/data.zip", "r") as zpf:
        zpf.extractall(f"{directory_path}")
        file_list, file_idx = sorted([x for x in zpf.namelist() if x[-3:] == "dat"]), 0
        dtype_key_len = len(DATA_STRUCT[data_type]["params"])
        while file_idx < len(file_list):
            file = file_list[file_idx]
            cur_name = "_".join(file.split("_")[:dtype_key_len])    
            obj = Dataset(dfile=f"{directory_path}/{file}", dtype=data_type)
            if attributes == ["full"] or attributes is None:
                data = Dataset._read_file(f"{directory_path}/{file}")
                for key, vals in data.items():
                    obj.setattr(key, vals)
                doc_keys, doc_vals = list(data.keys()), list(map(type, data.vals()))
                file_idx += 1
            else:
                key, flag = "_".join(file.split("_")[dtype_key_len:]).split(".")[0], False
                data = Dataset._read_file(f"{directory_path}/{file}")
                doc_keys, doc_vals = [key], [type(data)]
                obj.setattr(key, data)
                for key_idx in range(file_idx+1, len(file_list)):
                    nfile, flag = file_list[key_idx], True
                    nxt_name = "_".join(nfile.split("_")[:dtype_key_len])
                    if cur_name == nxt_name:
                        key = "_".join(nfile.split("_")[dtype_key_len:]).split(".")[0]
                        data = Dataset._read_file(f"{directory_path}/{nfile}")
                        doc_keys.append(key); doc_vals.append(type(data))  
                        obj.setattr(key, data)
                    else:
                        break
                file_idx = key_idx if flag else file_idx+1
            args_idx = [DATA_STRUCT[data_type]["keys"].index(x) for x in doc_keys]
            argsdocs = [DATA_STRUCT[data_type]["docstrings"][x] for x in args_idx]
            obj.setdocstr(DATA_STRUCT[data_type]["docstr"], doc_keys, doc_vals, argsdocs)
            data_files.append(obj)
    os.remove(f"{directory_path}/data.zip")
    return data_files


def _direc_to_dict(path):
    r"""Helper function to create dictionary structure from directory path"""
    for root, dirs, files in os.walk(path):
        if dirs:
            tree = {x: _direc_to_dict(os.path.join(root, x)) for x in dirs}
            vals = [x is None for x in tree.values()]
            if all(vals):
                return list(dirs)
            if any(vals):
                for key, val in tree.items():
                    if val is None:
                        tree.update({key: []})
            return tree
        return None


def list_datasets(folder_path=None):
    r"""Returns a list of datasets and their sizes

    Args:
        folder_path (str): Optional argument for getting datasets descriptor for some local database folder.

    Return:
        dict: Nested dictionary representing the directory structure of the hosted and local databases.

    **Example:**

    .. code-block :: pycon

        >>> qml.qdata.list_datasets()
        {
            'qchem': {
                'H2': {'STO3G': ['0.8']},
                'LiH': {'STO3G': ['1.1']},
                'NH3': {'STO3G': ['1.8']}
            }
        }
    """

    wdata = json.loads(requests.get(URL + "/download/about").content)
    if folder_path is None:
        return wdata
    else:
        fdata = _direc_to_dict(folder_path)
        return wdata, fdata


def _data_dfs(t, path=[]):
    r"""Perform Depth-First search on the nested directory structure"""
    if isinstance(t, dict):
        for key, val in t.items():
            yield from _data_dfs(val, [*path, key])
    else:
        yield path, t


def get_description(data, data_type, **kwargs):
    r"""Help prepare list of `data_param` arguments using nested directory structure

    Args:
        data (dict): Nested dictionary representing the directory structure of the database
        data_type (str): A string representing the type of the data required - qchem or qspin
        **kwargs: Extra arguments used for filtering the data_param based on the required data_type

    Returns:
        list(dict): List of data_param dictionaries matching the criterians provided by the user in kwargs.

    **Example:**

    .. code-block :: pycon

        >>> qml.qdata.get_description(qml.qdata.list_datasets(), "qchem")
        [{'molname': ['full'], 'basis': ['full'], 'bondlength': ['full']}]

    """

    params = DATA_STRUCT[data_type]["params"]
    if not set(kwargs.keys()).issubset(params):
        raise ValueError(
            f"Expected kwargs for the module {module} are {params}, but got {list(kwargs.items())}"
        )

    description = [["full"] for params in params]
    mtch_params = []
    for key, val in kwargs.items():
        description[params.index(key)] = val if isinstance(val, list) else [val]
        mtch_params.append(params.index(key))

    traverse_data = list(
        filter(
            lambda x: all(
                [
                    x[0][m] in description[m]
                    if m < len(params) - 1
                    else set(description[m]).issubset(x[1])
                    for m in mtch_params
                ]
            ),
            _data_dfs(data[data_type], []),
        )
    )

    description = []
    for data in traverse_data:
        dparams = {param: ["full"] for param in params}
        for idx in mtch_params:
            dparams[params[idx]] = [data[0][idx]] if idx < len(params) - 1 else data[1]
        if dparams not in description:
            description.append(dparams)

    return description


def get_attributes(data_type, description):
    r"""Help obtain the `attributes` for given `data_type` and `data_param` from the database

    Args:
        data_type (str):  A string representing the type of the data required - qchem or qspin
        data_param (dict): A dictionary with parameters for the required type of data.

    Returns:
        list[str]: List of strings representing all the filter keys available for the requested dataset
    """

    _validate_params(data_type, description)

    description = {
        key: (val if isinstance(val, list) else [val]) for (key, val) in description.items()
    }

    request_data = {
        "dparams": description,
    }

    response = requests.post(f"{URL}/download/about/{data_type}/keys", json=request_data)
    if response.status_code == 200:
        return json.loads(response.content)
    else:
        response.raise_for_status()
