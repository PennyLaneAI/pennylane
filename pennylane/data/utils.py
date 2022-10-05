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
import glob
import itertools
import requests

from pennylane.data.dataset import Dataset

# from pennylane.qdata.qchem_dataset import ChemDataset
# from pennylane.qdata.qspin_dataset import SpinDataset

DATA_STRUCT = {
    "qchem": {
        "docstr": "Quantum chemistry dataset.",
        "params": ["molname", "basis", "bondlength"],
        "attributes": [
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
            "Contains all the attributes related to the quantum chemistry datatset",
        ],
    },
    "qspin": {
        "docstr": "Quantum many-body spin system dataset.",
        "params": ["sysname", "periodicity", "lattice", "layout"],
        "attributes": [
            "parameters",
            "hamiltonians",
            "ground_states",
            "ground_energies",
            "phase_labels",
            "order_parameters",
            "classical_shadows",
            "full",
        ],
        "docstrings": [
            "Parameters describing the spin system",
            "Hamiltonians for the spin systems with different parameter values",
            "Ground states for the spin systems with different parameter values",
            "Ground state energies for the spin systems with different parameter values",
            "Phase labels for the spin systems with different parameter values",
            "Order parameters required to determine the phase labels",
            "Classical shadow description of the ground state of the spin systems",
            "Contains all the attributes related to the quantum spin datatset",
        ],
    },
}

URL = "https://pl-qd-flask-app.herokuapp.com"
S3_URL = "https://xanadu-quantum-data.s3.amazonaws.com"
FILEMAP_URL = os.path.join(S3_URL, "filemap.json")

_s3_filemap = {}


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


def _validate_params(data_type, description, attributes):
    r"""Validate parameters for loading the data"""

    data = DATA_STRUCT.get(data_type)
    if not data:
        raise ValueError(
            f"Currently we have data hosted from types: {DATA_STRUCT.keys()}, but got {data_type}."
        )

    params_needed = data["params"]
    if set(description) != set(params_needed):
        raise ValueError(
            f"Supported parameter values for {data_type} are {params_needed}, but got {list(description)}."
        )

    if not isinstance(attributes, list):
        raise TypeError(f"Arg 'attributes' should be a list, but got {type(attributes)}.")

    all_attributes = data["attributes"]
    if not set(attributes).issubset(set(all_attributes)):
        raise ValueError(
            f"Supported key values for {data_type} are {all_attributes}, but got {attributes}."
        )


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


def _s3_download(data_type, folders, attributes, dest_folder):
    """Download a file for each attribute from each folder to the specified destination."""
    for folder in folders:
        s3_folder = os.path.join(S3_URL, data_type, folder)
        local_folder = os.path.join(dest_folder, data_type, folder)
        if not os.path.exists(local_folder):
            os.makedirs(local_folder)

        file_prefix = folder.replace("/", "_")
        # TODO: if len(attributes) > 1, merge contents to single file like _partial.dat
        for attr in attributes:
            fname = f"{file_prefix}_{attr}.dat"
            response = requests.get(f"{s3_folder}/{fname}", timeout=5.0)
            if not response.ok:
                response.raise_for_status()
            with open(f"{local_folder}/{fname}", "wb") as f:
                f.write(response.content)


def _generate_folder_list(node, folders):
    """Recursive helper method for _generate_folders"""
    next_folders = folders[1:]
    folders = set(node) if folders[0] == ["full"] else set(folders[0]).intersection(set(node))
    if not next_folders:
        return folders
    return [
        os.path.join(folder, child)
        for folder in folders
        for child in _generate_folder_list(node[folder], next_folders)
    ]


def _generate_folders(data_type, folders):
    """Recursively generate and return a tree of all folder names matching a pattern.

    If the _s3_filemap is stale, this method will refresh it.

    Args:
        data_type (str): The data type for which folders are generated
        folders: (list[list[str]]): The ordered list of folder names requested.
            The value ``["full"]`` will expand to all possible folders at that depth

    Returns:
        list[str]: The paths of files that should be fetched from S3
    """

    global _s3_filemap
    if data_type not in _s3_filemap:
        response = requests.get(FILEMAP_URL, timeout=5.0)
        if not response.ok:
            response.raise_for_status()
        _s3_filemap = response.json()
    return _generate_folder_list(_s3_filemap[data_type], folders)


def load(data_type, attributes=None, lazy=False, folder_path="", force=False, **params):
    r"""Downloads the data if it is not already present in the directory and return it to user as a Datset object

    Args:
        data_type (str):  A string representing the type of the data required - qchem or qspin
        attributes (list): An optional list to specify individual data element that are required
        folder_path (str): Path to the root folder where download takes place. By default dataset folder will be created in the working directory.
        force (Bool): Bool representing whether data has to be downloaded even if it is still present
        params (kwargs): Keyword arguments exactly matching the parameters required for the data type. Note that these are not optional

    Returns:
        list[DatasetFile]

    """

    _ = lazy
    if not attributes:
        attributes = ["full"]
    _validate_params(data_type, params, attributes)

    description = {key: (val if isinstance(val, list) else [val]) for (key, val) in params.items()}
    data = DATA_STRUCT[data_type]
    directory_path = os.path.join(folder_path, "datasets", data_type)

    if not force:
        force = _check_data_exist(data_type, description, directory_path)

    folders = [description[param] for param in data["params"]]
    all_folders = _generate_folders(data_type, folders)
    _s3_download(data_type, all_folders, attributes, directory_path)

    data_files = []
    for folder in all_folders:
        file_prefix = os.path.join(directory_path, data_type, folder, folder.replace("/", "_"))
        # TODO: replace attributes[0] with actual name after _s3_download() is updated
        fname = f"{file_prefix}_{attributes[0]}.dat"
        obj = Dataset(dfile=fname, dtype=data_type)
        if attributes == ["full"]:
            qdata = Dataset._read_file(fname)
            for key, vals in qdata.items():
                obj.setattr(key, vals)
            doc_attrs, doc_vals = list(qdata.keys()), list(map(type, qdata.values()))
        else:
            doc_attrs, doc_vals = [], []
            for attr in attributes:
                fname = f"{file_prefix}_{attr}.dat"
                qdata = Dataset._read_file(fname)
                doc_attrs.append(attr)
                doc_vals.append(type(qdata))
                obj.setattr(attr, qdata)
        args_idx = [data["attributes"].index(x) for x in doc_attrs]
        argsdocs = [data["docstrings"][x] for x in args_idx]
        obj.setdocstr(data["docstr"], doc_attrs, doc_vals, argsdocs)
        data_files.append(obj)

    return data_files


def _direc_to_dict(path):
    r"""Helper function to create dictionary structure from directory path"""
    for root, dirs, _ in os.walk(path):
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

    wdata = requests.get(URL + "/download/about", timeout=5.0).json()
    if folder_path is None:
        return wdata
    return wdata, _direc_to_dict(folder_path)


def _data_dfs(t, path=None):
    r"""Perform Depth-First search on the nested directory structure"""
    path = path or []
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
    if not set(kwargs).issubset(params):
        raise ValueError(
            f"Expected kwargs for the module {data_type} are {params}, but got {list(kwargs.items())}"
        )

    description = [["full"] for params in params]
    mtch_params = []
    for key, val in kwargs.items():
        description[params.index(key)] = val if isinstance(val, list) else [val]
        mtch_params.append(params.index(key))

    traverse_data = list(
        filter(
            lambda x: all(
                x[0][m] in description[m]
                if m < len(params) - 1
                else set(description[m]).issubset(x[1])
                for m in mtch_params
            ),
            _data_dfs(data[data_type], []),
        )
    )

    description = []
    for tdata in traverse_data:
        dparams = {param: ["full"] for param in params}
        for idx in mtch_params:
            dparams[params[idx]] = [tdata[0][idx]] if idx < len(params) - 1 else tdata[1]
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

    _validate_params(data_type, description, [])

    description = {
        key: (val if isinstance(val, list) else [val]) for (key, val) in description.items()
    }

    request_data = {
        "dparams": description,
    }

    response = requests.post(
        f"{URL}/download/about/{data_type}/keys", json=request_data, timeout=5.0
    )
    if response.status_code != 200:
        response.raise_for_status()
    return response.json()
