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
# pylint:disable=too-many-arguments
from concurrent.futures import ThreadPoolExecutor, wait
import os
import requests
from pennylane.data.dataset import Dataset

S3_URL = "https://xanadu-quantum-datasets-test.s3.amazonaws.com"
FOLDERMAP_URL = os.path.join(S3_URL, "foldermap.json")
DATA_STRUCT_URL = os.path.join(S3_URL, "data_struct.json")

_foldermap = {}
_data_struct = {}


def _validate_params(data_type, description, attributes):
    r"""Validate parameters for loading the data"""

    data = _data_struct.get(data_type)
    if not data:
        raise ValueError(
            f"Currently we have data hosted from types: {_data_struct.keys()}, but got {data_type}."
        )

    params_needed = data["params"]
    if set(description) != set(params_needed):
        raise ValueError(
            f"Supported parameter values for {data_type} are {params_needed}, but got {list(description)}."
        )

    # check that the attributes exist in the server or locally
    # description is a dict containing attribute names and values, e.g. 'basis': 'sto-3g', 'bondlength':'0.41'
    tempmap = _foldermap[data_type]
    for key, val in description.items():
        try:
            tempmap = tempmap[val]
            # this will fail at the end when we just want to access the bondlength, e.g tempmap['sto-3g']['0.41'] should be tempmap['sto-eg']
        except:
            if not val in tempmap:
                raise ValueError(
                    f"'{key}' value of '{val}' not available. Available values are {', '.join(tempmap.keys() if isinstance(tempmap, dict) else tempmap)}"
                )

    if not isinstance(attributes, list):
        raise TypeError(f"Arg 'attributes' should be a list, but got {type(attributes)}.")

    all_attributes = data["attributes"]
    if not set(attributes).issubset(set(all_attributes)):
        raise ValueError(
            f"Supported key values for {data_type} are {all_attributes}, but got {attributes}."
        )


def _refresh_foldermap():
    """Refresh the foldermap from S3."""
    response = requests.get(FOLDERMAP_URL, timeout=5.0)
    if not response.ok:
        response.raise_for_status()

    global _foldermap
    _foldermap = response.json()


def _refresh_data_struct():
    """Refresh the data struct from S3."""
    response = requests.get(DATA_STRUCT_URL, timeout=5.0)
    if not response.ok:
        response.raise_for_status()

    global _data_struct
    _data_struct = response.json()


def _fetch_and_save(filename, dest_folder):
    """Download a single file from S3 and save it locally."""
    response = requests.get(os.path.join(S3_URL, filename), timeout=5.0)
    if not response.ok:
        response.raise_for_status()
    with open(os.path.join(dest_folder, filename), "wb") as f:
        f.write(response.content)


def _s3_download(data_type, folders, attributes, dest_folder, force, num_threads):
    """Download a file for each attribute from each folder to the specified destination.

    Args:
        data_type   (str)  : The type of the data required
        folders     (list) : A list of folders corresponding to S3 object prefixes
        attributes  (list) : A list to specify individual data elements that are required
        dest_folder (str)  : Path to the root folder where files should be saved
        force       (bool) : Whether data has to be downloaded even if it is still present
        num_threads (int)  : The maximum number of threads to spawn while downloading files
            (1 thread per file)
    """
    files = []
    for folder in folders:
        local_folder = os.path.join(dest_folder, data_type, folder)
        if not os.path.exists(local_folder):
            os.makedirs(local_folder)

        prefix = os.path.join(data_type, folder, f"{folder.replace('/', '_')}_")
        # TODO: consider combining files within a folder (switch to append)
        files.extend([f"{prefix}{attr}.dat" for attr in attributes])

    if not force:
        start = len(dest_folder.rstrip("/")) + 1
        existing_files = {
            os.path.join(path, name)[start:]
            for path, _, files in os.walk(dest_folder)
            for name in files
        }
        files = list(set(files) - existing_files)

    with ThreadPoolExecutor(num_threads) as pool:
        futures = [pool.submit(_fetch_and_save, f, dest_folder) for f in files]
        wait(futures)


def _generate_folders(node, folders):
    """Recursively generate and return a tree of all folder names below a node.

    Args:
        node (dict) : A sub-dict of the foldermap for which a list of sub-folders is generated
        folders (list[list[str]]) : The ordered list of folder names requested.
            The value ``["full"]`` will expand to all possible folders at that depth

    Returns:
        list[str]: The paths of files that should be fetched from S3
    """

    next_folders = folders[1:]
    folders = set(node) if folders[0] == ["full"] else set(folders[0]).intersection(set(node))
    if not next_folders:
        return folders
    return [
        os.path.join(folder, child)
        for folder in folders
        for child in _generate_folders(node[folder], next_folders)
    ]


def load(
    data_type, attributes=None, lazy=False, folder_path="", force=False, num_threads=50, **params
):
    r"""Downloads the data if it is not already present in the directory and return it to user as a Dataset object

    Args:
        data_type (str)   : A string representing the type of the data required - qchem or qspin
        attributes (list) : An optional list to specify individual data element that are required
        folder_path (str) : Path to the root folder where download takes place.
            By default dataset folder will be created in the working directory
        force (Bool)      : Bool representing whether data has to be downloaded even if it is still present
        num_threads (int) : The maximum number of threads to spawn while downloading files (1 thread per file)
        params (kwargs)   : Keyword arguments exactly matching the parameters required for the data type.
            Note that these are not optional

    Returns:
        list[DatasetFile]
    """

    _ = lazy

    if data_type not in _foldermap:
        _refresh_foldermap()

    if not _data_struct:
        _refresh_data_struct()
    if not attributes:
        attributes = ["full"]
    _validate_params(data_type, params, attributes)
    if len(attributes) > 1 and "full" in attributes:
        attributes = ["full"]

    description = {key: (val if isinstance(val, list) else [val]) for (key, val) in params.items()}
    data = _data_struct[data_type]
    directory_path = os.path.join(folder_path, "datasets")

    folders = [description[param] for param in data["params"]]
    all_folders = _generate_folders(_foldermap[data_type], folders)
    _s3_download(data_type, all_folders, attributes, directory_path, force, num_threads)

    data_files = []
    for folder in all_folders:
        real_folder = os.path.join(directory_path, data_type, folder)
        obj = Dataset(data_type, real_folder, folder.replace("/", "_"), standard=True)
        doc_attrs = obj.list_attributes()
        doc_vals = [type(getattr(obj, attr)) for attr in doc_attrs]
        args_idx = [data["attributes"].index(x) for x in doc_attrs]
        argsdocs = [data["docstrings"][x] for x in args_idx]
        obj.setdocstr(data["docstr"], doc_attrs, doc_vals, argsdocs)
        data_files.append(obj)

    return data_files


def read(filepath):
    """Create a new dataset from a file."""
    dataset = Dataset()
    dataset.read(filepath)
    return dataset


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

    if not _foldermap:
        _refresh_foldermap()
    if folder_path is None:
        return _foldermap
    return _foldermap, _direc_to_dict(folder_path)


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

    if not _data_struct:
        _refresh_data_struct()
    params = _data_struct[data_type]["params"]
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
