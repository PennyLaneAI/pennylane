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
# pylint:disable=too-many-arguments,global-statement
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_EXCEPTION
import os

import requests
from pennylane.data.dataset import Dataset

S3_URL = "https://xanadu-quantum-datasets-test.s3.amazonaws.com"
FOLDERMAP_URL = os.path.join(S3_URL, "foldermap.json")
DATA_STRUCT_URL = os.path.join(S3_URL, "data_struct.json")

_foldermap = {}
_data_struct = {}


def _format_detail(param, detail):
    if isinstance(detail, str):
        return detail
    if param == "layout" and isinstance(detail, Iterable):
        return "x".join(map(str, detail))
    if param == "bondlength":
        if isinstance(detail, float):
            return str(detail)
        if isinstance(detail, int):
            return f"{detail:.1f}"
    raise TypeError(f"Invalid type '{type(detail).__name__}' for parameter '{param}'")


def _validate_params(data_name, description, attributes):
    """Validate parameters for loading the data."""

    data = _data_struct.get(data_name)
    if not data:
        raise ValueError(
            f"Currently the hosted datasets are of types: {list(_data_struct)}, but got {data_name}."
        )

    if not isinstance(attributes, list):
        raise TypeError(f"Arg 'attributes' should be a list, but got {type(attributes).__name__}.")

    all_attributes = data["attributes"]
    if not set(attributes).issubset(set(all_attributes)):
        raise ValueError(
            f"Supported key values for {data_name} are {all_attributes}, but got {attributes}."
        )

    params_needed = data["params"]
    if set(description) != set(params_needed):
        raise ValueError(
            f"Supported parameter values for {data_name} are {params_needed}, but got {list(description)}."
        )

    def validate_structure(node, params_left):
        """Recursively validates that all values in `description` exist in the dataset."""
        param = params_left[0]
        params_left = params_left[1:]
        for detail in description[param]:
            detail = _format_detail(param, detail)
            if detail == "full":
                if not params_left:
                    return
                for child in node.values():
                    validate_structure(child, params_left)
            elif detail not in node:
                # TODO: shorten this limit without permanently changing it
                # sys.tracebacklimit = 0  # the recursive stack is disorienting
                raise ValueError(
                    f"{param} value of '{detail}' is not available. Available values are {list(node)}"
                )
            elif params_left:
                validate_structure(node[detail], params_left)

    validate_structure(_foldermap[data_name], params_needed)


def _refresh_foldermap():
    """Refresh the foldermap from S3."""
    response = requests.get(FOLDERMAP_URL, timeout=5.0)
    response.raise_for_status()

    global _foldermap
    _foldermap = response.json()


def _refresh_data_struct():
    """Refresh the data struct from S3."""
    response = requests.get(DATA_STRUCT_URL, timeout=5.0)
    response.raise_for_status()

    global _data_struct
    _data_struct = response.json()


def _fetch_and_save(filename, dest_folder):
    """Download a single file from S3 and save it locally."""
    response = requests.get(os.path.join(S3_URL, filename), timeout=5.0)
    response.raise_for_status()
    with open(os.path.join(dest_folder, filename), "wb") as f:
        f.write(response.content)


def _s3_download(data_name, folders, attributes, dest_folder, force, num_threads):
    """Download a file for each attribute from each folder to the specified destination.

    Args:
        data_name   (str)  : The type of the data required
        folders     (list) : A list of folders corresponding to S3 object prefixes
        attributes  (list) : A list to specify individual data elements that are required
        dest_folder (str)  : Path to the root folder where files should be saved
        force       (bool) : Whether data has to be downloaded even if it is still present
        num_threads (int)  : The maximum number of threads to spawn while downloading files
            (1 thread per file)
    """
    files = []
    for folder in folders:
        local_folder = os.path.join(dest_folder, data_name, folder)
        if not os.path.exists(local_folder):
            os.makedirs(local_folder)

        prefix = os.path.join(data_name, folder, f"{folder.replace('/', '_')}_")
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
        results = wait(futures, return_when=FIRST_EXCEPTION)
        for result in results.done:
            if result.exception():
                raise result.exception()


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
    data_name, attributes=None, lazy=False, folder_path="", force=False, num_threads=50, **params
):
    r"""Downloads the data if it is not already present in the directory and return it to user as a Dataset object

    Args:
        data_name (str)   : A string representing the type of data required such as `qchem`, `qpsin`, etc.
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

    if data_name not in _foldermap:
        _refresh_foldermap()
    if not _data_struct:
        _refresh_data_struct()
    if not attributes:
        attributes = ["full"]

    description = {key: (val if isinstance(val, list) else [val]) for (key, val) in params.items()}
    _validate_params(data_name, description, attributes)
    if len(attributes) > 1 and "full" in attributes:
        attributes = ["full"]
    for key, val in description.items():
        if len(val) > 1 and "full" in val:
            description[key] = ["full"]

    data = _data_struct[data_name]
    directory_path = os.path.join(folder_path, "datasets")

    folders = [description[param] for param in data["params"]]
    all_folders = _generate_folders(_foldermap[data_name], folders)
    _s3_download(data_name, all_folders, attributes, directory_path, force, num_threads)

    data_files = []
    docstring = f"{data['docstr']}\n\nArgs:\n"
    for arg, argtype, argdoc in zip(
        data["attributes"], data["attribute_types"], data["docstrings"]
    ):
        if arg == "full":
            continue
        docstring += f"\t{arg} ({argtype}): {argdoc}\n"
    docstring += f"\nReturns:\n\tDataset({data_name})"

    for folder in all_folders:
        real_folder = os.path.join(directory_path, data_name, folder)
        data_files.append(
            Dataset(data_name, real_folder, folder.replace("/", "_"), docstring, standard=True)
        )

    return data_files


def _direc_to_dict(path):
    r"""Helper function to create dictionary structure from directory path"""
    for root, dirs, _ in os.walk(path):
        if not dirs:
            return None
        tree = {x: _direc_to_dict(os.path.join(root, x)) for x in dirs}
        vals = [x is None for x in tree.values()]
        if all(vals):
            return list(dirs)
        return tree


def list_datasets(path=None):
    r"""Returns a list of datasets and their sizes

    Return:
        dict: Nested dictionary representing the directory structure of the hosted databases.

    **Example:**

    .. code-block :: pycon

        >>> qml.qdata.list_datasets()
        {
            'qchem': {
                'H2': {
                    '6-31G': ['0.46', '1.16', '0.58'],
                    'STO-3G': ['0.46', '1.05']
                },
                'HeH': {'STO-3G': ['0.9', '0.74', '0.6', '0.8']}
            },
            'qspin': {
                'Heisenberg': {'closed': {'chain': ['1x4']}},
                'Ising': {'open': {'chain': ['1x8']}}
            }
        }
    """

    if path:
        return _direc_to_dict(path)
    if not _foldermap:
        _refresh_foldermap()
    return _foldermap.copy()
