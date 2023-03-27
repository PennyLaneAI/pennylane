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
from os.path import sep as pathsep
from time import sleep
from urllib.parse import quote

import requests
from pennylane.data.dataset import Dataset

S3_URL = "https://xanadu-quantum-datasets.s3.amazonaws.com"
FOLDERMAP_URL = f"{S3_URL}/foldermap.json"
DATA_STRUCT_URL = f"{S3_URL}/data_struct.json"

_foldermap = {}
_data_struct = {}


# pylint:disable=too-many-branches
def _format_details(param, details):
    """Ensures each user-inputted parameter is a properly typed list.
    Also provides custom support for certain parameters."""
    if not isinstance(details, list):
        details = [details]
    if param == "layout":
        # if a user inputs layout=[1,2], they wanted "1x2"
        # note that the above conversion to a list of details wouldn't work as expected here
        if all(isinstance(dim, int) for dim in details):
            return ["x".join(map(str, details))]
        # will turn [(1,2), [3,4], "5x6"] into ["1x2", "3x4", "5x6"]
        for i, detail in enumerate(details):
            if isinstance(detail, Iterable) and all(isinstance(dim, int) for dim in detail):
                details[i] = "x".join(map(str, detail))
            elif not isinstance(detail, str):
                raise TypeError(
                    f"Invalid layout value of '{detail}'. Must be a string or a tuple of ints."
                )
    elif param == "bondlength":
        for i, detail in enumerate(details):
            if isinstance(detail, float):
                details[i] = str(detail)
            elif isinstance(detail, int):
                details[i] = f"{detail:.1f}"
            elif not isinstance(detail, str):
                raise TypeError(f"Invalid bondlength '{detail}'. Must be a string, int or float.")
    for detail in details:
        if not isinstance(detail, str):
            raise TypeError(f"Invalid type '{type(detail).__name__}' for parameter '{param}'")
    return details


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
            exc = None
            if detail == "full":
                if not params_left:
                    return None
                for child in node.values():
                    exc = validate_structure(child, params_left)
            elif detail not in node:  # error: return the error message to be raised
                return ValueError(
                    f"{param} value of '{detail}' is not available. Available values are {list(node)}"
                )
            elif params_left:
                exc = validate_structure(node[detail], params_left)
            if exc is not None:
                return exc
        return None

    exc = validate_structure(_foldermap[data_name], params_needed)
    if isinstance(exc, Exception):
        raise exc  # pylint:disable=raising-bad-type


def _refresh_foldermap():
    """Refresh the foldermap from S3."""
    global _foldermap
    if _foldermap:
        return
    response = requests.get(FOLDERMAP_URL, timeout=5.0)
    response.raise_for_status()
    _foldermap = response.json()


def _refresh_data_struct():
    """Refresh the data struct from S3."""
    global _data_struct
    if _data_struct:
        return
    response = requests.get(DATA_STRUCT_URL, timeout=5.0)
    response.raise_for_status()
    _data_struct = response.json()


def _fetch_and_save(filename, dest_folder):
    """Download a single file from S3 and save it locally."""
    webfile = filename if pathsep == "/" else filename.replace(pathsep, "/")
    response = requests.get(f"{S3_URL}/{quote(webfile)}", timeout=5.0)
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

        prefix = os.path.join(data_name, folder, f"{folder.replace(pathsep, '_')}_")
        # TODO: consider combining files within a folder (switch to append)
        files.extend([f"{prefix}{attr}.dat" for attr in attributes])

    if not force:
        start = len(dest_folder.rstrip(pathsep)) + 1
        existing_files = {
            os.path.join(path, name)[start:]
            for path, _, local_files in os.walk(dest_folder)
            for name in local_files
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
    if folders[0] == ["full"]:
        folders = node
    else:
        values_for_this_node = set(folders[0]).intersection(set(node))
        folders = [f for f in folders[0] if f in values_for_this_node]
    return (
        [
            os.path.join(folder, child)
            for folder in folders
            for child in _generate_folders(node[folder], next_folders)
        ]
        if next_folders
        else folders
    )


def load(
    data_name, attributes=None, lazy=False, folder_path="", force=False, num_threads=50, **params
):
    r"""Downloads the data if it is not already present in the directory and return it to user as a
    :class:`~pennylane.data.Dataset` object. For the full list of available datasets, please see the
    `datasets website <https://pennylane.ai/qml/datasets.html>`_.

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
        list[:class:`~pennylane.data.Dataset`]

    .. warning::

        PennyLane datasets use the ``dill`` module to compress, store, and read data. Since ``dill``
        is built on the ``pickle`` module, we reproduce an important warning from the ``pickle``
        module: it is possible to construct malicious pickle data which will execute arbitrary code
        during unpickling. Never unpickle data that could have come from an untrusted source, or
        that could have been tampered with.
    """

    _ = lazy

    _refresh_foldermap()
    _refresh_data_struct()
    if not attributes:
        attributes = ["full"]

    description = {param: _format_details(param, details) for param, details in params.items()}
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
    docstring = data["docstr"]
    for folder in all_folders:
        real_folder = os.path.join(directory_path, data_name, folder)
        data_files.append(
            Dataset(data_name, real_folder, folder.replace(pathsep, "_"), docstring, standard=True)
        )

    return data_files


def _direc_to_dict(path):
    r"""Helper function to create dictionary structure from directory path"""
    for root, dirs, _ in os.walk(path):
        if not dirs:
            return None
        tree = {x: _direc_to_dict(os.path.join(root, x)) for x in dirs}
        return list(dirs) if all(x is None for x in tree.values()) else tree


def list_datasets(path=None):
    r"""Returns a dictionary of the available datasets.

    Return:
        dict: Nested dictionary representing the directory structure of the hosted datasets.

    **Example:**

    Note that the results of calling this function may differ from this example as more datasets
    are added. For updates on available data see the `datasets website <https://pennylane.ai/qml/datasets.html>`_.

    .. code-block :: pycon

        >>> qml.data.list_datasets()
        {'qchem': {'H2': {'6-31G': ['0.5', '0.54', '0.58', ... '2.02', '2.06', '2.1'],
                          'STO-3G': ['0.5', '0.54', '0.58', ... '2.02', '2.06', '2.1']},
                   'HeH+': {'6-31G': ['0.5', '0.54', '0.58', ... '2.02', '2.06', '2.1'],
                            'STO-3G': ['0.5', '0.54', '0.58', ... '2.02', '2.06', '2.1']},
                   'LiH': {'STO-3G': ['0.5', '0.54', '0.58', ... '2.02', '2.06', '2.1']},
                   'OH-': {'STO-3G': ['0.5', '0.54', '0.58', ... '0.94', '0.98', '1.02']}},
        'qspin': {'Heisenberg': {'closed': {'chain': ['1x16', '1x4', '1x8'],
                                            'rectangular': ['2x2', '2x4', '2x8', '4x4']},
                                 'open': {'chain': ['1x16', '1x4', '1x8'],
                                        'rectangular': ['2x2', '2x4', '2x8', '4x4']}},
                  'Ising': {'closed': {'chain': ['1x16', '1x4', '1x8'],
                                        'rectangular': ['2x2', '2x4', '2x8', '4x4']},
                            'open': {'chain': ['1x16', '1x4', '1x8'],
                                    'rectangular': ['2x2', '2x4', '2x8', '4x4']}}}}
    """

    if path:
        return _direc_to_dict(path)
    _refresh_foldermap()
    return _foldermap.copy()


def list_attributes(data_name):
    r"""List the attributes that exist for a specific ``data_name``.

    Args:
        data_name (str): The type of the desired data

    Returns:
        list (str): A list of accepted attributes for a given data name
    """
    _refresh_data_struct()
    if data_name not in _data_struct:
        raise ValueError(
            f"Currently the hosted datasets are of types: {list(_data_struct)}, but got {data_name}."
        )
    return _data_struct[data_name]["attributes"]


def _interactive_request_attributes(options):
    """Prompt the user to select a list of attributes."""
    prompt = "Please select attributes:"
    for i, option in enumerate(options):
        if option == "full":
            option = "full (all attributes)"
        prompt += f"\n\t{i+1}) {option}"
    print(prompt)
    choices = input(f"Choice (comma-separated list of options) [1-{len(options)}]: ").split(",")
    try:
        choices = list(map(int, choices))
    except ValueError as e:
        raise ValueError(f"Must enter a list of integers between 1 and {len(options)}") from e
    if any(choice < 1 or choice > len(options) for choice in choices):
        raise ValueError(f"Must enter a list of integers between 1 and {len(options)}")
    return [options[choice - 1] for choice in choices]


def _interactive_request_single(node, param):
    """Prompt the user to select a single option from a list."""
    options = list(node)
    if len(options) == 1:
        print(f"Using {options[0]} as it is the only {param} available.")
        sleep(1)
        return options[0]
    print(f"Please select a {param}:")
    print("\n".join(f"\t{i+1}) {option}" for i, option in enumerate(options)))
    try:
        choice = int(input(f"Choice [1-{len(options)}]: "))
    except ValueError as e:
        raise ValueError(f"Must enter an integer between 1 and {len(options)}") from e
    if choice < 1 or choice > len(options):
        raise ValueError(f"Must enter an integer between 1 and {len(options)}")
    return options[choice - 1]


def load_interactive():
    r"""Download a dataset using an interactive load prompt.

    Returns:
        :class:`~pennylane.data.Dataset`

    **Example**

    .. code-block :: pycon

        >>> qml.data.load_interactive()
        Please select a data name:
            1) qspin
            2) qchem
        Choice [1-2]: 1
        Please select a sysname:
            ...
        Please select a periodicity:
            ...
        Please select a lattice:
            ...
        Please select a layout:
            ...
        Please select attributes:
            ...
        Force download files? (Default is no) [y/N]: N
        Folder to download to? (Default is pwd, will download to /datasets subdirectory):

        Please confirm your choices:
        dataset: qspin/Ising/open/rectangular/4x4
        attributes: ['parameters', 'ground_states']
        force: False
        dest folder: /Users/jovyan/Downloads/datasets
        Would you like to continue? (Default is yes) [Y/n]:
        <Dataset = description: qspin/Ising/open/rectangular/4x4, attributes: ['parameters', 'ground_states']>
    """

    _refresh_foldermap()
    _refresh_data_struct()

    node = _foldermap
    data_name = _interactive_request_single(node, "data name")

    description = {}
    value = data_name

    params = _data_struct[data_name]["params"]
    for param in params:
        node = node[value]
        value = _interactive_request_single(node, param)
        description[param] = value

    attributes = _interactive_request_attributes(_data_struct[data_name]["attributes"])
    force = input("Force download files? (Default is no) [y/N]: ") in ["y", "Y"]
    dest_folder = input(
        "Folder to download to? (Default is pwd, will download to /datasets subdirectory): "
    )

    print("\nPlease confirm your choices:")
    print("dataset:", "/".join([data_name] + [description[param] for param in params]))
    print("attributes:", attributes)
    print("force:", force)
    print("dest folder:", os.path.join(dest_folder, "datasets"))

    approve = input("Would you like to continue? (Default is yes) [Y/n]: ")
    if approve not in ["Y", "", "y"]:
        print("Aborting and not downloading!")
        return None
    return load(
        data_name, attributes=attributes, folder_path=dest_folder, force=force, **description
    )[0]
