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
Contains functions for querying available datasets and downloading
them.
"""

import typing
import urllib.parse
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from functools import lru_cache
from pathlib import Path
from time import sleep
from typing import List, Optional, Union

from requests import get

from pennylane.data.base import Dataset
from pennylane.data.base.hdf5 import open_hdf5_s3

from .foldermap import DataPath, FolderMapView, ParamArg
from .params import DEFAULT, FULL, format_params

S3_URL = "https://datasets.cloud.pennylane.ai/datasets/h5"
FOLDERMAP_URL = f"{S3_URL}/foldermap.json"
DATA_STRUCT_URL = f"{S3_URL}/data_struct.json"


@lru_cache(maxsize=1)
def _get_foldermap():
    """Fetch the foldermap from S3."""
    response = get(FOLDERMAP_URL, timeout=5.0)
    response.raise_for_status()

    return FolderMapView(response.json())


@lru_cache(maxsize=1)
def _get_data_struct():
    """Fetch the data struct from S3."""
    response = get(DATA_STRUCT_URL, timeout=5.0)
    response.raise_for_status()

    return response.json()


def _download_partial(
    s3_url: str,
    dest: Path,
    attributes: Optional[typing.Iterable[str]],
    overwrite: bool,
    block_size: int,
) -> None:
    """Download the requested attributes of the Dataset at ``s3_path`` into ``dest``.

    Args:
        s3_url: URL of the remote dataset
        dest: Destination dataset path
        attributes: Requested attributes to download. Passing ``None`` is equivalent
            to requesting all attributes of the remote dataset.
        overwrite: If True, overwrite attributes that already exist at ``dest``. Otherwise,
            only download attributes that do not exist at ``dest``.
    """

    dest_dataset = Dataset.open(dest, mode="a")
    remote_dataset = None

    attributes_to_fetch = set()

    if attributes is not None:
        attributes_to_fetch.update(attributes)
    else:
        remote_dataset = Dataset(open_hdf5_s3(s3_url, block_size=block_size))
        attributes_to_fetch.update(remote_dataset.attrs)

    if not overwrite:
        attributes_to_fetch.difference_update(dest_dataset.attrs)

    if len(attributes_to_fetch) > 0:
        remote_dataset = remote_dataset or Dataset(open_hdf5_s3(s3_url, block_size=block_size))
        remote_dataset.write(dest_dataset, "a", attributes, overwrite=overwrite)

    if remote_dataset:
        remote_dataset.close()

    dest_dataset.close()
    del remote_dataset
    del dest_dataset


def _download_full(s3_url: str, dest: Path):
    """Download the full dataset file at ``s3_url`` to ``path``."""

    with open(dest, "wb") as f:
        resp = get(s3_url, timeout=5.0)
        resp.raise_for_status()

        f.write(resp.content)


def _download_dataset(
    data_path: DataPath,
    dest: Path,
    attributes: Optional[typing.Iterable[str]],
    block_size: int,
    force: bool = False,
) -> None:
    """Downloads the dataset at ``data_path`` to ``dest``, optionally downloading
    only requested attributes. If ``attributes`` is not provided, every attribute
    will be requested.

    If any of the attributes of the remote dataset are already downloaded locally,
    they will not be overwritten unless ``force`` is True.
    """

    # URL-escape special characters like '+', '$', and '%' in the data path
    url_safe_datapath = urllib.parse.quote(str(data_path))
    s3_url = f"{S3_URL}/{url_safe_datapath}"

    if attributes is not None or dest.exists():
        _download_partial(
            s3_url, dest=dest, attributes=attributes, overwrite=force, block_size=block_size
        )
    else:
        _download_full(s3_url, dest=dest)


def _validate_attributes(data_struct: dict, data_name: str, attributes: typing.Iterable[str]):
    """Checks that ``attributes`` contains only valid attributes for the given
    ``data_name``. If any attributes do not exist, raise a ValueError."""
    invalid_attributes = [
        attr for attr in attributes if attr not in data_struct[data_name]["attributes"]
    ]
    if not invalid_attributes:
        return

    if len(invalid_attributes) == 1:
        values_err = f"'{invalid_attributes[0]}' is an invalid attribute for '{data_name}'"
    else:
        values_err = f"{invalid_attributes} are invalid attributes for '{data_name}'"

    raise ValueError(f"{values_err}. Valid attributes are: {data_struct[data_name]['attributes']}")


def load(  # pylint: disable=too-many-arguments
    data_name: str,
    attributes: Optional[typing.Iterable[str]] = None,
    folder_path: Path = Path("./datasets/"),
    force: bool = False,
    num_threads: int = 50,
    block_size: int = 8388608,
    **params: Union[ParamArg, str, List[str]],
):
    r"""Downloads the data if it is not already present in the directory and returns it as a list of
    :class:`~pennylane.data.Dataset` objects. For the full list of available datasets, please see
    the `datasets website <https://pennylane.ai/datasets>`_.

    Args:
        data_name (str)   : A string representing the type of data required such as `qchem`, `qpsin`, etc.
        attributes (list[str]) : An optional list to specify individual data element that are required
        folder_path (str) : Path to the directory used for saving datasets. Defaults to './datasets'
        force (Bool)      : Bool representing whether data has to be downloaded even if it is still present
        num_threads (int) : The maximum number of threads to spawn while downloading files (1 thread per file)
        block_size (int)  : The number of bytes to fetch per read operation when fetching datasets from S3.
            Larger values may improve performance for large datasets, but will slow down small reads. Defaults
            to 8MB
        params (kwargs)   : Keyword arguments exactly matching the parameters required for the data type.
            Note that these are not optional

    Returns:
        list[:class:`~pennylane.data.Dataset`]

    .. seealso:: :func:`~.load_interactive`, :func:`~.list_attributes`, :func:`~.list_datasets`.

    **Example**

    The :func:`~pennylane.data.load` function returns a ``list`` with the desired data.

    >>> H2datasets = qml.data.load("qchem", molname="H2", basis="STO-3G", bondlength=1.1)
    >>> print(H2datasets)
    [<Dataset = molname: H2, basis: STO-3G, bondlength: 1.1, attributes: ['basis', 'basis_rot_groupings', ...]>]

    .. note::

        If not otherwise specified, ``qml.data.load`` will download the
        default parameter value specified by the dataset.

        The default values for attributes are as follows:

        - Molecules: ``basis`` is the smallest available basis, usually ``"STO-3G"``, and ``bondlength`` is the optimal bondlength for the molecule or an alternative if the optimal is not known.

        - Spin systems: ``periodicity`` is ``"open"``, ``lattice`` is ``"chain"``, and ``layout`` is ``1x4`` for ``chain`` systems and ``2x2`` for ``rectangular`` systems.

    We can load datasets for multiple parameter values by providing a list of values instead of a single value.
    To load all possible values, use the special value :const:`~pennylane.data.FULL` or the string 'full':

    >>> H2datasets = qml.data.load("qchem", molname="H2", basis="full", bondlength=[0.5, 1.1])
    >>> print(H2datasets)
    [<Dataset = molname: H2, basis: STO-3G, bondlength: 0.5, attributes: ['basis', 'basis_rot_groupings', ...]>,
        <Dataset = molname: H2, basis: STO-3G, bondlength: 1.1, attributes: ['basis', 'basis_rot_groupings', ...]>,
        <Dataset = molname: H2, basis: CC-PVDZ, bondlength: 0.5, attributes: ['basis', 'basis_rot_groupings', ...]>,
        <Dataset = molname: H2, basis: CC-PVDZ, bondlength: 1.1, attributes: ['basis', 'basis_rot_groupings', ...]>,
        <Dataset = molname: H2, basis: 6-31G, bondlength: 0.5, attributes: ['basis', 'basis_rot_groupings', ...]>,
        <Dataset = molname: H2, basis: 6-31G, bondlength: 1.1, attributes: ['basis', 'basis_rot_groupings', ...]>]

    When we only want to download portions of a large dataset, we can specify
    the desired properties  (referred to as 'attributes'). For example, we
    can download or load only the molecule and energy of a dataset as
    follows:

    >>> part = qml.data.load(
    ...     "qchem",
    ...     molname="H2",
    ...     basis="STO-3G",
    ...     bondlength=1.1,
    ...     attributes=["molecule", "fci_energy"])[0]
    >>> part.molecule
    <Molecule = H2, Charge: 0, Basis: STO-3G, Orbitals: 2, Electrons: 2>

    To determine what attributes are available, please see :func:`~.list_attributes`.

    The loaded data items are fully compatible with PennyLane. We can
    therefore use them directly in a PennyLane circuit as follows:

    >>> H2data = qml.data.load("qchem", molname="H2", basis="STO-3G", bondlength=1.1)[0]
    >>> dev = qml.device("default.qubit",wires=4)
    >>> @qml.qnode(dev)
    ... def circuit():
    ...     qml.BasisState(H2data.hf_state, wires = [0, 1, 2, 3])
    ...     for op in H2data.vqe_gates:
    ...         qml.apply(op)
    ...     return qml.expval(H2data.hamiltonian)
    >>> print(circuit())
    -1.0791430411076344
    """
    foldermap = _get_foldermap()
    data_struct = _get_data_struct()

    params = format_params(**params)

    if attributes:
        _validate_attributes(data_struct, data_name, attributes)

    folder_path = Path(folder_path)

    data_paths = [data_path for _, data_path in foldermap.find(data_name, **params)]

    dest_paths = [folder_path / data_path for data_path in data_paths]

    for path_parents in set(path.parent for path in dest_paths):
        path_parents.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(min(num_threads, len(dest_paths))) as pool:
        futures = [
            pool.submit(
                _download_dataset,
                data_path,
                dest_path,
                attributes,
                force=force,
                block_size=block_size,
            )
            for data_path, dest_path in zip(data_paths, dest_paths)
        ]
        results = wait(futures, return_when=FIRST_EXCEPTION)
        for result in results.done:
            if result.exception() is not None:
                raise result.exception()

    return [Dataset.open(Path(dest_path), "a") for dest_path in dest_paths]


def list_datasets() -> dict:
    r"""Returns a dictionary of the available datasets.

    Return:
        dict: Nested dictionary representing the directory structure of the hosted datasets.

    .. seealso:: :func:`~.load_interactive`, :func:`~.list_attributes`, :func:`~.load`.

    **Example:**

    Note that the results of calling this function may differ from this example as more datasets
    are added. For updates on available data see the `datasets website <https://pennylane.ai/datasets>`_.

    >>> available_data = qml.data.list_datasets()
    >>> available_data.keys()
    dict_keys(["qspin", "qchem"])
    >>> available_data["qchem"].keys()
    dict_keys(["H2", "LiH", ...])
    >>> available_data['qchem']['H2'].keys()
    dict_keys(["CC-PVDZ", "6-31G", "STO-3G"])
    >>> print(available_data['qchem']['H2']['STO-3G'])
    ["0.5", "0.54", "0.62", "0.66", ...]

    Note that this example limits the results of the function calls for
    clarity and that as more data becomes available, the results of these
    function calls will change.
    """

    def remove_paths(foldermap):
        """Copies the foldermap, converting the bottom-level mapping of parameters
        to Paths to a list of the parameters."""
        value = next(iter(foldermap.values()))

        if not isinstance(value, typing.Mapping):
            return sorted(foldermap.keys())

        return {param: remove_paths(foldermap[param]) for param in foldermap.keys()}

    return remove_paths(_get_foldermap())


def list_attributes(data_name):
    r"""List the attributes that exist for a specific ``data_name``.

    Args:
        data_name (str): The type of the desired data

    Returns:
        list (str): A list of accepted attributes for a given data name

    .. seealso:: :func:`~.load_interactive`, :func:`~.list_datasets`, :func:`~.load`.

    **Example**

    >>> qml.data.list_attributes(data_name="qchem")
    ['molname',
     'basis',
     'bondlength',
     ...
     'vqe_params',
     'vqe_energy']
    """
    data_struct = _get_data_struct()
    if data_name not in data_struct:
        raise ValueError(
            f"Currently the hosted datasets are of types: {list(data_struct)}, but got {data_name}."
        )
    return data_struct[data_name]["attributes"]


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

    .. seealso:: :func:`~.load`, :func:`~.list_attributes`, :func:`~.list_datasets`.

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

    foldermap = _get_foldermap()
    data_struct = _get_data_struct()

    node = foldermap
    data_name = _interactive_request_single(node, "data name")

    description = {}
    value = data_name

    params = data_struct[data_name]["params"]
    for param in params:
        node = node[value]
        value = _interactive_request_single(node, param)
        description[param] = value

    attributes = _interactive_request_attributes(
        [attribute for attribute in data_struct[data_name]["attributes"] if attribute not in params]
    )
    force = input("Force download files? (Default is no) [y/N]: ") in ["y", "Y"]
    dest_folder = Path(
        input("Folder to download to? (Default is pwd, will download to /datasets subdirectory): ")
    )

    print("\nPlease confirm your choices:")
    print("dataset:", "/".join([data_name] + [description[param] for param in params]))
    print("attributes:", attributes)
    print("force:", force)
    print("dest folder:", dest_folder / "datasets")

    approve = input("Would you like to continue? (Default is yes) [Y/n]: ")
    if approve not in ["Y", "", "y"]:
        print("Aborting and not downloading!")
        return None

    return load(
        data_name, attributes=attributes, folder_path=dest_folder, force=force, **description
    )[0]


__all__ = (
    "load",
    "load_interactive",
    "list_datasets",
    "list_attributes",
    "FULL",
    "DEFAULT",
    "ParamArg",
)
