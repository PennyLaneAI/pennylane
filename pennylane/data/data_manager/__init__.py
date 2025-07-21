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

import sys
import urllib.parse
from concurrent import futures
from functools import lru_cache
from pathlib import Path
from time import sleep
from typing import Any, Optional
from collections.abc import Iterable, Mapping

from requests import get, head

from pennylane.data.base import Dataset
from pennylane.data.base.hdf5 import open_hdf5_s3
from pennylane.data.data_manager import progress

from .graphql import (
    get_dataset_urls,
    _get_parameter_tree,
    list_data_names,
    list_attributes,
)
from .foldermap import DataPath, FolderMapView, ParamArg
from .params import DEFAULT, FULL, format_params, provide_defaults


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


S3_URL = "https://datasets.cloud.pennylane.ai/datasets/h5"


def _download_partial(  # pylint: disable=too-many-arguments
    s3_url: str,
    dest: Path,
    attributes: Iterable[str] | None,
    overwrite: bool,
    block_size: int,
    pbar_task: progress.Task | None,
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

    if pbar_task:
        file_size = dest.stat().st_size
        pbar_task.update(completed=file_size, total=file_size)


def _download_full(s3_url: str, dest: Path, block_size: int, pbar_task: progress.Task | None):
    """Download the full dataset file at ``s3_url`` to ``path``."""
    resp = get(s3_url, timeout=5.0, stream=True)
    resp.raise_for_status()

    with open(dest, "wb") as f:
        if pbar_task is not None:
            for block in resp.iter_content(chunk_size=block_size):
                f.write(block)
                pbar_task.update(advance=len(block))
        else:
            for block in resp.iter_content(chunk_size=block_size):
                f.write(block)


def _download_dataset(  # pylint: disable=too-many-arguments
    dataset_url: str,
    dest: Path,
    attributes: Iterable[str] | None,
    block_size: int,
    force: bool,
    pbar_task: progress.Task | None,
) -> None:
    """Downloads the dataset at ``dataset_url`` to ``dest``, optionally downloading
    only requested attributes. If ``attributes`` is not provided, every attribute
    will be requested.

    If any of the attributes of the remote dataset are already downloaded locally,
    they will not be overwritten unless ``force`` is True.

    If ``pbar_task`` is provided, it will be updated with the download progress.
    """

    if attributes is not None or dest.exists():
        _download_partial(
            dataset_url,
            dest=dest,
            attributes=attributes,
            overwrite=force,
            block_size=block_size,
            pbar_task=pbar_task,
        )
    else:
        _download_full(dataset_url, dest=dest, block_size=block_size, pbar_task=pbar_task)


def _download_datasets(  # pylint: disable=too-many-arguments
    data_name: str,
    folder_path: Path,
    dataset_urls: list[str],
    dataset_ids: list[str],
    attributes: Iterable[str] | None,
    force: bool,
    block_size: int,
    num_threads: int,
    pbar: progress.Progress | None,
) -> list[Path]:
    """Downloads the datasets with given ``dataset_urls`` to ``folder_path``.

    If ``pbar`` is provided, a progress task will be added for each requested dataset.

    Returns:
        list[Path]: List of downloaded dataset paths
    """
    file_names = [dataset_id + ".h5" for dataset_id in dataset_ids]
    dest_paths = [folder_path / data_name / data_id for data_id in file_names]

    for path_parents in {path.parent for path in dest_paths}:
        path_parents.mkdir(parents=True, exist_ok=True)

    if pbar is not None:
        if attributes is None:
            file_sizes = [
                int(head(url, timeout=5).headers["Content-Length"]) for url in dataset_urls
            ]
        else:
            # Can't get file sizes for partial downloads
            file_sizes = (None for _ in dataset_urls)

        pbar_tasks = [
            pbar.add_task(str(dest_path.relative_to(folder_path)), total=file_size)
            for dest_path, file_size in zip(dest_paths, file_sizes)
        ]
    else:
        pbar_tasks = (None for _ in dest_paths)

    with futures.ThreadPoolExecutor(min(num_threads, len(dest_paths))) as pool:
        for url, dest_path, pbar_task in zip(dataset_urls, dest_paths, pbar_tasks):
            futs = [
                pool.submit(
                    _download_dataset,
                    url,
                    dest_path,
                    attributes=attributes,
                    force=force,
                    block_size=block_size,
                    pbar_task=pbar_task,
                )
            ]
            for result in futures.wait(futs, return_when=futures.FIRST_EXCEPTION).done:
                if result.exception() is not None:
                    raise result.exception()

    return dest_paths


def _validate_attributes(data_name: str, attributes: Iterable[str]):
    """Checks that ``attributes`` contains only valid attributes for the given
    ``data_name``. If any attributes do not exist, raise a ValueError."""
    valid_attributes = list_attributes(data_name)
    invalid_attributes = [attr for attr in attributes if attr not in valid_attributes]
    if not invalid_attributes:
        return

    if len(invalid_attributes) == 1:
        values_err = f"'{invalid_attributes[0]}' is an invalid attribute for '{data_name}'"
    else:
        values_err = f"{invalid_attributes} are invalid attributes for '{data_name}'"

    raise ValueError(f"{values_err}. Valid attributes are: {valid_attributes}")


def load(  # pylint: disable=too-many-arguments
    data_name: str,
    attributes: Iterable[str] | None = None,
    folder_path: Path = Path("./datasets/"),
    force: bool = False,
    num_threads: int = 50,
    block_size: int = 8388608,
    progress_bar: bool | None = None,
    **params: ParamArg | str | list[str],
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
        progress_bar (bool) : Whether to show a progress bars for downloads. Defaults to True if running
            in an interactive terminal, False otherwise.
        params (kwargs)   : Keyword arguments exactly matching the parameters required for the data type.
            Note that these are not optional

    Returns:
        list[:class:`~pennylane.data.Dataset`]

    .. seealso:: :func:`~.load_interactive`, :func:`~.list_attributes`, :func:`~.list_data_names`.

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
    params = format_params(**params)

    if data_name == "other":
        data_name = params[0]["values"][0]
        params = []

    if attributes:
        _validate_attributes(data_name, attributes)

    folder_path = Path(folder_path)

    params = provide_defaults(data_name, params)
    params = [param for param in params if ("values", ParamArg.FULL) not in list(param.items())]

    dataset_ids_and_urls = get_dataset_urls(data_name, params)
    if dataset_ids_and_urls == []:
        raise ValueError(
            "No datasets exist for the provided configuration.\n"
            "Please check the available datasets by using the ``qml.data.list_datasets()`` function."
        )

    dataset_urls = [dataset_url for _, dataset_url in dataset_ids_and_urls]
    dataset_ids = [dataset_id for dataset_id, _ in dataset_ids_and_urls]

    progress_bar = progress_bar if progress_bar is not None else sys.stdout.isatty()

    if progress_bar:
        with progress.Progress() as pbar:
            download_paths = _download_datasets(
                data_name,
                folder_path,
                dataset_urls,
                dataset_ids,
                attributes,
                force=force,
                block_size=block_size,
                num_threads=num_threads,
                pbar=pbar,
            )

    else:
        download_paths = _download_datasets(
            data_name,
            folder_path,
            dataset_urls,
            dataset_ids,
            attributes,
            force=force,
            block_size=block_size,
            num_threads=num_threads,
            pbar=None,
        )

    return [Dataset.open(path, "a") for path in download_paths]


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

        if not isinstance(value, Mapping):
            return sorted(foldermap.keys())

        return {param: remove_paths(foldermap[param]) for param in foldermap.keys()}

    return remove_paths(_get_foldermap())


def _interactive_request_data_name(data_names):
    """Prompt the user to select a data name."""
    print("Please select the data name from the following:")
    for i, option in enumerate(data_names):
        print(f"{i + 1}: {option}")
    choice = input("Choice of data name: ").strip()
    if choice not in data_names:
        raise ValueError(f"Must select a single data name from {data_names}")
    return choice


def _interactive_request_attributes(attribute_options):
    """Prompt the user to select a list of attributes."""
    print(
        'Please select a list of attributes from the following available attributes or "full" for all attributes.'
    )
    for i, option in enumerate(attribute_options):
        print(f"{i + 1}: {option}")

    choice_input = input("Comma-separated list of attributes: ")
    choices = [str(choice).strip() for choice in choice_input.strip("[]").split(",")]
    if "full" in choices:
        return attribute_options
    if not (choices and set(choices).issubset(set(attribute_options))):
        raise ValueError(f"Must select a list of attributes from {attribute_options}")

    return choices


def _interactive_requests(parameters, parameter_tree):
    """Prompts the user to select parameters for datasets one at a time."""

    branch = parameter_tree
    for param in parameters:

        if len(branch["next"]) == 1:
            branch = next(iter(branch["next"].values()))
            continue

        print(f"Available options for {param}:")
        for i, option in enumerate(branch["next"].keys()):
            print(f"{i + 1}: {option}")
        user_value = input(f"Please select a {param}:").strip()
        try:
            branch = branch["next"][user_value]
        except KeyError as e:
            raise ValueError(f"Must enter a valid {param}:") from e

    return branch


def load_interactive():
    r"""Download a dataset using an interactive load prompt.

    Returns:
        :class:`~pennylane.data.Dataset`

    **Example**

    .. seealso:: :func:`~.load`, :func:`~.list_attributes`, :func:`~.list_data_names`.

    .. code-block :: pycon

        >>> qml.data.load_interactive()
        Please select the data name from the following:
            1: qspin
            2: qchem
            3: other
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
    """

    data_names = list_data_names()
    data_name = _interactive_request_data_name(data_names)

    parameters, attribute_options, parameter_tree = _get_parameter_tree(data_name)

    dataset_id = _interactive_requests(parameters, parameter_tree)
    attributes = _interactive_request_attributes(attribute_options)
    force = input("Force download files? (Default is no) [y/N]: ") in ["y", "Y"]
    dest_folder = Path(
        input("Folder to download to? (Default is pwd, will download to /datasets subdirectory): ")
    )
    print("\nPlease confirm your choices:")
    print("attributes:", attributes)
    print("force:", force)
    print("dest folder:", dest_folder / "datasets")
    print("dataset:", dataset_id)

    approve = input("Would you like to continue? (Default is yes) [Y/n]: ")
    if approve not in ["Y", "", "y"]:
        print("Aborting and not downloading!")
        return None

    return load(
        data_name,
        attributes=attributes,
        folder_path=dest_folder,
        force=force,
    )[0]


__all__ = (
    "load",
    "load_interactive",
    "list_data_names",
    "list_attributes",
    "FULL",
    "DEFAULT",
    "ParamArg",
)
