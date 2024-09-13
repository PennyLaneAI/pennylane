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
Unit tests for the :class:`pennylane.data.data_manager` functions.
"""
import os
import re
from pathlib import Path, PosixPath
from typing import NamedTuple
from unittest.mock import MagicMock, call, patch

import pytest
import requests

import pennylane as qml
import pennylane.data.data_manager
from pennylane.data import Dataset
from pennylane.data.data_manager import GRAPHQL_URL, _validate_attributes

from .support import (
    _dataclass_ids,
    _get_urls_resp,
    _list_attrs_resp,
    _list_datasets_resp,
    _parameter_tree,
)

has_rich = False
try:
    import rich  # pylint:disable=unused-import

    has_rich = True
except ImportError:
    pass

# pylint:disable=protected-access,redefined-outer-name


pytestmark = pytest.mark.data


_folder_map = {
    "__params": {
        "qchem": ["molname", "basis", "bondlength"],
        "qspin": ["sysname", "periodicity", "lattice", "layout"],
    },
    "qchem": {
        "H2": {
            "6-31G": {
                "0.46": PosixPath("qchem/H2/6-31G/0.46.h5"),
                "1.16": PosixPath("qchem/H2/6-31G/1.16.h5"),
                "1.0": PosixPath("qchem/H2/6-31G/1.0.h5"),
            }
        }
    },
    "qspin": {
        "Heisenberg": {
            "closed": {"chain": {"1x4": PosixPath("qspin/Heisenberg/closed/chain/1x4/1.4.h5")}}
        }
    },
}

_data_struct = {
    "qchem": {
        "docstr": "Quantum chemistry dataset.",
        "params": ["molname", "basis", "bondlength"],
        "attributes": ["molecule", "hamiltonian", "sparse_hamiltonian", "hf_state", "full"],
    },
    "qspin": {
        "docstr": "Quantum many-body spin system dataset.",
        "params": ["sysname", "periodicity", "lattice", "layout"],
        "attributes": ["parameters", "hamiltonians", "ground_states", "full"],
    },
}


@pytest.fixture(scope="session")
def httpserver_listen_address():
    return ("localhost", 8888)


# pylint:disable=unused-argument
def get_mock(url, timeout=1.0):
    """Return the foldermap or data_struct according to URL"""
    resp = MagicMock(ok=True)
    resp.json.return_value = _folder_map if "foldermap" in url else _data_struct
    return resp


# pylint:disable=unused-argument
def graphql_mock(url, query, variables=None):
    """Return the JSON according to the query"""
    if "ListAttributes" in query:
        json_data = _list_attrs_resp
    elif "ListDatasets" in query:
        json_data = _list_datasets_resp
    elif "GetDatasetsForDownload" in query:
        json_data = _get_urls_resp
    elif "GetParameterTree" in query:
        json_data = _parameter_tree
    elif "GetDatasetClasses" in query:
        json_data = _dataclass_ids

    return json_data


def head_mock(url):
    """Return a fake header stating content-length is 1."""
    return NamedTuple("Head", headers=dict)(headers={"Content-Length": 10000})


@pytest.fixture
def mock_get_args():
    """A Mock object that tracks the arguments passed to ``mock_requests_get``."""

    return MagicMock()


@pytest.fixture(autouse=True)
def mock_requests_get(request, monkeypatch, mock_get_args):
    """Patches `requests.get()` in the data_manager module so that
    it returns mock JSON data for the foldermap and data struct."""

    def mock_get(url, *args, **kwargs):
        mock_get_args(url, *args, **kwargs)

        mock_resp = MagicMock()
        json_data = None

        mock_resp.json.return_value = json_data
        if hasattr(request, "param"):
            content = request.param
            mock_resp.content = content

            def mock_iter_content(chunk_size: int):
                """Mock for Response.iter_content()."""
                for i in range(0, len(content), chunk_size):
                    yield content[i : i + chunk_size]

            mock_resp.iter_content = mock_iter_content

        return mock_resp

    monkeypatch.setattr(qml.data.data_manager, "get", mock_get)

    return mock_get


def submit_download_mock(_self, _fetch_and_save, filename, dest_folder):
    """Patch to write a nonsense dataset rather than a downloaded one."""
    # If filename == foo/bar/x_y_z_attr.dat, content == "x_y_z_attr"
    content = os.path.splitext(os.path.basename(filename))[0]
    if content.split("_")[-1] == "full":
        content = {"molecule": content}
    qml.data.Dataset._write_file(content, os.path.join(dest_folder, filename))


# pylint:disable=unused-argument
def wait_mock_fixture(_futures, return_when=None):
    """Patch to avoid raising exceptions after collecting threads."""
    return MagicMock(done=[])


@pytest.fixture
def mock_load(monkeypatch):
    mock = MagicMock(return_value=[qml.data.Dataset()])
    monkeypatch.setattr(qml.data.data_manager, "load", mock)

    return mock


@patch.object(pennylane.data.data_manager, "_get_graphql", graphql_mock)
@patch.object(pennylane.data.data_manager, "head", head_mock)
@patch("pennylane.data.data_manager.sleep")
@patch("builtins.input")
class TestLoadInteractive:
    """
    Test the load_interactive function for various inputs. Side-effect args are ordered as:
    [data name, *params, attributes, Force, Folder, continue]
    """

    @pytest.mark.parametrize(
        ("side_effect"),
        [
            (["qspin", "Heisenberg", "1x4", "open", "full", True, PosixPath("/my/path"), "Y"]),
        ],
    )
    def test_load_interactive_success(
        self,
        mock_input,
        mock_sleep,
        mock_load,
        side_effect,
    ):  # pylint:disable=too-many-arguments, redefined-outer-name
        """Test that load_interactive succeeds."""
        mock_input.side_effect = side_effect
        assert isinstance(qml.data.load_interactive(), qml.data.Dataset)

    def test_load_interactive_without_confirm(
        self, mock_input, _mock_sleep, mock_load
    ):  # pylint:disable=redefined-outer-name
        """Test that load_interactive returns None if the user doesn't confirm."""
        mock_input.side_effect = [
            "qspin",
            "Heisenberg",
            "1x4",
            "open",
            "full",
            True,
            PosixPath("/my/path"),
            "n",
        ]
        assert qml.data.load_interactive() is None
        mock_load.assert_not_called()

    @pytest.mark.parametrize(
        ("side_effect", "error_message"),
        [
            (["foo"], re.escape("Must select a single data name from ['other', 'qchem', 'qspin']")),
            (["qspin", "foo"], "Must enter a valid sysname:"),
            (["qspin", "Ising", "foo"], "Must enter a valid layout:"),
            (["qspin", "Ising", "1x4", "foo"], "Must enter a valid periodicity:"),
        ],
    )
    def test_load_interactive_invalid_inputs(
        self, mock_input, _mock_sleep, mock_load, side_effect, error_message
    ):  # pylint: disable=redefined-outer-name
        """Test that load_interactive raises errors as expected."""
        mock_input.side_effect = side_effect
        with pytest.raises(ValueError, match=error_message):
            qml.data.load_interactive()


@patch.object(pennylane.data.data_manager, "_get_graphql", graphql_mock)
class TestMiscHelpers:
    """Test miscellaneous helper functions in data_manager."""

    def test_list_datasets(self):
        """Test list_datasets."""
        assert qml.data.list_datasets() == {
            "id": "qchem",
            "datasets": [
                {
                    "parameterValues": [
                        {"name": "molname", "value": "H2"},
                        {"name": "bondlength", "value": "1.16"},
                        {"name": "basis", "value": "STO-3G"},
                    ]
                }
            ],
        }

    def test_list_attributes(self):
        """Test list_attributes."""
        assert qml.data.list_attributes("qchem") == [
            "molecule",
            "hamiltonian",
            "sparse_hamiltonian",
            "hf_state",
            "full",
        ]


@pytest.fixture
def mock_download_dataset(monkeypatch):
    # pylint:disable=too-many-arguments
    def mock(data_path, dest, attributes, force, block_size, pbar_task):
        dset = Dataset.open(Path(dest), "w")
        dset.close()

    monkeypatch.setattr(pennylane.data.data_manager, "_download_dataset", mock)

    return mock


# pylint: disable=too-many-arguments
@patch.object(pennylane.data.data_manager, "head", head_mock)
@patch.object(pennylane.data.data_manager, "_get_graphql", graphql_mock)
@pytest.mark.usefixtures("mock_download_dataset")
@pytest.mark.parametrize(
    "data_name, params, expect_paths",
    [
        (
            "qchem",
            {"molname": "H2", "basis": "STO-3G", "bondlength": ["1.0", "0.46", "1.16"]},
            ["qchem/h2_sto-3g_1.0.h5", "qchem/h2_sto-3g_0.46.h5", "qchem/h2_sto-3g_1.16.h5"],
        )
    ],
)
@pytest.mark.parametrize("progress_bar", [True, False])
@pytest.mark.parametrize("attributes", [None, ["molecule"]])
def test_load(tmp_path, data_name, params, expect_paths, progress_bar, attributes):
    """Test that load fetches the correct datasets at the
    expected paths."""

    folder_path = tmp_path
    dsets = pennylane.data.data_manager.load(
        data_name=data_name,
        folder_path=folder_path,
        block_size=1,
        progress_bar=progress_bar,
        attributes=attributes,
        **params,
    )
    assert {Path(dset.bind.filename) for dset in dsets} == {
        Path(tmp_path, path) for path in expect_paths
    }


@patch.object(pennylane.data.data_manager, "_get_graphql", graphql_mock)
@patch.object(pennylane.data.data_manager, "head", head_mock)
def test_load_except(monkeypatch, tmp_path):
    """Test that an exception raised by _download_dataset is propagated."""
    monkeypatch.setattr(
        pennylane.data.data_manager, "_download_dataset", MagicMock(side_effect=ValueError("exc"))
    )

    with pytest.raises(ValueError, match="exc"):
        pennylane.data.data_manager.load(
            "qchem", molname="H2", basis="6-31G", bondlength="0.46", folder_path=tmp_path
        )


@patch("pennylane.data.data_manager._download_partial")
@patch("pennylane.data.data_manager._download_full")
@pytest.mark.parametrize("force", (True, False))
@pytest.mark.parametrize(
    "attributes, dest_exists, called_partial",
    [(["x"], True, True), (["x"], False, True), (None, True, True), (None, False, False)],
)
def test_download_dataset_full_or_partial(
    download_full, download_partial, attributes, dest_exists, force, called_partial
):  # pylint: disable=too-many-arguments
    """Test that _download_dataset calls ``_download_partial()`` if ``attributes`` is not None,
    or the dataset already exists at ``dest``, and that it only calls ``_download_full()`` if
    the dataset does not exist at ``dest`` and ``attributes`` is None.
    """

    dest = MagicMock()
    dest.exists.return_value = dest_exists

    pennylane.data.data_manager._download_dataset(
        "dataset/path", attributes=attributes, dest=dest, force=force, block_size=1, pbar_task=None
    )

    assert download_partial.called is called_partial
    assert download_full.called is not called_partial


@patch.object(pennylane.data.data_manager, "_get_graphql", graphql_mock)
@pytest.mark.parametrize("force", (True, False))
@patch("pennylane.data.data_manager._download_full")
def test_download_dataset_full_call(download_full, force):
    """Test that ``_download_dataset()`` passes the correct parameters
    to ``_download_full()``
    """
    dest = MagicMock()
    dest.exists.return_value = False
    pbar_task = MagicMock()

    pennylane.data.data_manager._download_dataset(
        f"{GRAPHQL_URL}/dataset/path",
        attributes=None,
        dest=dest,
        force=force,
        block_size=1,
        pbar_task=pbar_task,
    )

    download_full.assert_called_once_with(
        f"{GRAPHQL_URL}/dataset/path", block_size=1, dest=dest, pbar_task=pbar_task
    )


@patch.object(pennylane.data.data_manager, "_get_graphql", graphql_mock)
@pytest.mark.parametrize("attributes", [None, ["x"]])
@pytest.mark.parametrize("force", (True, False))
@patch("pennylane.data.data_manager._download_partial")
def test_download_dataset_partial_call(download_partial, attributes, force):
    """Test that ``_download_dataset()`` passes the correct parameters
    to ``_download_partial()``
    """
    dest = MagicMock()
    dest.exists.return_value = True
    pbar_task = MagicMock()

    pennylane.data.data_manager._download_dataset(
        f"{GRAPHQL_URL}/dataset/path",
        attributes=attributes,
        dest=dest,
        force=force,
        block_size=1,
        pbar_task=pbar_task,
    )

    download_partial.assert_called_once_with(
        f"{GRAPHQL_URL}/dataset/path",
        dest=dest,
        attributes=attributes,
        overwrite=force,
        block_size=1,
        pbar_task=pbar_task,
    )


@pytest.mark.usefixtures("mock_requests_get")
@pytest.mark.parametrize("mock_requests_get", [b"This is binary data"], indirect=True)
def test_download_full(tmp_path):
    """Tests that _download_dataset will fetch the dataset file
    at ``s3_url`` into ``dest``."""
    pennylane.data.data_manager._download_full(
        f"{GRAPHQL_URL}/dataset/path", tmp_path / "dataset", block_size=1, pbar_task=None
    )

    with open(tmp_path / "dataset", "rb") as f:
        assert f.read() == b"This is binary data"


@pytest.mark.usefixtures("mock_requests_get")
@pytest.mark.parametrize("mock_requests_get", [b"0123456789"], indirect=True)
def test_download_full_with_progress(tmp_path):
    """Tests that _download_dataset will fetch the dataset file
    at ``s3_url`` into ``dest`` and call the ``update`` method
    of the progress bar task."""
    pbar_task = MagicMock()
    pennylane.data.data_manager._download_full(
        "dataset/path", tmp_path / "dataset", block_size=4, pbar_task=pbar_task
    )

    with open(tmp_path / "dataset", "rb") as f:
        assert f.read() == b"0123456789"

    pbar_task.update.assert_has_calls([call(advance=4), call(advance=4), call(advance=2)])


@pytest.mark.parametrize("overwrite", [True, False])
@pytest.mark.parametrize(
    "attributes, expect_attrs",
    [(None, {"x": 1, "y": 2}), (["x"], {"x": 1}), (["x", "y"], {"x": 1, "y": 2})],
)
def test_download_partial_dest_not_exists(
    tmp_path, monkeypatch, attributes, expect_attrs, overwrite
):
    """Tests that _download_dataset will fetch only the requested attributes
    of a dataset when the destination does not exist."""
    remote_dataset = Dataset()
    remote_dataset.x = 1
    remote_dataset.y = 2

    monkeypatch.setattr(
        pennylane.data.data_manager, "open_hdf5_s3", MagicMock(return_value=remote_dataset.bind)
    )

    pennylane.data.data_manager._download_partial(
        "dataset/path",
        dest=tmp_path / "dataset",
        attributes=attributes,
        overwrite=overwrite,
        block_size=1,
        pbar_task=MagicMock(),
    )

    local = Dataset.open(tmp_path / "dataset")

    assert local.attrs == expect_attrs


@pytest.mark.parametrize(
    "attributes, overwrite, expect_attrs",
    [
        (None, False, {"x": "local_value", "y": 2, "z": 3}),
        (["x", "y", "z"], False, {"x": "local_value", "y": 2, "z": 3}),
        (None, True, {"x": "remote_value", "y": 2, "z": 3}),
        (["x", "y", "z"], True, {"x": "remote_value", "y": 2, "z": 3}),
        (["x"], False, {"x": "local_value"}),
        (["x"], True, {"x": "remote_value"}),
        (["y"], True, {"x": "local_value", "y": 2}),
        (["y"], False, {"x": "local_value", "y": 2}),
    ],
)
def test_download_partial_dest_exists(tmp_path, monkeypatch, attributes, overwrite, expect_attrs):
    """Tests that _download_dataset will fetch only the requested attributes
    of a dataset when the dataset does not exist locally."""
    remote_dataset = Dataset()
    remote_dataset.x = "remote_value"
    remote_dataset.y = 2
    remote_dataset.z = 3

    monkeypatch.setattr(
        pennylane.data.data_manager, "open_hdf5_s3", MagicMock(return_value=remote_dataset.bind)
    )

    local_dataset = Dataset.open(tmp_path / "dataset", "w")
    local_dataset.x = "local_value"
    local_dataset.close()

    pennylane.data.data_manager._download_partial(
        "dataset/path",
        dest=tmp_path / "dataset",
        attributes=attributes,
        overwrite=overwrite,
        block_size=1,
        pbar_task=MagicMock(),
    )

    local = Dataset.open(tmp_path / "dataset")

    assert local.attrs == expect_attrs


@patch("pennylane.data.data_manager.open_hdf5_s3")
def test_download_partial_no_check_remote(open_hdf5_s3, tmp_path):
    """Test that `_download_partial()` will not open the remote dataset if all requested attributes
    are present in the local dataset, and ``overwrite`` is ``False``."""
    local_dataset = Dataset.open(tmp_path / "dataset", "w")
    local_dataset.x = 1
    local_dataset.y = 2

    local_dataset.close()

    pennylane.data.data_manager._download_partial(
        "dataset_url",
        tmp_path / "dataset",
        ["x", "y"],
        overwrite=False,
        block_size=1,
        pbar_task=MagicMock(),
    )

    open_hdf5_s3.assert_not_called()


@patch("builtins.open")
@pytest.mark.usefixtures("mock_requests_get")
@patch.object(pennylane.data.data_manager, "head", head_mock)
@pytest.mark.parametrize(
    "dataset_url, escaped, dataset_id",
    [
        (
            f"{GRAPHQL_URL}/data/NH3+/data.h5",
            f"{GRAPHQL_URL}/data/NH3%2B/data.h5",
            "h2_sto-3g_0.46",
        ),
        (
            f"{GRAPHQL_URL}/data/CA$H/money.h5",
            f"{GRAPHQL_URL}/data/CA%24H/money.h5",
            "h2_sto-3g_0.46",
        ),
    ],
)
def test_download_datasets_escapes_url(
    _, tmp_path, mock_get_args, dataset_url, escaped, dataset_id
):
    """Tests that _download_dataset escapes special characters in a URL when doing a full download."""

    dest = MagicMock()
    dest.exists.return_value = False

    pennylane.data.data_manager._download_datasets(
        "qchem",
        folder_path=tmp_path,
        dataset_urls=[dataset_url],
        dataset_ids=[dataset_id],
        attributes=None,
        force=True,
        block_size=1,
        num_threads=1,
        pbar=MagicMock(),
    )

    mock_get_args.assert_called_once()
    assert mock_get_args.call_args[0] == (f"{escaped}",)


@patch.object(pennylane.data.data_manager, "_get_graphql", graphql_mock)
@patch("pennylane.data.data_manager._download_partial")
@pytest.mark.parametrize(
    "dataset_url, escaped, dataset_id",
    [
        (
            f"{GRAPHQL_URL}/data/NH3+/data.h5",
            f"{GRAPHQL_URL}/data/NH3%2B/data.h5",
            "h2_sto-3g_0.46",
        ),
        (
            f"{GRAPHQL_URL}/data/CA$H/money.h5",
            f"{GRAPHQL_URL}/data/CA%24H/money.h5",
            "h2_sto-3g_0.46",
        ),
    ],
)
def test_download_datasets_escapes_url_partial(
    download_partial, tmp_path, dataset_url, escaped, dataset_id
):
    """Tests that _download_dataset escapes special characters in a URL when doing a partial
    download."""
    attributes = ["attr"]
    force = False
    pbar = MagicMock()
    pbar_task = MagicMock()
    pbar.add_task.return_value = pbar_task
    data_name = "data_name"
    file_name = dataset_id + ".h5"

    pennylane.data.data_manager._download_datasets(
        data_name=data_name,
        folder_path=tmp_path,
        dataset_urls=[dataset_url],
        dataset_ids=[dataset_id],
        attributes=attributes,
        force=force,
        block_size=1,
        num_threads=1,
        pbar=pbar,
    )

    download_partial.assert_called_once_with(
        f"{escaped}",
        dest=tmp_path / data_name / file_name,
        attributes=attributes,
        overwrite=force,
        block_size=1,
        pbar_task=pbar_task,
    )


@patch.object(pennylane.data.data_manager, "_get_graphql", graphql_mock)
@pytest.mark.parametrize(
    "attributes,msg",
    [
        (
            ["molecule", "hamiltonian", "sparse_hamiltonian", "hf_state", "full", "foo"],
            r"'foo' is an invalid attribute for 'my_dataset'. Valid attributes are: \['molecule', 'hamiltonian', 'sparse_hamiltonian', 'hf_state', 'full'\]",
        ),
        (
            ["molecule", "hamiltonian", "sparse_hamiltonian", "hf_state", "full", "foo", "bar"],
            r"\['foo', 'bar'\] are invalid attributes for 'my_dataset'. Valid attributes are: \['molecule', 'hamiltonian', 'sparse_hamiltonian', 'hf_state', 'full'\]",
        ),
    ],
)
def test_validate_attributes_except(attributes, msg):
    """Test that ``_validate_attributes()`` raises a ValueError when passed
    invalid attributes."""

    with pytest.raises(ValueError, match=msg):
        _validate_attributes("my_dataset", attributes)
