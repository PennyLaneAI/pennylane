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
from pathlib import Path, PosixPath
from unittest.mock import MagicMock, patch

import pytest
import requests

import pennylane as qml
import pennylane.data.data_manager
from pennylane.data import Dataset
from pennylane.data.data_manager import DataPath, S3_URL, _validate_attributes

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
        if url == qml.data.data_manager.FOLDERMAP_URL:
            json_data = _folder_map
        elif url == qml.data.data_manager.DATA_STRUCT_URL:
            json_data = _data_struct
        else:
            json_data = None

        mock_resp.json.return_value = json_data
        if hasattr(request, "param"):
            mock_resp.content = request.param

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


@patch.object(requests, "get", get_mock)
@patch("pennylane.data.data_manager.sleep")
@patch("builtins.input")
class TestLoadInteractive:
    """
    Test the load_interactive function for various inputs. Side-effect args are ordered as:
    [data name, *params, attributes, Force, Folder, continue]
    """

    @pytest.mark.parametrize(
        ("side_effect", "data_name", "kwargs", "sleep_call_count"),
        [
            (
                ["1", "1", "2", "", "", ""],
                "qchem",
                {
                    "attributes": ["hamiltonian"],
                    "folder_path": PosixPath(""),
                    "force": False,
                    "molname": "H2",
                    "basis": "6-31G",
                    "bondlength": "0.46",
                },
                2,
            ),
            (
                ["2", "1, 4", "Y", "/my/path", "y"],
                "qspin",
                {
                    "attributes": ["parameters", "full"],
                    "folder_path": PosixPath("/my/path"),
                    "force": True,
                    "sysname": "Heisenberg",
                    "periodicity": "closed",
                    "lattice": "chain",
                    "layout": "1x4",
                },
                4,
            ),
        ],
    )
    def test_load_interactive_success(
        self, mock_input, mock_sleep, mock_load, side_effect, data_name, kwargs, sleep_call_count
    ):  # pylint:disable=too-many-arguments, redefined-outer-name
        """Test that load_interactive succeeds."""
        mock_input.side_effect = side_effect
        assert isinstance(qml.data.load_interactive(), qml.data.Dataset)
        mock_load.assert_called_once_with(data_name, **kwargs)
        assert mock_sleep.call_count == sleep_call_count

    def test_load_interactive_without_confirm(
        self, mock_input, _mock_sleep, mock_load
    ):  # pylint:disable=redefined-outer-name
        """Test that load_interactive returns None if the user doesn't confirm."""
        mock_input.side_effect = ["1", "1", "2", "", "", "n"]
        assert qml.data.load_interactive() is None
        mock_load.assert_not_called()

    @pytest.mark.parametrize(
        ("side_effect", "error_message"),
        [
            (["foo"], "Must enter an integer between 1 and 2"),
            (["0"], "Must enter an integer between 1 and 2"),
            (["3"], "Must enter an integer between 1 and 2"),
            (["1", "1", "0"], "Must enter a list of integers between 1 and 5"),
            (["1", "1", "1 2"], "Must enter a list of integers between 1 and 5"),
        ],
    )
    def test_load_interactive_invalid_inputs(
        self, mock_input, _mock_sleep, mock_load, side_effect, error_message
    ):  # pylint: disable=redefined-outer-name
        """Test that load_interactive raises errors as expected."""
        mock_input.side_effect = side_effect
        with pytest.raises(ValueError, match=error_message):
            qml.data.load_interactive()


@patch.object(requests, "get", get_mock)
class TestMiscHelpers:
    """Test miscellaneous helper functions in data_manager."""

    def test_list_datasets(self, tmp_path):
        """Test that list_datasets returns either the S3 foldermap, or the local tree."""
        assert qml.data.list_datasets() == {
            "qspin": {"Heisenberg": {"closed": {"chain": ["1x4"]}}},
            "qchem": {"H2": {"6-31G": ["0.46", "1.0", "1.16"]}},
        }

    def test_list_attributes(self):
        """Test list_attributes"""
        assert qml.data.list_attributes("qchem") == _data_struct["qchem"]["attributes"]
        with pytest.raises(ValueError, match="Currently the hosted datasets are of types"):
            qml.data.list_attributes("invalid_data_name")


@pytest.fixture
def mock_download_dataset(monkeypatch):
    def mock(data_path, dest, attributes, force):
        dset = Dataset.open(Path(dest), "w")
        dset.close()

    monkeypatch.setattr(pennylane.data.data_manager, "_download_dataset", mock)

    return mock


@pytest.mark.usefixtures("mock_download_dataset")
@pytest.mark.parametrize(
    "data_name, params, expect_paths",
    [
        (
            "qchem",
            {"molname": "H2", "basis": "6-31G", "bondlength": ["0.46", "1.16"]},
            ["qchem/H2/6-31G/0.46.h5", "qchem/H2/6-31G/1.16.h5"],
        )
    ],
)
def test_load(tmp_path, data_name, params, expect_paths):
    """Test that load fetches the correct datasets at the
    expected paths."""

    folder_path = tmp_path
    dsets = pennylane.data.data_manager.load(
        data_name=data_name,
        folder_path=folder_path,
        **params,
    )

    assert {Path(dset.bind.filename) for dset in dsets} == {
        Path(tmp_path, path) for path in expect_paths
    }


def test_load_except(monkeypatch, tmp_path):
    """Test that an exception raised by _download_dataset is propagated."""
    monkeypatch.setattr(
        pennylane.data.data_manager, "_download_dataset", MagicMock(side_effect=ValueError("exc"))
    )

    with pytest.raises(ValueError, match="exc"):
        pennylane.data.data_manager.load(
            "qchem", molname="H2", basis="6-31G", bondlength="0.46", folder_path=tmp_path
        )


@pytest.mark.usefixtures("mock_requests_get")
@pytest.mark.parametrize("mock_requests_get", [b"This is binary data"], indirect=True)
def test_download_dataset_full(tmp_path):
    """Tests that _download_dataset will fetch the dataset file
    using requests if all attributes are requested."""

    pennylane.data.data_manager._download_dataset(
        "dataset/path",
        tmp_path / "dataset",
        attributes=None,
    )

    with open(tmp_path / "dataset", "rb") as f:
        assert f.read() == b"This is binary data"


@pytest.mark.usefixtures("mock_requests_get")
@pytest.mark.parametrize("mock_requests_get", [b"This is downloaded data"], indirect=True)
@pytest.mark.parametrize(
    "force, expect_data", [(False, b"This is local data"), (True, b"This is downloaded data")]
)
def test_download_dataset_full_already_exists(tmp_path, force, expect_data):
    """Tests that _download_dataset will only overwrite an exists
    dataset in the same path if `force` is True."""

    with open(tmp_path / "dataset", "wb") as f:
        f.write(b"This is local data")

    pennylane.data.data_manager._download_dataset(
        "dataset/path", tmp_path / "dataset", attributes=None, force=force
    )

    with open(tmp_path / "dataset", "rb") as f:
        assert f.read() == expect_data


def test_download_dataset_partial(tmp_path, monkeypatch):
    """Tests that _download_dataset will fetch the dataset file
    using requests if all attributes are requested."""

    remote_dataset = Dataset()
    remote_dataset.x = 1
    remote_dataset.y = 2

    monkeypatch.setattr(
        pennylane.data.data_manager, "open_hdf5_s3", MagicMock(return_value=remote_dataset.bind)
    )

    pennylane.data.data_manager._download_dataset(
        "dataset/path", tmp_path / "dataset", attributes=["x"]
    )

    local = Dataset.open(tmp_path / "dataset")

    assert local.x == 1
    assert not hasattr(local, "y")


@patch("builtins.open")
@pytest.mark.parametrize(
    "datapath, escaped",
    [("data/NH3+/data.h5", "data/NH3%2B/data.h5"), ("data/CA$H/money.h5", "data/CA%24H/money.h5")],
)
def test_download_dataset_escapes_url(_, mock_get_args, datapath, escaped):
    """Tests that _download_dataset escapes special characters in a URL when doing a full download."""

    dest = MagicMock()
    dest.exists.return_value = False

    pennylane.data.data_manager._download_dataset(DataPath(datapath), dest=dest, attributes=None)

    mock_get_args.assert_called_once()
    assert mock_get_args.call_args[0] == (f"{S3_URL}/{escaped}",)


@patch("pennylane.data.data_manager._download_partial")
@pytest.mark.parametrize(
    "datapath, escaped",
    [("data/NH3+/data.h5", "data/NH3%2B/data.h5"), ("data/CA$H/money.h5", "data/CA%24H/money.h5")],
)
def test_download_dataset_escapes_url_partial(mock_download_partial, datapath, escaped):
    """Tests that _download_dataset escapes special characters in a URL when doing a partial
    download."""
    dest = Path("dest")
    attributes = ["attr"]
    force = False

    pennylane.data.data_manager._download_dataset(
        DataPath(datapath), dest=dest, attributes=attributes, force=force
    )

    mock_download_partial.assert_called_once_with(
        f"{S3_URL}/{escaped}", dest, attributes, overwrite=force
    )


@pytest.mark.parametrize(
    "attributes,msg",
    [
        (
            ["x", "y", "z", "foo"],
            r"'foo' is an invalid attribute for 'my_dataset'. Valid attributes are: \['x', 'y', 'z'\]",
        ),
        (
            ["x", "y", "z", "foo", "bar"],
            r"\['foo', 'bar'\] are invalid attributes for 'my_dataset'. Valid attributes are: \['x', 'y', 'z'\]",
        ),
    ],
)
def test_validate_attributes_except(attributes, msg):
    """Test that ``_validate_attributes()`` raises a ValueError when passed
    invalid attributes."""

    data_struct = {"my_dataset": {"attributes": ["x", "y", "z"]}}

    with pytest.raises(ValueError, match=msg):
        _validate_attributes(data_struct, "my_dataset", attributes)
