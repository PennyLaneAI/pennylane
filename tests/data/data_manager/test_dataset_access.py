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

# pylint:disable=protected-access
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from pathlib import PosixPath
from unittest.mock import MagicMock, patch

import pytest
import requests

import pennylane as qml

pytestmark = pytest.mark.data


_folder_map = {
    "qchem": {
        "H2": {
            "6-31G": {"0.46": PosixPath("0.46"), "1.16": PosixPath("1.16"), "1.0": PosixPath("1.0")}
        }
    },
    "qspin": {"Heisenberg": {"closed": {"chain": {"1x4": PosixPath("1.4")}}}},
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


@pytest.fixture(autouse=True)
def patch_requests_get(monkeypatch):
    """Patches `requests.get()` in the data_manager module so that
    it returns mock JSON data for the foldermap and data struct."""

    def mock_get(url, **kwargs):
        mock_resp = MagicMock()
        if url == qml.data.data_manager.FOLDERMAP_URL:
            json_data = _folder_map
        elif url == qml.data.data_manager.DATA_STRUCT_URL:
            json_data = _data_struct
        else:
            json_data = None

        mock_resp.json.return_value = json_data

        return mock_resp

    monkeypatch.setattr(qml.data.data_manager, "get", mock_get)


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


@patch.object(requests, "get", get_mock)
@patch("pennylane.data.data_manager.sleep")
@patch("pennylane.data.data_manager.load", return_value=[qml.data.Dataset()])
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
        self, mock_input, mock_load, mock_sleep, side_effect, data_name, kwargs, sleep_call_count
    ):  # pylint:disable=too-many-arguments
        """Test that load_interactive succeeds."""
        mock_input.side_effect = side_effect
        assert isinstance(qml.data.load_interactive(), qml.data.Dataset)
        mock_load.assert_called_once_with(data_name, **kwargs)
        assert mock_sleep.call_count == sleep_call_count

    def test_load_interactive_without_confirm(self, mock_input, mock_load, _mock_sleep):
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
        self, mock_input, _mock_load, _mock_sleep, side_effect, error_message
    ):
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
