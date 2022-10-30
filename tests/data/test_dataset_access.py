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
Unit tests for the :class:`pennylane.data.Dataset` class and its functions.
"""
# pylint:disable=protected-access
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch
from glob import glob
import os

import pytest
import requests
import pennylane as qml
from pennylane.data.data_manager import _generate_folders as original_generate_folders

_folder_map = {
    "qchem": {"H2": {"6-31G": ["0.46", "1.16", "1.0"]}},
    "qspin": {"Heisenberg": {"closed": {"chain": ["1x4"]}}},
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


def get_mock(url, timeout=1.0):
    """Return the foldermap or data_struct according to URL"""
    resp = MagicMock(ok=True)
    resp.json.return_value = _folder_map if "foldermap" in url else _data_struct
    return resp


def submit_download_mock(_self, _fetch_and_save, filename, dest_folder):
    """Patch to write a nonsense dataset rather than a downloaded one."""
    # If filename == foo/bar/x_y_z_attr.dat, content == "x_y_z_attr"
    content = os.path.splitext(os.path.basename(filename))[0]
    if content.split("_")[-1] == "full":
        content = {"molecule": content}
    qml.data.Dataset._write_file(content, os.path.join(dest_folder, filename))


def wait_mock_fixture(_futures, return_when=None):
    """Patch to avoid raising exceptions after collecting threads."""
    return MagicMock(done=[])


@patch.object(qml.data.data_manager, "_foldermap", _folder_map)
@patch.object(qml.data.data_manager, "_data_struct", _data_struct)
@patch.object(requests, "get", get_mock)
class TestValidateParams:
    """Test the _validate_params function."""

    def test_data_name_error(self):
        """Test that _validate_params fails when an unknown data_name is passed in."""
        with pytest.raises(ValueError, match="Currently the hosted datasets are of types"):
            qml.data.data_manager._validate_params("qspn", {}, [])

    @pytest.mark.parametrize(
        ("description", "error_message"),
        [
            (
                {"molname": ["H2"], "basis": ["6-31G"]},
                r"Supported parameter values for qchem are \['molname', 'basis', 'bondlength'\], but got \['molname', 'basis'\].",
            ),
            (
                {
                    "molname": ["H2"],
                    "basis": ["6-31G"],
                    "bondlength": ["0.46"],
                    "unexpected": ["foo"],
                },
                r"Supported parameter values for qchem are \['molname', 'basis', 'bondlength'\], but got \['molname', 'basis', 'bondlength', 'unexpected'\].",
            ),
        ],
    )
    def test_incorrect_set_of_params(self, description, error_message):
        """Test that _validate_params fails when the kwargs do not exactly match the dataset."""
        with pytest.raises(ValueError, match=error_message):
            qml.data.data_manager._validate_params("qchem", description, [])

    @pytest.mark.parametrize(
        ("description", "error_message"),
        [
            (
                {"molname": ["foo"], "basis": ["6-31G"], "bondlength": ["0.46"]},
                r"molname value of 'foo' is not available. Available values are \['H2'\]",
            ),
            (
                {"molname": ["H2"], "basis": ["foo"], "bondlength": ["0.46"]},
                r"basis value of 'foo' is not available. Available values are \['6-31G'\]",
            ),
            (
                {"molname": ["H2"], "basis": ["6-31G"], "bondlength": ["foo"]},
                r"bondlength value of 'foo' is not available. Available values are \['0.46', '1.16', '1.0'\]",
            ),
            (
                {"molname": ["H2"], "basis": ["6-31G", "foo"], "bondlength": ["0.46"]},
                r"basis value of 'foo' is not available. Available values are \['6-31G'\]",
            ),
        ],
    )
    def test_incorrect_param_values(self, description, error_message):
        """Test that _validate_params fails when an unrecognized parameter value is given."""
        with pytest.raises(ValueError, match=error_message):
            qml.data.data_manager._validate_params("qchem", description, ["full"])

    @pytest.mark.parametrize(
        ("attributes", "error_type", "error_message"),
        [
            (None, TypeError, "Arg 'attributes' should be a list, but got NoneType"),
            (
                ["molecule", "full", "foo"],
                ValueError,
                r"Supported key values for qchem are \['molecule', 'hamiltonian', 'sparse_hamiltonian', 'hf_state', 'full'], but got \['molecule', 'full', 'foo'\].",
            ),
            (
                ["foo"],
                ValueError,
                r"Supported key values for qchem are \['molecule', 'hamiltonian', 'sparse_hamiltonian', 'hf_state', 'full'\], but got \['foo'\].",
            ),
        ],
    )
    def test_attributes_must_be_list(self, attributes, error_type, error_message):
        """Tests that 'attributes' must be a list of attributes from the _data_struct."""
        with pytest.raises(error_type, match=error_message):
            qml.data.data_manager._validate_params(
                "qchem",
                {"molname": ["H2"], "basis": ["6-31G"], "bondlength": ["0.46"]},
                attributes,
            )

    @pytest.mark.parametrize(
        ("description", "error_message"),
        [
            (
                {"molname": [15], "basis": ["6-31G"], "bondlength": ["0.46"]},
                "Invalid type 'int' for parameter 'molname'",
            ),
            (
                {"molname": ["H2"], "basis": ["6-31G"], "bondlength": [1.1, 1 + 2j]},
                r"Invalid bondlength '\(1\+2j\)'. Must be a string, int or float",
            ),
            (
                {"layout": [1, [1, 4]]},
                "Invalid layout value of '1'. Must be a string or a tuple of ints.",
            ),
        ],
    )
    def test_parameters_with_bad_type_fail(self, description, error_message):
        """Test that _format_details fails if a parameter fails a custom type-check."""
        with pytest.raises(TypeError, match=error_message):
            _ = {
                param: qml.data.data_manager._format_details(param, details)
                for param, details in description.items()
            }

    @pytest.mark.parametrize(
        ("param", "details", "expected"),
        [
            ("layout", [1], ["1"]),
            ("layout", ["foo", "bar", "baz"], ["foo", "bar", "baz"]),
            ("layout", [1, 4], ["1x4"]),
            ("layout", [[1, 4]], ["1x4"]),
            ("bondlength", [1, 1.100], ["1.0", "1.1"]),
            ("random", "foo", ["foo"]),
            ("random", ["foo", "bar"], ["foo", "bar"]),
        ],
    )
    def test_format_details_success(self, param, details, expected):
        """Test that _format_detail behaves as expected."""
        assert qml.data.data_manager._format_details(param, details) == expected

    @pytest.mark.parametrize(
        ("data_name", "description", "attributes"),
        [
            (
                "qchem",
                {"molname": "H2", "basis": "6-31G", "bondlength": "0.46"},
                ["full"],
            ),
            (
                "qchem",
                {"molname": ["full"], "basis": ["6-31G"], "bondlength": [0.460]},
                ["full"],
            ),
            (
                "qchem",
                {"molname": ["full"], "basis": ["6-31G"], "bondlength": [1]},
                ["full"],
            ),
            (
                "qchem",
                {"molname": ["full"], "basis": ["6-31G"], "bondlength": [1.0]},
                ["full"],
            ),
            (
                "qchem",
                {"molname": ["full"], "basis": ["6-31G"], "bondlength": ["full"]},
                ["molecule", "hamiltonian", "sparse_hamiltonian"],
            ),
            (
                "qspin",
                {
                    "sysname": ["Heisenberg"],
                    "periodicity": ["closed"],
                    "lattice": ["chain"],
                    "layout": ["1x4"],
                },
                ["full", "ground_states"],
            ),
            (
                "qspin",
                {
                    "sysname": ["Heisenberg"],
                    "periodicity": ["closed"],
                    "lattice": ["chain"],
                    "layout": [(1, 4)],
                },
                ["full"],
            ),
            (
                "qspin",
                {
                    "sysname": ["Heisenberg"],
                    "periodicity": ["closed"],
                    "lattice": ["chain"],
                    "layout": [[1, 4]],
                },
                ["full"],
            ),
            (
                "qspin",
                {
                    "sysname": "Heisenberg",
                    "periodicity": "closed",
                    "lattice": "chain",
                    "layout": [1, 4],
                },
                ["full"],
            ),
            (
                "qspin",
                {
                    "sysname": ["full"],
                    "periodicity": ["closed"],
                    "lattice": ["full"],
                    "layout": ["full"],
                },
                ["full", "ground_states"],
            ),
        ],
    )
    def test_validate_params_successes(self, data_name, description, attributes):
        """Test that the _validate_params method passes with valid parameters."""
        description = {
            param: qml.data.data_manager._format_details(param, details)
            for param, details in description.items()
        }
        qml.data.data_manager._validate_params(data_name, description, attributes)


class TestLoadHelpers:
    """Test the helper functions used by load()."""

    @pytest.mark.parametrize(
        ("node", "folders", "output"),
        [
            (
                {
                    "H2": {"STO-3G": ["0.46", "0.48", "0.50"], "6-31G": ["0.46", "0.50"]},
                    "HeH": {"STO-3G": ["0.50"]},
                },
                [["full"], ["full"], ["0.48", "0.50"]],
                ["H2/6-31G/0.50", "H2/STO-3G/0.48", "H2/STO-3G/0.50", "HeH/STO-3G/0.50"],
            ),
            (
                {
                    "H2": {"STO-3G": ["0.46"], "6-31G": ["0.46", "0.50"]},
                    "HeH": {"STO-3G": ["0.46"]},
                },
                [["full"], ["STO-3G"], ["0.50"]],
                [],
            ),
        ],
    )
    def test_generate_folders(self, node, folders, output):
        """Test the _generate_folders helper function."""
        assert sorted(qml.data.data_manager._generate_folders(node, folders)) == output

    @patch("concurrent.futures.ThreadPoolExecutor.submit", return_value=True)
    @patch("pennylane.data.data_manager.wait", return_value=MagicMock(done=[]))
    class TestS3Download:
        """Test the _s3_download utility function with various inputs."""

        def test_s3_download_basic(self, wait_mock, submit_mock, tmp_path):
            """Test the _s3_download helper function."""
            dest = str(tmp_path)
            qml.data.data_manager._s3_download(
                "qchem",
                ["H2/6-31G/0.50", "H2/STO-3G/0.48"],
                ["molecule"],
                dest,
                False,
                50,
            )
            assert submit_mock.call_count == 2
            expected_args_used = [
                "qchem/H2/STO-3G/0.48/H2_STO-3G_0.48_molecule.dat",
                "qchem/H2/6-31G/0.50/H2_6-31G_0.50_molecule.dat",
            ]
            actual_args_used = [i[0][1] for i in submit_mock.call_args_list]  # [args][second arg]
            assert sorted(expected_args_used) == sorted(actual_args_used)
            assert wait_mock.called_once_with([True, True])

        def test_s3_download_force_false(self, _wait_mock, submit_mock, tmp_path):
            """Test _s3_download with force=False"""
            dest = str(tmp_path)
            data = qml.data.Dataset(molecule="already_exists")
            filename = os.path.join(dest, "qchem/H2/6-31G/0.50/H2_6-31G_0.50_molecule.dat")
            data.write(filename)
            qml.data.data_manager._s3_download(
                "qchem",
                ["H2/6-31G/0.50", "H2/STO-3G/0.48"],
                ["molecule"],
                dest,
                False,
                50,
            )
            assert submit_mock.call_count == 1
            assert (
                submit_mock.call_args_list[0][0][1]  # [first call][args][second arg]
                == "qchem/H2/STO-3G/0.48/H2_STO-3G_0.48_molecule.dat"
            )

            loaded_data = qml.data.Dataset()
            loaded_data.read(filename)
            assert loaded_data.molecule == "already_exists"

        def test_s3_download_force_true(self, _wait_mock, submit_mock, tmp_path):
            """Test _s3_download with force=True"""

            def submit_side_effect(_fetch, filename, dest_folder):
                dataset = qml.data.Dataset(molecule="new_data")
                dataset.write(os.path.join(dest_folder, filename))

            submit_mock.side_effect = submit_side_effect
            dest = str(tmp_path)
            data = qml.data.Dataset(molecule="already_exists")
            filename = os.path.join(dest, "qchem/H2/6-31G/0.50/H2_6-31G_0.50_molecule.dat")
            data.write(filename)
            qml.data.data_manager._s3_download(
                "qchem",
                ["H2/6-31G/0.50", "H2/STO-3G/0.48"],
                ["molecule"],
                dest,
                True,
                50,
            )
            assert submit_mock.call_count == 2
            loaded_data = qml.data.Dataset()
            loaded_data.read(filename)
            assert loaded_data.molecule == "new_data"

        def test_s3_download_thread_failure(self, wait_mock, _submit_mock, tmp_path):
            """Test that _s3_download raises errors from download failures."""
            wait_mock.return_value = MagicMock(
                done=[MagicMock(exception=MagicMock(return_value=ValueError("network error")))]
            )
            with pytest.raises(ValueError, match="network error"):
                qml.data.data_manager._s3_download(
                    "qchem", ["H2/6-31G/0.50"], ["molecule"], str(tmp_path), False, 50
                )

    @pytest.mark.parametrize(
        ("filename", "called_with"),
        [
            ("my/file/in/s3.dat", os.path.join(qml.data.data_manager.S3_URL, "my/file/in/s3.dat")),
            (
                "qchem/H3+/STO-3G/1.0/H3+_STO-3G_1.0_full.dat",
                os.path.join(
                    qml.data.data_manager.S3_URL, "qchem/H3%2B/STO-3G/1.0/H3%2B_STO-3G_1.0_full.dat"
                ),
            ),
        ],
    )
    @patch("requests.get")
    def test_fetch_and_save(self, get_and_write_mock, tmp_path, filename, called_with):
        """Test the _fetch_and_save helper function."""
        get_return = MagicMock()
        get_return.raise_for_status.return_value = None
        get_return.content = b"foobar"
        get_and_write_mock.return_value = get_return

        dest = str(tmp_path / "datasets")
        destfile = os.path.join(dest, filename)
        os.makedirs(os.path.dirname(destfile))

        qml.data.data_manager._fetch_and_save(filename, dest)
        get_and_write_mock.assert_called_once_with(called_with, timeout=5.0)
        with open(destfile, "rb") as f:
            assert f.read() == b"foobar"


@patch.object(requests, "get", get_mock)
@patch.object(ThreadPoolExecutor, "submit", submit_download_mock)
@patch.object(qml.data.data_manager, "wait", wait_mock_fixture)
class TestLoad:
    """Test the load() method."""

    def test_bad_parameters(self):
        """Test the user experience of entering invalid parameters."""
        with pytest.raises(ValueError, match="Currently the hosted datasets are of types"):
            qml.data.load(None)
        with pytest.raises(
            ValueError,
            match=r"Supported parameter values for qchem are \['molname', 'basis', 'bondlength'\], but got \[\].",
        ):
            qml.data.load("qchem")
        with pytest.raises(
            ValueError,
            match=r"molname value of 'foo' is not available. Available values are \['H2'\]",
        ):
            qml.data.load("qchem", molname="foo", basis="bar", bondlength="baz")
        with pytest.raises(
            ValueError,
            match=r"basis value of 'bar' is not available. Available values are \['6-31G'\]",
        ):
            qml.data.load("qchem", molname="H2", basis="bar", bondlength="baz")
        with pytest.raises(
            ValueError,
            match=r"bondlength value of 'baz' is not available. Available values are \['0.46', '1.16', '1.0'\]",
        ):
            qml.data.load("qchem", molname="H2", basis="6-31G", bondlength="baz")

    def test_successful_load_single(self, tmp_path):
        """Test that qml.data.load successfully loads a dataset"""
        dest = str(tmp_path)
        datasets = qml.data.load(
            "qchem",
            molname="H2",
            basis="6-31G",
            bondlength="0.46",
            attributes=["molecule"],
            folder_path=dest,
        )
        assert len(datasets) == 1
        data = datasets[0]

        assert data._is_standard
        assert data.attrs == {"molecule": None}
        assert data.molecule == "H2_6-31G_0.46_molecule"
        assert data._dtype == "qchem"
        assert data._folder == os.path.join(dest, "datasets/qchem/H2/6-31G/0.46")
        assert data._prefix == os.path.join(
            dest, "datasets/qchem/H2/6-31G/0.46/H2_6-31G_0.46_{}.dat"
        )
        assert data._fullfile is None

    def test_successful_load_many(self, tmp_path):
        """Test that qml.data.load successfully loads multiple datasets at once."""
        datasets = qml.data.load(
            "qchem", molname="H2", basis="6-31G", bondlength="full", folder_path=str(tmp_path)
        )
        assert len(datasets) == 3
        assert sorted([d.molecule for d in datasets]) == [
            "H2_6-31G_0.46_full",
            "H2_6-31G_1.0_full",
            "H2_6-31G_1.16_full",
        ]

    @pytest.mark.parametrize(
        ("attributes", "fullfile"),
        [(["full"], "H2/6-31G/0.46/H2_6-31G_0.46_full.dat"), (["molecule"], None)],
    )
    def test_fullfile_attribute(self, tmp_path, attributes, fullfile):
        """Test that the _fullfile attribute is set if a 'full' file is downloaded."""
        dest = str(tmp_path)
        data = qml.data.load(
            "qchem",
            molname="H2",
            basis="6-31G",
            bondlength="0.46",
            attributes=attributes,
            folder_path=dest,
        )[0]
        assert data._fullfile is None if fullfile is None else os.path.join(dest, fullfile)

    @patch("pennylane.data.data_manager._generate_folders", side_effect=original_generate_folders)
    def test_full_overrides_nonfull(self, generate_mock, tmp_path):
        """When a list has 'full' plus other things, assert that only 'full' is downloaded."""
        dest = str(tmp_path)
        datasets = qml.data.load(
            "qchem",
            molname="H2",
            basis="6-31G",
            bondlength=["full", "0.46"],
            attributes=["full", "molecule"],
            folder_path=dest,
        )
        assert len(datasets) == 3
        assert generate_mock.call_count == 3
        arglist = generate_mock.call_args_list
        assert arglist[0][0] == (  # [first call][args]
            {"H2": {"6-31G": ["0.46", "1.16", "1.0"]}},
            [["H2"], ["6-31G"], ["full"]],  # 0.46 was removed from the last list!
        )

        # these test that _generate_folders carried on as expected
        assert arglist[1][0] == ({"6-31G": ["0.46", "1.16", "1.0"]}, [["6-31G"], ["full"]])
        assert arglist[2][0] == (["0.46", "1.16", "1.0"], [["full"]])

        # assert that the "molecule" attribute was dropped
        assert sorted(glob(os.path.join(dest, "**/*.dat"), recursive=True)) == [
            os.path.join(dest, "datasets/qchem/H2/6-31G/0.46/H2_6-31G_0.46_full.dat"),
            os.path.join(dest, "datasets/qchem/H2/6-31G/1.0/H2_6-31G_1.0_full.dat"),
            os.path.join(dest, "datasets/qchem/H2/6-31G/1.16/H2_6-31G_1.16_full.dat"),
        ]

    def test_docstr_is_added_to_loaded_dataset(self, tmp_path):
        """Test that a docstring describing all attributes on a Dataset is set."""
        dest = str(tmp_path)
        data = qml.data.load(
            "qchem",
            molname="H2",
            basis="6-31G",
            bondlength="0.46",
            attributes=["molecule"],
            folder_path=dest,
        )[0]
        assert data.__doc__ == "Quantum chemistry dataset."


@patch.object(requests, "get", get_mock)
def test_list_datasets(tmp_path):
    """Test that list_datasets returns either the S3 foldermap, or the local tree."""
    qml.data.data_manager._foldermap = {}
    assert qml.data.list_datasets() == _folder_map

    first = qml.data.Dataset(foo="this")
    second = qml.data.Dataset(bar="that")

    d = tmp_path / "datasets"
    d1 = d / "qspin" / "data_1" / "foo_size"
    d2 = d / "qspin" / "data_2" / "bar_size"

    first.write(str(d1 / "first.dat"))
    second.write(str(d2 / "second.dat"))

    assert qml.data.list_datasets(str(d)) == {
        "qspin": {"data_1": ["foo_size"], "data_2": ["bar_size"]}
    }
