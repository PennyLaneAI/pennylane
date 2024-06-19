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
Tests for the :mod:`pennylane.data.base.hdf5` functions.
"""

from unittest.mock import MagicMock

import pytest

from pennylane.data.base import hdf5

pytestmark = pytest.mark.data

h5py = pytest.importorskip("h5py")


@pytest.fixture
def mock_fsspec(tmp_path, monkeypatch):
    """Returns a mock for the fsspec module."""

    with h5py.File(tmp_path / "temp.h5", "w") as f:
        f["data"] = "abc"

    mock_fs = MagicMock()
    mock_fs.open.return_value = tmp_path / "temp.h5"

    fsspec = MagicMock()
    fsspec.open.return_value = mock_fs

    monkeypatch.setattr(hdf5, "fsspec", fsspec)

    return fsspec


@pytest.fixture
def patch_h5py(monkeypatch):
    """Patches the 'h5py' module with a mock."""
    monkeypatch.setattr(hdf5, "h5py", MagicMock())


def test_open_hdf5_s3(mock_fsspec):  # pylint: disable=redefined-outer-name
    """Test that open_hdf5_s3 calls fsspec.open() with the expected arguments."""

    ret = hdf5.open_hdf5_s3("/bucket")

    assert isinstance(ret, h5py.File)
    mock_fsspec.open.assert_called_once_with(
        "/bucket", **{"cache_type": "mmap", "block_size": 8 * (2**20)}
    )


def test_copy_all_conflict_overwrite(tmp_path):
    """Test that on_conflict=overwrite overwrites an
    existing attribute in the destination."""

    src = h5py.File(tmp_path / "src", "w")
    src["x"] = 1

    dst = h5py.File(tmp_path / "dst", "w")
    dst["x"] = 2

    hdf5.copy_all(src, dst, on_conflict="overwrite")

    assert int(dst["x"][()]) == 1


def test_copy_all_conflict_raise(tmp_path):
    """Test that on_conflict=raise raises a
    ValueError if the attribute already exists in the destination."""

    src = h5py.File(tmp_path / "src", "w")
    src["x"] = 1

    dst = h5py.File(tmp_path / "dst", "w")
    dst["x"] = 2

    with pytest.raises(ValueError, match="Key 'x' already exists in '/x'"):
        hdf5.copy_all(src, dst, on_conflict="raise")


def test_copy_all_conflict_ignore(tmp_path):
    """Test that on_conflict=ignore does nothing if
    the attribute already exists in the destination."""

    src = h5py.File(tmp_path / "src", "w")
    src["x"] = 1

    dst = h5py.File(tmp_path / "dst", "w")
    dst["y"] = 3
    dst["x"] = 2

    assert int(dst["y"][()]) == 3
    assert int(dst["x"][()]) == 2
