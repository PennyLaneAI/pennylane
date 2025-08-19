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
Tests for the :class:`pennylane.data.data_manger.FolderMapView` class.
"""

import re

import pytest

from pennylane.data.data_manager import DEFAULT, FULL, DataPath
from pennylane.data.data_manager.foldermap import Description, FolderMapView

pytestmark = pytest.mark.data

FOLDERMAP = {
    "__params": {
        "qchem": ["molname", "basis", "bondlength"],
    },
    "qchem": {
        "O2": {
            "__default": "STO-3G",
            "STO-3G": {
                "__default": "0.5",
                "0.5": "qchem/O2/STO-3G/0.5.h5",
                "0.6": "qchem/O2/STO-3G/0.6.h5",
            },
            "6-31G": {
                "__default": "0.5",
                "0.5": "qchem/O2/6-31G/0.5.h5",
            },
        },
        "H2": {
            "__default": "STO-3G",
            "STO-3G": {"__default": "0.7", "0.7": "qchem/H2/STO-3G/0.7.h5"},
        },
    },
}


@pytest.fixture
def foldermap():
    """Foldermap fixture"""
    return FolderMapView(FOLDERMAP)


class TestFolderMapView:
    """Tests for ``FolderMapView``"""

    @pytest.mark.parametrize(
        "kwds, expect",
        [
            (
                {
                    "missing_default": DEFAULT,
                    "molname": "O2",
                    "basis": "STO-3G",
                    "bondlength": "0.5",
                },
                [
                    (
                        {"molname": "O2", "basis": "STO-3G", "bondlength": "0.5"},
                        "qchem/O2/STO-3G/0.5.h5",
                    )
                ],
            ),
            (
                {"missing_default": DEFAULT, "molname": "O2"},
                [
                    (
                        {"molname": "O2", "basis": "STO-3G", "bondlength": "0.5"},
                        "qchem/O2/STO-3G/0.5.h5",
                    )
                ],
            ),
            (
                {"missing_default": FULL, "molname": "O2"},
                [
                    (
                        {"molname": "O2", "basis": "STO-3G", "bondlength": "0.5"},
                        "qchem/O2/STO-3G/0.5.h5",
                    ),
                    (
                        {"molname": "O2", "basis": "STO-3G", "bondlength": "0.6"},
                        "qchem/O2/STO-3G/0.6.h5",
                    ),
                    (
                        {"molname": "O2", "basis": "6-31G", "bondlength": "0.5"},
                        "qchem/O2/6-31G/0.5.h5",
                    ),
                ],
            ),
            (
                {"missing_default": FULL, "molname": "O2", "basis": "STO-3G"},
                [
                    (
                        {"molname": "O2", "basis": "STO-3G", "bondlength": "0.5"},
                        "qchem/O2/STO-3G/0.5.h5",
                    ),
                    (
                        {"molname": "O2", "basis": "STO-3G", "bondlength": "0.6"},
                        "qchem/O2/STO-3G/0.6.h5",
                    ),
                ],
            ),
            (
                {"missing_default": DEFAULT, "molname": "O2", "basis": ["STO-3G", "6-31G"]},
                [
                    (
                        {"molname": "O2", "basis": "STO-3G", "bondlength": "0.5"},
                        "qchem/O2/STO-3G/0.5.h5",
                    ),
                    (
                        {"molname": "O2", "basis": "6-31G", "bondlength": "0.5"},
                        "qchem/O2/6-31G/0.5.h5",
                    ),
                ],
            ),
            (
                {"missing_default": DEFAULT, "molname": "O2", "basis": FULL, "bondlength": ["0.6"]},
                [
                    (
                        {"molname": "O2", "basis": "STO-3G", "bondlength": "0.6"},
                        "qchem/O2/STO-3G/0.6.h5",
                    ),
                ],
            ),
            (
                {
                    "missing_default": DEFAULT,
                    "molname": "O2",
                    "basis": FULL,
                    "bondlength": ["0.6", "200"],
                },
                [
                    (
                        {"molname": "O2", "basis": "STO-3G", "bondlength": "0.6"},
                        "qchem/O2/STO-3G/0.6.h5",
                    ),
                ],
            ),
            (
                {"missing_default": FULL, "molname": "O2", "bondlength": ["0.6"]},
                [
                    (
                        {"molname": "O2", "basis": "STO-3G", "bondlength": "0.6"},
                        "qchem/O2/STO-3G/0.6.h5",
                    ),
                ],
            ),
        ],
    )
    def test_find(self, foldermap, kwds, expect):  # pylint: disable=redefined-outer-name
        """Test that the ``find()`` method returns the expected results
        for a range of arguments."""

        assert set(foldermap.find("qchem", **kwds)) == {
            (Description(desc), DataPath(path)) for desc, path in expect
        }

    def test_find_not_top_level(self):
        """Test that a RuntimeError is raised if calling
        find() from below the top level of the foldermap."""

        with pytest.raises(
            RuntimeError, match=r"Can only call find\(\) from top level of foldermap"
        ):
            FolderMapView(FOLDERMAP)["qchem"].find("qchem")

    def test_find_invalid_data_name(self):
        """Test that a ValueError is raised for an invalid data_name."""
        with pytest.raises(ValueError, match="No datasets with data name: 'foo'"):
            FolderMapView(FOLDERMAP).find("foo")

    def test_find_missing_arg_no_default(self):
        """Test that a ValueError is raised when a parameter is not
        provided, and it doesn't have a default."""
        with pytest.raises(ValueError, match="No default available for parameter 'molname'"):
            FolderMapView(FOLDERMAP).find("qchem")

    @pytest.mark.parametrize(
        "arg, error_fmt", [("Z3", repr(["Z3"])), (("Z3", "Z4"), repr(["Z3", "Z4"]))]
    )
    def test_find_invalid_parameter(self, arg, error_fmt):
        """Test that a ValueError is raised when a parameter provided
        does not exist."""

        with pytest.raises(
            ValueError,
            match=re.escape(
                f"molname value(s) {error_fmt} are not available. Available values are: ['O2', 'H2']"
            ),
        ):
            FolderMapView(FOLDERMAP).find("qchem", molname=arg)

    @pytest.mark.parametrize("basis", [FULL, DEFAULT])
    def test_find_invalid_parameters_after_full_default(self, basis):
        """Test that a ValueError is raised when a parameter provided
        does not exist, after a 'full' or 'default' parameter has been provided for a
        higher-priority parameter."""

        with pytest.raises(
            ValueError,
            match=(
                r"bondlength value\(s\) \['0.20', '200'\] are not available. Available values are: \['0.5', '0.6'\]"
            ),
        ):
            FolderMapView(FOLDERMAP).find(
                "qchem", molname="O2", basis=basis, bondlength=["0.20", "200"]
            )

    @pytest.mark.parametrize(
        "init, key, expect",
        [
            (FOLDERMAP, "qchem", FolderMapView(FOLDERMAP["qchem"])),
            (FOLDERMAP["qchem"], "H2", FolderMapView(FOLDERMAP["qchem"]["H2"])),
            (FOLDERMAP["qchem"]["O2"]["STO-3G"], "0.5", DataPath("qchem/O2/STO-3G/0.5.h5")),
            (FOLDERMAP["qchem"]["O2"], DEFAULT, FolderMapView(FOLDERMAP["qchem"]["O2"]["STO-3G"])),
            (FOLDERMAP["qchem"]["O2"]["STO-3G"], "0.6", DataPath("qchem/O2/STO-3G/0.6.h5")),
            (FOLDERMAP["qchem"]["O2"]["STO-3G"], DEFAULT, DataPath("qchem/O2/STO-3G/0.5.h5")),
        ],
    )
    def test_gettitem(self, init, key, expect):
        """Test that ``getitem`` returns the expected values, including
        for default values and nested foldermaps."""
        assert FolderMapView(init)[key] == expect

    @pytest.mark.parametrize("key", ["__default"])
    @pytest.mark.parametrize(
        "init",
        [
            FOLDERMAP,
            FOLDERMAP["qchem"],
            FOLDERMAP["qchem"]["O2"],
        ],
    )
    def test_getitem_private_key(self, key, init):
        """Test that ``getitem`` raises a KeyError if the ``__default``
        key is passed."""

        with pytest.raises(KeyError):
            _ = FolderMapView(init)[key]

    @pytest.mark.parametrize(
        "init",
        [
            FOLDERMAP,
            FOLDERMAP["qchem"],
        ],
    )
    def test_getitem_default_none(self, init):
        """Test that ``getitem`` raises a ValueError if ``DEFAULT`` is
        used, but there is no default defined for that level."""

        with pytest.raises(ValueError, match="No default available"):
            _ = FolderMapView(init)[DEFAULT]

    @pytest.mark.parametrize(
        "init, keys",
        [
            (FOLDERMAP, {"qchem"}),
            (FOLDERMAP["qchem"], {"O2", "H2"}),
            (FOLDERMAP["qchem"]["O2"], {"STO-3G", "6-31G"}),
            (FOLDERMAP["qchem"]["O2"]["STO-3G"], {"0.5", "0.6"}),
        ],
    )
    def test_keys(self, init, keys):
        """Test that the ``keys()`` method returns only publicly visible keys."""

        assert set(FolderMapView(init).keys()) == keys

    @pytest.mark.parametrize(
        "init, len_",
        [
            (FOLDERMAP, 1),
            (FOLDERMAP["qchem"], 2),
            (FOLDERMAP["qchem"]["O2"], 2),
            (FOLDERMAP["qchem"]["O2"]["STO-3G"], 2),
        ],
    )
    def test_len(self, init, len_):
        """Test that the ``len()`` method returns the number of
        publicly visible keys only."""

        assert len(FolderMapView(init)) == len_

    def test_repr(self):
        """Test that __repr__ is equivalent to a dict repr, with the private
        keys removed."""
        assert repr(
            FolderMapView(
                {
                    "__params": {
                        "qchem": ["molname", "basis", "bondlength"],
                    },
                    "qchem": {
                        "H2": {
                            "__default": "STO-3G",
                            "STO-3G": {"__default": "0.7", "0.7": "qchem/H2/STO-3G/0.7.h5"},
                        }
                    },
                }
            )
        ) == repr({"qchem": {"H2": {"STO-3G": {"0.7": "qchem/H2/STO-3G/0.7.h5"}}}})

    def test_str(self):
        """Test that __str__ is equivalent to a dict __str__, with the private
        keys removed."""
        assert str(
            FolderMapView(
                {
                    "__params": {
                        "qchem": ["molname", "basis", "bondlength"],
                    },
                    "qchem": {
                        "H2": {
                            "__default": "STO-3G",
                            "STO-3G": {"__default": "0.7", "0.7": "qchem/H2/STO-3G/0.7.h5"},
                        }
                    },
                }
            )
        ) == str({"qchem": {"H2": {"STO-3G": {"0.7": "qchem/H2/STO-3G/0.7.h5"}}}})
