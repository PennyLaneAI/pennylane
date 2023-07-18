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
Tests for :mod:`pennylane.data.base.attribute`.
"""

from typing import Iterable, List

import numpy as np
import pytest

from pennylane.data.attributes import (
    DatasetArray,
    DatasetNone,
    DatasetOperator,
    DatasetScalar,
    DatasetSparseArray,
    DatasetString,
)
from pennylane.data.base.attribute import AttributeInfo, match_obj_type


def _sort_types(types: Iterable[type]) -> List[type]:
    """
    pytest-split requires that test parameters are always in the same
    order between runs. This function ensures that collections of types
    used in test parameters are ordered.
    """
    return sorted(types, key=str)


@pytest.mark.parametrize(
    "type_or_obj, attribute_type",
    [
        (str, DatasetString),
        ("", DatasetString),
        ("abc", DatasetString),
        (0, DatasetScalar),
        (0.0, DatasetScalar),
        (np.int64(0), DatasetScalar),
        (complex(1, 2), DatasetScalar),
        (int, DatasetScalar),
        (complex, DatasetScalar),
        (np.array, DatasetArray),
        (np.array([0]), DatasetArray),
        (np.array([np.int64(0)]), DatasetArray),
        (np.array([complex(1, 2)]), DatasetArray),
        (np.zeros(shape=(5, 5, 7)), DatasetArray),
        (None, DatasetNone),
        (type(None), DatasetNone),
        *(
            (sp_cls, DatasetSparseArray)
            for sp_cls in _sort_types(DatasetSparseArray.consumes_types())
        ),
        *((op, DatasetOperator) for op in _sort_types(DatasetOperator.consumes_types())),
    ],
)
def test_match_obj_type(type_or_obj, attribute_type):
    """Test that ``match_obj_type`` returns the expected attribute
    type for each argument."""
    assert match_obj_type(type_or_obj) is attribute_type


class TestAttributeInfo:
    """Tests for ``AttributeInfo``."""

    @pytest.mark.parametrize(
        "kwargs, py_type",
        [
            ({"py_type": None}, "None"),
            ({"py_type": "None"}, "None"),
            ({"py_type": List[None]}, "list[None]"),
            ({}, None),
        ],
    )
    def test_py_type(self, kwargs, py_type):
        """Test that py_type can be set."""
        info = AttributeInfo(**kwargs)
        assert info.py_type == py_type

    @pytest.mark.parametrize("kwargs, doc", [({}, None), ({"doc": "some docs"}, "some docs")])
    def test_doc(self, kwargs, doc):
        """Test that doc can be set."""
        info = AttributeInfo(**kwargs)

        assert info.doc == doc

    def test_save(self):
        """Test that save() copies all items into the other
        AttributeInfo."""
        x, y = AttributeInfo(py_type="list", data=1), AttributeInfo(data=2)

        x.save(y)

        assert y.py_type == "list"
        assert y["data"] == 1

    def test_load(self):
        """Test the load() copies all items from the other AttributeInfo."""
        x, y = AttributeInfo(py_type="list", data=1), AttributeInfo(data=2)

        y.load(x)

        assert y.py_type == "list"
        assert y["data"] == 1

    def test_len(self):
        """Test that __iter__ returns the keys and attributes of ``AttributeInfo``."""

        info = AttributeInfo(py_type="list", data=1)

        assert set(iter(info)) == {"py_type", "data"}

    def test_iter(self):
        """Test that __len__ returns the number of items in ``AttributeInfo``."""

        info = AttributeInfo(py_type="list", doc="docs", data=1)

        assert len(info) == 3

    def test_delitem(self):
        """Test that __delitem__ deletes items and preserves __len__."""
        info = AttributeInfo(py_type="list", doc="docs", data=1)

        del info["py_type"]
        del info["data"]

        assert len(info) == 1
        assert set(info.keys()) == {"doc"}

    @pytest.mark.parametrize(
        "key, expect", [("py_type", "list"), ("doc", "some_docs"), ("data", 1)]
    )
    def test_getitem(self, key, expect):
        """Test that __getitem__ returns None if there is no value for a key
        that corresponds to an annotated attribute (e.g 'doc', 'py_type')."""
        info = AttributeInfo(py_type="list", doc="some_docs", data=1)

        assert info[key] == expect

    def test_setitem(self):
        """Test that __setitem__ can set a value on AttributeInfo."""
        info = AttributeInfo()

        info["data"] = 1
        assert info["data"] == 1

    def test_bind_key(self):
        """Test that bind_key returns the input, prefixed with the namespace."""
        assert AttributeInfo.bind_key("abc") == f"{AttributeInfo.attrs_namespace}.abc"

    def test_repr(self):
        """Test that __repr__ returns the expected value."""
        assert (
            repr(AttributeInfo(data=1, py_type="list"))
            == "AttributeInfo({'data': 1, 'py_type': 'list'})"
        )
