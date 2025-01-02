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

# pylint: disable=unused-argument,unused-variable,too-many-public-methods

from collections.abc import Iterable
from copy import copy, deepcopy
from typing import Any

import numpy as np
import pytest

from pennylane.data.attributes import (
    DatasetArray,
    DatasetList,
    DatasetNone,
    DatasetOperator,
    DatasetScalar,
    DatasetSparseArray,
    DatasetString,
    DatasetTuple,
)
from pennylane.data.base.attribute import (
    UNSET,
    AttributeInfo,
    DatasetAttribute,
    attribute,
    match_obj_type,
)
from pennylane.data.base.hdf5 import HDF5Group, create_group

pytestmark = pytest.mark.data

pytest.importorskip("h5py")


def _sort_types(types: Iterable[type]) -> list[type]:
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
        (DatasetNone(), DatasetNone),
        (DatasetArray([1, 2, 3]), DatasetArray),
        (DatasetString("abc"), DatasetString),
        (DatasetList([1, 2, 3]), DatasetList),
        ([1, 2, 3], DatasetList),
        (DatasetTuple((1, "a", [3])), DatasetTuple),
        ((1, 2, 3), DatasetTuple),
    ],
)
def test_match_obj_type(type_or_obj, attribute_type):
    """Test that ``match_obj_type`` returns the expected attribute
    type for each argument."""
    assert match_obj_type(type_or_obj) is attribute_type


@pytest.mark.parametrize(
    "val, attribute_type",
    (
        ("", DatasetString),
        ("abc", DatasetString),
        (0, DatasetScalar),
        (0.0, DatasetScalar),
        (np.int64(0), DatasetScalar),
        (complex(1, 2), DatasetScalar),
        (None, DatasetNone),
        ([1, (2, 3)], DatasetList),
        ((1, [2, 3]), DatasetTuple),
    ),
)
def test_attribute(val, attribute_type):
    """Test that attribute() returns the correct attribute type
    for values."""
    dset_attr = attribute(val)

    assert isinstance(dset_attr, attribute_type)
    assert dset_attr.get_value() == val


class TestAttributeInfo:
    """Tests for ``AttributeInfo``."""

    @pytest.mark.parametrize(
        "kwargs, py_type",
        [
            ({"py_type": None}, "None"),
            ({"py_type": "None"}, "None"),
            ({"py_type": list[None]}, "list[None]"),
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
        x, y = AttributeInfo(py_type="list", data=1, extra="abc"), AttributeInfo(data=2)

        y.load(x)

        assert y.py_type == "list"
        assert y["data"] == 1
        assert y["extra"] == "abc"

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

    def test_getattr_none(self):
        """Test that __getattr__ returns None if the key
        does not exist."""

        info = AttributeInfo()
        assert info.foo is None


class NoDefaultAttribute(DatasetAttribute):
    """Example dataset attribute with no defined default value."""

    type_id = "test"

    def value_to_hdf5(
        self, bind_parent: HDF5Group, key: str, value: Any
    ) -> Any:  # pylint: disable=unused-argument
        return None

    def hdf5_to_value(self, bind: Any) -> Any:
        return None


class TestAttribute:
    """Tests for DatasetAttribute."""

    def test_init_no_default_value(self):
        """Test that an attribute class with no defined default
        will raise a ValueError if initialized with no arguments."""

        with pytest.raises(
            TypeError, match=r"__init__\(\) missing 1 required positional argument: 'value'"
        ):
            NoDefaultAttribute()

    def test_default_value_is_unset(self):
        """Test that the default_value() method returns UNSET if
        not overidden in a subclass."""

        assert NoDefaultAttribute.default_value() is UNSET

    def test_bind_init_from_invalid_bind(self):
        """Test that a ValueError is raised when the bind
        does not contain a dataset attribute."""

        with pytest.raises(ValueError, match="'bind' does not contain a dataset attribute."):
            DatasetNone(bind=create_group())

    def test_bind_init_from_other_bind(self):
        """Test that a TypeError is raised when the bind contains
        a dataset attribute of a different type."""

        attr = DatasetNone()

        with pytest.raises(TypeError, match="'bind' is bound to another attribute type 'none'"):
            DatasetString(bind=attr.bind)

    @pytest.mark.parametrize(
        "val, attribute_type",
        (
            ("", DatasetString),
            ("abc", DatasetString),
            (0, DatasetScalar),
            (0.0, DatasetScalar),
            (np.int64(0), DatasetScalar),
            (complex(1, 2), DatasetScalar),
            (None, DatasetNone),
        ),
    )
    def test_repr(self, val, attribute_type):
        """Test that __repr__ has the expected format."""
        with np.printoptions(legacy="1.21"):
            assert repr(attribute(val)) == f"{attribute_type.__name__}({repr(val)})"

    @pytest.mark.parametrize(
        "val",
        (
            "",
            "abc",
            0,
            0.0,
            np.int64(0),
            complex(1, 2),
            None,
        ),
    )
    def test_str(self, val):
        """Test that __str__ returns the string representation of the value"""

        assert str(attribute(val)) == str(val)

    @pytest.mark.parametrize("copy_func", [copy, deepcopy])
    @pytest.mark.parametrize(
        "val",
        (
            "abc",
            0,
            0.0,
            np.int64(0),
            complex(1, 2),
            None,
        ),
    )
    def test_copy_preserves_values(self, copy_func, val):
        """Test that copy preserves values"""

        dset_attr = attribute(val)
        assert copy_func(dset_attr) == dset_attr

    def test_abstract_subclass_not_registered(self):  # pylint: disable=unused-variable
        """Test that a DatasetAttribute subclass marked as
        abstract will not be registered."""

        class AbstractAttribute(
            DatasetAttribute, abstract=True
        ):  # pylint: disable=too-few-public-methods
            """An abstract attribute."""

            type_id = "_abstract_test_"

        assert "_abstract_test_" not in DatasetAttribute.registry

    def test_conflicting_type_id(self):
        """Test that a TypeError is raised if when a subclass of
        a DatasetAttribute has the same type id as another."""

        class Attribute(DatasetAttribute):  # pylint: disable=too-few-public-methods
            """An attribute"""

            type_id = "_attr_"

        with pytest.raises(
            TypeError, match=f"DatasetAttribute with type_id '_attr_' already exists: {Attribute}"
        ):

            class Conflicting(DatasetAttribute):  # pylint: disable=too-few-public-methods
                """A conflicting attribute"""

                type_id = "_attr_"

    def test_conflicting_consumes_types(self):
        """Test that a Warning is raised if an subclass captures the same
        type as another"""

        class MyType:  # pylint: disable=too-few-public-methods
            pass

        class Attribute(
            DatasetAttribute
        ):  # pylint: disable=unused-variable, too-few-public-methods
            """An attribute"""

            type_id = "_attr_2_"

            @classmethod
            def consumes_types(cls):
                return (MyType,)

        with pytest.warns(
            Warning,
            match="Conflicting default types: Both 'Conflicting' and 'Attribute' consume type 'MyType'. 'MyType' will now be consumed by 'Conflicting'",
        ):

            class Conflicting(DatasetAttribute):  # pylint: disable=too-few-public-methods
                """A conflicting attribute"""

                type_id = "_attr_3_"

                @classmethod
                def consumes_types(cls):
                    return (MyType,)

            assert match_obj_type(MyType) is Conflicting
