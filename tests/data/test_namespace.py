# Copyright 2018-2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for namespace compatibility in serialized data."""

import numpy as np
import pytest

import pennylane as qp
from pennylane.data import DatasetList, DatasetPyTree, DatasetScalar
from pennylane.data.attributes.serialization import _get_typename_type
from pennylane.data.base.attribute import AttributeInfo
from pennylane.pytrees.pytrees import get_typename

pytestmark = pytest.mark.data

pytest.importorskip("h5py")


def _attr_namespace(attrs, old_namespace, new_namespace):
    """Rewrite data attribute metadata keys to simulate older serialized files."""
    for key, value in list(attrs.items()):
        if key.startswith(f"{old_namespace}.data."):
            del attrs[key]
            attrs[f"{new_namespace}.data.{key.removeprefix(f'{old_namespace}.data.')}"] = value
        elif key == f"{old_namespace}.__data_len__":
            del attrs[key]
            attrs[f"{new_namespace}.__data_len__"] = value


def _assert_namespace(attrs, *names):
    assert "qml.__data_len__" not in attrs
    for name in names:
        assert f"qml.data.{name}" not in attrs
        assert f"qp.data.{name}" in attrs


def test_attribute_info_reads_legacy_namespace():
    """Test that metadata stored with the old namespace can still be read."""
    attrs = {
        "qml.data.type_id": "scalar",
        "qml.data.py_type": "float",
        "qml.__data_len__": 2,
    }

    info = AttributeInfo(attrs)

    assert info["type_id"] == "scalar"
    assert info.py_type == "float"
    assert len(info) == 2
    assert set(info) == {"type_id", "py_type"}


def test_attribute_info_copies_legacy_metadata_to_current_namespace():
    """Test that metadata copied from legacy attrs is written with the current namespace."""
    legacy_info = AttributeInfo(
        {
            "qml.data.type_id": "scalar",
            "qml.data.py_type": "float",
            "qml.data.doc": "legacy docs",
            "qml.__data_len__": 3,
        }
    )
    current_attrs = {}

    AttributeInfo(current_attrs).load(legacy_info)

    assert dict(AttributeInfo(current_attrs)) == {
        "type_id": "scalar",
        "py_type": "float",
        "doc": "legacy docs",
    }
    _assert_namespace(current_attrs, "type_id", "py_type", "doc")
    assert current_attrs["qp.__data_len__"] == 3


def test_attribute_info_updates_existing_legacy_key_without_duplicate():
    """Test that writes update an existing legacy key instead of duplicating it."""
    attrs = {"qml.data.type_id": "scalar", "qml.__data_len__": 1}
    info = AttributeInfo(attrs)

    info["type_id"] = "array"
    assert "qml.data.type_id" not in attrs
    assert attrs["qp.data.type_id"] == "array"
    assert len(info) == 1

    info["data_interface"] = "numpy"
    assert len(info) == 2
    assert "qml.__data_len__" not in attrs


def test_attribute_info_save_updates_legacy_attrs_without_duplicate_fields():
    """Test that saving extra metadata into legacy attrs preserves logical fields."""
    attrs = {
        "qml.data.type_id": "scalar",
        "qml.data.doc": "legacy docs",
        "qml.__data_len__": 2,
    }

    AttributeInfo(doc="updated docs", data_interface="numpy").save(AttributeInfo(attrs))

    info = AttributeInfo(attrs)
    assert dict(info) == {
        "type_id": "scalar",
        "doc": "updated docs",
        "data_interface": "numpy",
    }
    _assert_namespace(attrs, "doc", "data_interface")
    assert len(info) == 3


def test_attribute_info_deduplicates_namespaced_keys_on_write():
    """Test that duplicated metadata keys are treated as one logical field."""
    attrs = {
        "qml.data.doc": "legacy docs",
        "qp.data.doc": "new docs",
        "qml.__data_len__": 1,
    }
    info = AttributeInfo(attrs)
    assert info["doc"] == "new docs"

    info["doc"] = "updated docs"
    doc_keys = [key for key in attrs if key.endswith(".data.doc")]

    assert len(info) == 1
    assert doc_keys == ["qp.data.doc"]
    assert info["doc"] == "updated docs"


def test_attribute_info_deletes_all_namespaced_duplicates():
    """Test that deleting a field removes all namespace variants."""
    attrs = {
        "qml.data.doc": "legacy docs",
        "qp.data.doc": "new docs",
        "qp.data.type_id": "scalar",
        "qml.__data_len__": 2,
    }
    info = AttributeInfo(attrs)

    del info["doc"]

    assert set(info) == {"type_id"}
    assert not any(key.endswith(".data.doc") for key in attrs)
    assert attrs["qp.__data_len__"] == 1
    assert "qml.__data_len__" not in attrs


def test_attribute_info_delete_missing_key_raises_key_error():
    """Test that deleting missing metadata keeps normal mapping behavior."""
    info = AttributeInfo({})

    with pytest.raises(KeyError, match="doc"):
        del info["doc"]


def test_attribute_info_setting_missing_key_to_none_is_noop():
    """Test that removing absent metadata does not create namespace bookkeeping."""
    attrs = {}
    info = AttributeInfo(attrs)

    info["doc"] = None

    assert attrs == {}
    assert len(info) == 0


def test_attribute_info_setting_existing_key_to_none_removes_metadata():
    """Test that assigning None removes existing namespaced metadata."""
    attrs = {"qml.data.doc": "legacy docs", "qml.__data_len__": 1}
    info = AttributeInfo(attrs)

    info["doc"] = None

    assert "qml.data.doc" not in attrs
    assert "qp.data.doc" not in attrs
    assert attrs["qp.__data_len__"] == 0
    assert "qml.__data_len__" not in attrs
    assert len(info) == 0


def test_dataset_attribute_bind_init_reads_legacy_attribute_metadata():
    """Test that DatasetAttribute bind initialization can read legacy metadata keys."""
    attr = DatasetScalar(1.5)
    _attr_namespace(attr.bind.attrs, "qp", "qml")

    loaded_attr = DatasetScalar(bind=attr.bind)

    assert loaded_attr.get_value() == 1.5
    assert loaded_attr.info["type_id"] == "scalar"
    assert loaded_attr.info.py_type == "float"


def test_dataset_attribute_info_updates_legacy_metadata_on_existing_bind():
    """Test that metadata updates through DatasetAttribute.info normalize legacy keys."""
    attr = DatasetScalar(1.5)
    _attr_namespace(attr.bind.attrs, "qp", "qml")

    loaded_attr = DatasetScalar(bind=attr.bind)
    loaded_attr.info.doc = "updated docs"

    assert loaded_attr.info.doc == "updated docs"
    _assert_namespace(attr.bind.attrs, "doc")
    assert AttributeInfo(attr.bind.attrs)["type_id"] == "scalar"


def test_nested_attribute_reads_legacy_child_metadata():
    """Test that collection of attributes can read children with legacy metadata."""
    attr = DatasetList([1, "two"])
    _attr_namespace(attr.bind["0"].attrs, "qp", "qml")
    _attr_namespace(attr.bind["1"].attrs, "qp", "qml")

    loaded_attr = DatasetList(bind=attr.bind)

    assert loaded_attr.copy_value() == [1, "two"]
    assert loaded_attr[0] == 1
    assert loaded_attr[1] == "two"


def test_dataset_pytree_loads_alternate_namespace_treedef():
    """Test that serialized pytree attributes load after the namespace switch."""
    value = [qp.RX(0.1, wires=0), qp.adjoint(qp.RY(0.2, wires=1))]
    attr = DatasetPyTree(value)

    treedef = attr.bind["treedef"][()].tobytes()
    canonical_typename = get_typename(qp.RX).encode("utf-8")
    current_namespace = canonical_typename.split(b".", maxsplit=1)[0]
    alternate_namespace = b"qml" if current_namespace == b"qp" else b"qp"
    assert current_namespace + b"." in treedef

    del attr.bind["treedef"]
    attr.bind["treedef"] = np.void(
        treedef.replace(current_namespace + b".", alternate_namespace + b".")
    )

    loaded_value = DatasetPyTree(bind=attr.bind).get_value()

    assert isinstance(loaded_value, list)
    qp.assert_equal(loaded_value[0], value[0])
    qp.assert_equal(loaded_value[1], value[1])


def test_pytree_typename_fallback_reraises_unresolved_type():
    """Test that unresolved pytree type names still raise after namespace fallback."""
    with pytest.raises(ValueError, match="'not.a.typename' is not the name of a Pytree type."):
        _get_typename_type("not.a.typename")
