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
"""Tests for qml/qp namespace compatibility in serialized data."""

import numpy as np
import pytest

import pennylane as qp
from pennylane.data import DatasetPyTree
from pennylane.data.base.attribute import AttributeInfo
from pennylane.pytrees.pytrees import get_typename, get_typename_type

pytestmark = pytest.mark.data

pytest.importorskip("h5py")


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


def test_attribute_info_updates_existing_legacy_key_without_duplicate():
    """Test that writes update an existing legacy key instead of duplicating it."""
    attrs = {"qml.data.type_id": "scalar", "qml.__data_len__": 1}
    info = AttributeInfo(attrs)

    info["type_id"] = "array"

    assert attrs["qml.data.type_id"] == "array"
    assert "qp.data.type_id" not in attrs
    assert attrs["qp.__data_len__"] == 1
    assert "qml.__data_len__" not in attrs


def test_attribute_info_deduplicates_namespaced_keys_on_write():
    """Test that duplicated qml/qp metadata keys are treated as one logical field."""
    attrs = {
        "qml.data.doc": "legacy docs",
        "qp.data.doc": "new docs",
        "qml.__data_len__": 2,
    }
    info = AttributeInfo(attrs)

    assert len(info) == 1

    info["doc"] = "updated docs"

    doc_keys = [key for key in attrs if key.endswith(".data.doc")]
    assert doc_keys == ["qp.data.doc"]
    assert attrs["qp.data.doc"] == "updated docs"
    assert attrs["qp.__data_len__"] == 1


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


def test_pytree_typename_lookup_accepts_qml_and_qp_namespaces():
    """Test that pytree type lookup accepts either PennyLane namespace."""
    assert get_typename_type("qml.RX") is qp.RX
    assert get_typename_type("qp.RX") is qp.RX


def test_dataset_pytree_loads_alternate_namespace_treedef():
    """Test that serialized pytrees can be loaded after the qml/qp namespace switch."""
    value = qp.RX(0.1, wires=0)
    attr = DatasetPyTree(value)
    canonical_typename = get_typename(type(value)).encode("utf-8")
    alternate_typename = b"qp.RX" if canonical_typename.startswith(b"qml.") else b"qml.RX"

    treedef = attr.bind["treedef"][()].tobytes()
    assert canonical_typename in treedef

    del attr.bind["treedef"]
    attr.bind["treedef"] = np.void(treedef.replace(canonical_typename, alternate_typename))

    qp.assert_equal(DatasetPyTree(bind=attr.bind).get_value(), value)
