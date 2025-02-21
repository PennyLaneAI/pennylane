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
Tests for the ``DatasetPyTree`` attribute type.
"""

from dataclasses import dataclass

import pytest

import pennylane as qml
from pennylane.data import Dataset, DatasetPyTree
from pennylane.pytrees.pytrees import (
    _register_pytree_with_pennylane,
    flatten_registrations,
    type_to_typename,
    typename_to_type,
    unflatten_registrations,
)


@dataclass
class CustomNode:
    """Example Pytree for testing."""

    def __init__(self, data, metadata):
        self.data = data
        self.metadata = metadata


def flatten_custom(node):
    return (node.data, node.metadata)


def unflatten_custom(data, metadata):
    return CustomNode(data, metadata)


@pytest.fixture(autouse=True)
def register_test_node():
    """Fixture that temporarily registers the ``CustomNode`` class as
    a Pytree."""
    # Use this instead of ``register_pytree()`` so that ``CustomNode`` will not
    # be registered with jax.
    _register_pytree_with_pennylane(CustomNode, "test.CustomNode", flatten_custom, unflatten_custom)

    yield

    del flatten_registrations[CustomNode]
    del unflatten_registrations[CustomNode]
    del typename_to_type[type_to_typename[CustomNode]]
    del type_to_typename[CustomNode]


class TestDatasetPyTree:
    """Tests for ``DatasetPyTree``."""

    def test_consumes_type(self):
        """Test that PyTree-compatible types that are not builtin are
        consumed by ``DatasetPyTree``."""
        dset = Dataset()
        dset.attr = CustomNode([1, 2, 3, 4], {"meta": "data"})

        assert isinstance(dset.attrs["attr"], DatasetPyTree)

    @pytest.mark.parametrize("obj", [[1, 2], {"a": 1}, (1, 2)])
    def test_builtins_not_consumed(self, obj):
        """Test that built-in containers like dict, list and tuple are
        not consumed by the ``DatasetPyTree`` type."""

        dset = Dataset()
        dset.attr = obj

        assert not isinstance(dset.attrs["attr"], DatasetPyTree)

    def test_value_init(self):
        """Test that ``DatasetPyTree`` can be initialized from a value."""

        value = CustomNode(
            [{"a": 1}, (3, 5), [7, 9, {"x": CustomNode("data", None)}]], {"meta": "data"}
        )
        attr = DatasetPyTree(value)

        assert attr.type_id == "pytree"
        assert attr.get_value() == value

    def test_bind_init(self):
        """Test that a ``DatasetPyTree`` can be bind-initialized."""

        value = CustomNode(
            [{"a": 1}, (3, 5), [7, 9, {"x": CustomNode("data", None)}]], {"meta": "data"}
        )
        bind = DatasetPyTree(value).bind

        attr = DatasetPyTree(bind=bind)

        assert attr == value


@pytest.mark.parametrize("shots", [None, 1, [1, 2]])
def test_quantum_scripts(shots):
    """Test that ``QuantumScript`` can be serialized as Pytrees."""
    script = qml.tape.QuantumScript([qml.X(0)], shots=shots)

    qml.assert_equal(DatasetPyTree(script).get_value(), script)
