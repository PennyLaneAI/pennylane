# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
Tests for the pytrees serialization module.
"""

import json
from typing import Any

import numpy as np
import pytest

import pennylane as qml
from pennylane.data.attributes.serialization import pytree_structure_dump, pytree_structure_load
from pennylane.measurements import Shots
from pennylane.ops import PauliX, Prod, Sum
from pennylane.pytrees import PyTreeStructure, flatten, is_pytree, leaf, unflatten
from pennylane.pytrees.pytrees import (
    _register_pytree_with_pennylane,
    flatten_registrations,
    type_to_typename,
    typename_to_type,
    unflatten_registrations,
)
from pennylane.wires import Wires


class CustomNode:
    """Example Pytree for testing."""

    # pylint: disable=too-few-public-methods

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


@pytest.mark.parametrize(
    "cls, result",
    [
        (CustomNode, True),
        (list, True),
        (tuple, True),
        (Sum, True),
        (Prod, True),
        (PauliX, True),
        (int, False),
        (set, False),
    ],
)
def test_is_pytree(cls, result):
    """Test for ``is_pytree()``."""
    assert is_pytree(cls) is result


@pytest.mark.parametrize("decode", [True, False])
def test_pytree_structure_dump(decode):
    """Test that ``pytree_structure_dump()`` creates JSON in the expected
    format."""
    _, struct = flatten(
        {
            "list": ["a", 1],
            "dict": {"a": 1},
            "tuple": ("a", 1),
            "custom": CustomNode([1, 5, 7], {"wires": Wires([1, "a", 3.4, None])}),
        }
    )

    assert json.loads(pytree_structure_dump(struct, decode=decode)) == [
        "builtins.dict",
        ["list", "dict", "tuple", "custom"],
        [
            ["builtins.list", None, [None, None]],
            [
                "builtins.dict",
                [
                    "a",
                ],
                [None],
            ],
            ["builtins.tuple", None, [None, None]],
            ["test.CustomNode", {"wires": [1, "a", 3.4, None]}, [None, None, None]],
        ],
    ]


@pytest.mark.parametrize(
    "shots, expect_metadata",
    [
        (Shots(), None),
        (Shots(1), [[1, 1]]),
        (Shots([1, 2]), [[1, 1], [2, 1]]),
    ],
)
def test_pytree_structure_dump_shots(shots, expect_metadata):
    """Test that ``pytree_structure_dump`` handles all forms of shots."""
    _, struct = flatten(CustomNode([], {"shots": shots}))

    flattened = pytree_structure_dump(struct)

    assert json.loads(flattened) == ["test.CustomNode", {"shots": expect_metadata}, []]


def test_pytree_structure_dump_unserializable_metadata():
    """Test that a ``TypeError`` is raised if a Pytree has unserializable metadata."""
    _, struct = flatten(CustomNode([1, 2, 4], {"operator": qml.PauliX(0)}))

    with pytest.raises(TypeError, match=r"Could not serialize metadata object: X\(0\)"):
        pytree_structure_dump(struct)


def test_pytree_structure_load():
    """Test that ``pytree_structure_load()`` can parse a JSON-serialized PyTree."""
    jsoned = json.dumps(
        [
            "builtins.dict",
            ["list", "dict", "tuple", "custom"],
            [
                ["builtins.list", None, [None, None]],
                [
                    "builtins.dict",
                    [
                        "a",
                    ],
                    [None],
                ],
                ["builtins.tuple", None, [None, None]],
                ["test.CustomNode", {"wires": [1, "a", 3.4, None]}, [None, None, None]],
            ],
        ]
    )

    assert pytree_structure_load(jsoned) == PyTreeStructure(
        dict,
        ["list", "dict", "tuple", "custom"],
        [
            PyTreeStructure(list, None, [leaf, leaf]),
            PyTreeStructure(dict, ["a"], [leaf]),
            PyTreeStructure(tuple, None, [leaf, leaf]),
            PyTreeStructure(CustomNode, {"wires": [1, "a", 3.4, None]}, [leaf, leaf, leaf]),
        ],
    )


H_ONE_QUBIT = np.array([[1.0, 0.5j], [-0.5j, 2.5]])
H_TWO_QUBITS = np.array(
    [[0.5, 1.0j, 0.0, -3j], [-1.0j, -1.1, 0.0, -0.1], [0.0, 0.0, -0.9, 12.0], [3j, -0.1, 12.0, 0.0]]
)


@pytest.mark.parametrize(
    "obj_in",
    [
        qml.tape.QuantumScript(
            [qml.adjoint(qml.RX(0.1, wires=0))],
            [qml.expval(2 * qml.X(0))],
            shots=50,
            trainable_params=[0, 1],
        ),
        Prod(qml.X(0), qml.RX(0.1, wires=0), qml.X(1), id="id"),
        Sum(
            qml.Hermitian(H_ONE_QUBIT, 2),
            qml.Hermitian(H_TWO_QUBITS, [0, 1]),
            qml.PauliX(1),
            qml.Identity("a"),
        ),
        qml.Hamiltonian(
            (1.1, -0.4, 0.333), (qml.PauliX(0), qml.Hermitian(H_ONE_QUBIT, 2), qml.PauliZ(2))
        ),
        qml.Hamiltonian(
            np.array([-0.1, 0.5]), [qml.Hermitian(H_TWO_QUBITS, [0, 1]), qml.PauliY(0)]
        ),
    ],
)
def test_pennylane_pytree_roundtrip(obj_in: Any):
    """Test that Pennylane Pytree objects are equal to themselves after
    a serialization roundtrip."""
    data, struct = flatten(obj_in)
    obj_out = unflatten(data, pytree_structure_load(pytree_structure_dump(struct)))

    qml.assert_equal(obj_in, obj_out)


@pytest.mark.parametrize(
    "obj_in",
    [
        [
            qml.tape.QuantumScript(
                [qml.adjoint(qml.RX(0.1, wires=0))],
                [qml.expval(2 * qml.X(0))],
                trainable_params=[0, 1],
            ),
            Prod(qml.X(0), qml.RX(0.1, wires=0), qml.X(1), id="id"),
            Sum(
                qml.Hermitian(H_ONE_QUBIT, 2),
                qml.Hermitian(H_TWO_QUBITS, [0, 1]),
                qml.PauliX(1),
                qml.Identity("a"),
            ),
        ]
    ],
)
def test_pennylane_pytree_roundtrip_list(obj_in: Any):
    """Test that lists Pennylane Pytree objects are equal to themselves after
    a serialization roundtrip."""
    data, struct = flatten(obj_in)
    obj_out = unflatten(data, pytree_structure_load(pytree_structure_dump(struct)))

    assert all(qml.equal(in_, out) for in_, out in zip(obj_in, obj_out))
