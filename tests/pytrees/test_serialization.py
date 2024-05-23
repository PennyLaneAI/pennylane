import json

import pytest

import pennylane as qml
from pennylane.measurements.shots import Shots
from pennylane.ops import PauliX, Prod, Sum
from pennylane.pytrees import (
    PyTreeStructure,
    flatten,
    is_pytree,
    leaf,
    list_pytree_types,
    register_pytree,
    unflatten,
)
from pennylane.pytrees.pytrees import (
    flatten_registrations,
    type_to_typename,
    typename_to_type,
    unflatten_registrations,
)
from pennylane.pytrees.serialization import pytree_structure_dump, pytree_structure_load
from pennylane.wires import Wires


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
    register_pytree(CustomNode, flatten_custom, unflatten_custom, namespace="test")

    yield

    del flatten_registrations[CustomNode]
    del unflatten_registrations[CustomNode]
    del typename_to_type[type_to_typename[CustomNode]]
    del type_to_typename[CustomNode]


def test_list_pytree_types():
    """Test for ``list_pytree_types()``."""
    assert list(list_pytree_types("test")) == [CustomNode]


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


def test_structure_load():
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


def test_nested_pl_object_roundtrip():
    tape_in = qml.tape.QuantumScript(
        [qml.adjoint(qml.RX(0.1, wires=0))],
        [qml.expval(2 * qml.X(0))],
        shots=50,
        trainable_params=(0, 1),
    )

    data, struct = flatten(tape_in)
    tape_out = unflatten(data, pytree_structure_load(pytree_structure_dump(struct)))

    assert type(tape_out) == type(tape_in)
    assert repr(tape_out) == repr(tape_in)
    assert list(tape_out) == list(tape_in)
