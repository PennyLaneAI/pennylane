import json

import pytest

from pennylane.pytrees import PyTreeStructure, flatten, leaf, register_pytree
from pennylane.pytrees.serialization import pytree_structure_dump, pytree_structure_load
from pennylane.wires import Wires


class CustomNode:

    def __init__(self, data, metadata):
        self.data = data
        self.metadata = metadata


def flatten_custom(node):
    return (node.data, node.metadata)


def unflatten_custom(data, metadata):
    return CustomNode(data, metadata)


register_pytree(CustomNode, flatten_custom, unflatten_custom, typename_prefix="test")


def test_structure_dump():
    _, struct = flatten(
        {
            "list": ["a", 1],
            "dict": {"a": 1},
            "tuple": ("a", 1),
            "custom": CustomNode([1, 5, 7], {"wires": Wires([1, "a", 3.4, None])}),
        }
    )

    assert pytree_structure_dump(struct) == [
        "builtins.dict",
        ("list", "dict", "tuple", "custom"),
        [
            ["builtins.list", None, [None, None]],
            ["builtins.dict", ("a",), [None]],
            ["builtins.tuple", None, [None, None]],
            ["test.CustomNode", {"wires": Wires([1, "a", 3.4, None])}, [None, None, None]],
        ],
    ]


@pytest.mark.parametrize("string", [True, False])
def test_structure_load(string):
    jsoned = [
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
    if string:
        jsoned = json.dumps(jsoned)

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
