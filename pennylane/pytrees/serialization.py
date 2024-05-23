import json
from collections.abc import Callable
from typing import Any, Literal, Optional, Union, overload

from pennylane.typing import JSON
from pennylane.wires import Wires

from .pytrees import PyTreeStructure, get_typename, get_typename_type, leaf


@overload
def pytree_structure_dump_json(
    root: PyTreeStructure, *, indent: Optional[int] = None, encode: Literal[True]
) -> bytes: ...


@overload
def pytree_structure_dump_json(
    root: PyTreeStructure, *, indent: Optional[int] = None, encode: Literal[False] = False
) -> str: ...


def pytree_structure_dump_json(
    root: PyTreeStructure,
    *,
    indent: Optional[int] = None,
    encode: bool = False,
    json_default: Optional[Callable[[Any], JSON]] = None,
) -> Union[bytes, str]:
    """Convert Pytree structure ``root`` into JSON.

    Args:
        root: Root of a Pytree structure
        indent: If not None, the resulting JSON will be pretty-printed with the
            given indent level. Otherwise, the output will use the most compact
            possible representation
        encode: Whether to return the output as bytes

    Returns:
        bytes: If ``encode`` is True
        str: If ``encode`` is False

    """
    jsoned = pytree_structure_dump(root)

    if indent:
        data = json.dumps(jsoned, indent=indent, default=_json_default)
    else:
        data = json.dumps(jsoned, separators=(",", ":"), default=_json_default)

    if encode:
        return data.encode("utf-8")

    return data


def pytree_structure_dump(root: PyTreeStructure) -> list[JSON]:
    """Convert Pytree structure at ``root`` into a JSON-able representation."""
    if root.is_leaf:
        raise ValueError("Cannot dump Pytree: root node may not be a leaf")

    jsoned: list[Any] = [get_typename(root.type), root.metadata, list(root.children)]

    todo: list[list[Union[PyTreeStructure, None]]] = [jsoned[2]]

    while todo:
        curr = todo.pop()

        for i in range(len(curr)):
            child = curr[i]
            if child.is_leaf:
                curr[i] = None
                continue

            child_list = list(child.children)
            curr[i] = [get_typename(child.type), child.metadata, child_list]
            todo.append(child_list)

    return jsoned


def pytree_structure_load(data: str | bytes | bytearray | list[JSON]) -> PyTreeStructure:
    """Load a previously serialized Pytree structure."""
    if isinstance(data, (str, bytes, bytearray)):
        jsoned = json.loads(data)
    else:
        jsoned = data

    root = PyTreeStructure(get_typename_type(jsoned[0]), jsoned[1], jsoned[2])

    todo: list[list[Any]] = [root.children]

    while todo:
        curr = todo.pop()

        for i in range(len(curr)):
            child = curr[i]
            if child is None:
                curr[i] = leaf
                continue

            curr[i] = PyTreeStructure(get_typename_type(child[0]), child[1], child[2])

            todo.append(child[2])

    return root


def _json_default(default: Callable[[Any], JSON]):
    def default(o: Any):
        if isinstance(o, Wires):
            return o.tolist()

        return

    if isinstance(o, Wires):
        return o.tolist()

    raise TypeError
