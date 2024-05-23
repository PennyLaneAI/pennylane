import json
from collections.abc import Callable
from typing import Any, Literal, Optional, Union, overload

from pennylane.typing import JSON
from pennylane.wires import Wires

from .pytrees import PyTreeStructure, get_typename, get_typename_type, leaf


@overload
def pytree_structure_dump(
    root: PyTreeStructure, *, indent: Optional[int] = None, encode: Literal[True]
) -> bytes: ...


@overload
def pytree_structure_dump(
    root: PyTreeStructure, *, indent: Optional[int] = None, encode: Literal[False] = False
) -> str: ...


def pytree_structure_dump(
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
    jsoned = _jsonify_pytree_structure(root)
    dump_args = {"indent": indent} if indent else {"separators": (",", ":")}
    if json_default:
        dump_args["default"] = _wrap_user_json_default(json_default)
    else:
        dump_args["default"] = _json_default

    data = json.dumps(jsoned, **dump_args)

    if encode:
        return data.encode("utf-8")

    return data


def _jsonify_pytree_structure(root: PyTreeStructure) -> list[JSON]:
    """Convert Pytree structure at ``root`` into a JSON-able representation."""
    if root.is_leaf:
        raise ValueError("Cannot dump Pytree: root node may not be a leaf")

    jsoned: list[Any] = [get_typename(root.type_), root.metadata, list(root.children)]

    todo: list[list[Union[PyTreeStructure, None]]] = [jsoned[2]]

    while todo:
        curr = todo.pop()

        for i in range(len(curr)):
            child = curr[i]
            if child.is_leaf:
                curr[i] = None
                continue

            child_list = list(child.children)
            curr[i] = [get_typename(child.type_), child.metadata, child_list]
            todo.append(child_list)

    return jsoned


def pytree_structure_load(data: str | bytes | bytearray) -> PyTreeStructure:
    """Load a previously serialized Pytree structure."""

    jsoned = json.loads(data)
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


def _json_default(obj: Any) -> JSON:
    """Default function for ``json.dump()``. Adds handling for the following types:
    - ``pennylane.wires.Wires``
    """
    if isinstance(obj, Wires):
        return obj.tolist()

    raise TypeError


def _wrap_user_json_default(user_default: Callable[[Any], JSON]) -> Callable[[Any], JSON]:
    """Wraps a user-provided JSON default function. If ``user_default`` raises a TypeError,
    calls ``_json_default``."""

    def _default_wrapped(obj: Any) -> JSON:
        try:
            return user_default(obj)
        except TypeError:
            return _json_default(obj)

    return _default_wrapped
