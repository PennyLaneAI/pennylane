import json
from collections.abc import Callable
from typing import Any, Literal, Optional, Union, overload

from pennylane.typing import JSON
from pennylane.wires import Wires

from .pytrees import PyTreeStructure, get_typename, get_typename_type, leaf


@overload
def pytree_structure_dump(
    root: PyTreeStructure, *, indent: Optional[int] = None, decode: Literal[False] = False
) -> bytes: ...


@overload
def pytree_structure_dump(
    root: PyTreeStructure, *, indent: Optional[int] = None, decode: Literal[True]
) -> str: ...


def pytree_structure_dump(
    root: PyTreeStructure,
    *,
    indent: Optional[int] = None,
    decode: bool = False,
    json_default: Optional[Callable[[Any], JSON]] = None,
) -> Union[bytes, str]:
    """Convert Pytree structure ``root`` into JSON.

    A non-leaf structure is represented as a 3-element list. The first element will
    be the type name, the second element metadata, and the third element is
    the list of children.

    A leaf structure is represented by `null`.

    Metadata can only contain ``pennylane.Wires`` objects, JSON-serializable
    data or objects that can be handled by ``json_default`` if provided.

    >>> from pennylane.pytrees import PyTreeStructure, leaf, flatten
    >>> from pennylane.pytrees.serialization import pytree_structure_dump

    >>> _, struct = flatten([{"a": 1}, 2])
    >>> struct
    'PyTreeStructure(<class 'list'>, None, [PyTreeStructure(<class 'dict'>, ("a",), [PyTreeStructure()]), PyTreeStructure()])'

    >>> pytree_structure_dump(struct)
    b'["builtins.list",null,[["builtins.dict",["a"],[null]],null]'

    Args:
        root: Root of a Pytree structure
        indent: If not None, the resulting JSON will be pretty-printed with the
            given indent level. Otherwise, the output will use the most compact
            possible representation
        decode: If True, return a string instead of bytes
        json_default: Handler for objects that can't otherwise be serialized. Should
            return a JSON-compatible value or raise a ``TypeError`` if the value
            can't be handled

    Returns:
        bytes: If ``encode`` is True
        str: If ``encode`` is False
    """
    dump_args = {"indent": indent} if indent else {"separators": (",", ":")}
    if json_default:
        dump_args["default"] = _wrap_user_json_default(json_default)
    else:
        dump_args["default"] = _json_default

    data = json.dumps(root, **dump_args)

    if not decode:
        return data.encode("utf-8")

    return data


def pytree_structure_load(data: str | bytes | bytearray) -> PyTreeStructure:
    """Load a previously serialized Pytree structure.

    >>> from pennylane.pytrees.serialization import pytree_structure_dump

    >>> pytree_structure_load('["builtins.list",null,[["builtins.dict",["a"],[null]],null]')
    'PyTreeStructure(<class 'list'>, None, [PyTreeStructure(<class 'dict'>, ["a"], [PyTreeStructure()]), PyTreeStructure()])'
    """
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

            # Child structures will be converted in place
            todo.append(child[2])

    return root


def _json_default(obj: Any) -> JSON:
    """Default function for ``json.dump()``. Adds handling for the following types:
    - ``pennylane.pytrees.PyTreeStructure``
    - ``pennylane.wires.Wires``
    """
    if isinstance(obj, PyTreeStructure):
        if obj.is_leaf:
            return None
        return [get_typename(obj.type_), obj.metadata, obj.children]

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
