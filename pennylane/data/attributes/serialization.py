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
An internal module for serializing and deserializing Pennylane pytrees.
"""

import json
from collections.abc import Callable
from typing import Any, Literal, overload

from pennylane.measurements.shots import Shots
from pennylane.pytrees.pytrees import PyTreeStructure, get_typename, get_typename_type, leaf
from pennylane.typing import JSON
from pennylane.wires import Wires


@overload
def pytree_structure_dump(
    root: PyTreeStructure, *, indent: int | None = None, decode: Literal[False] = False
) -> bytes: ...


@overload
def pytree_structure_dump(
    root: PyTreeStructure, *, indent: int | None = None, decode: Literal[True]
) -> str: ...


def pytree_structure_dump(
    root: PyTreeStructure, *, indent: int | None = None, decode: bool = False
) -> bytes | str:
    """Convert Pytree structure ``root`` into JSON.

    A non-leaf structure is represented as a 3-element list. The first element will
    be the type name, the second element metadata, and the third element is
    the list of children.

    A leaf structure is represented by `null`.

    Metadata may contain ``pennylane.Shots`` and ``pennylane.Wires`` objects,
    as well as any JSON-serializable data.

    >>> from pennylane.pytrees import PyTreeStructure, leaf, flatten
    >>> from pennylane.pytrees.serialization import pytree_structure_dump

    >>> _, struct = flatten([{"a": 1}, 2])
    >>> struct
    PyTreeStructure(list, None, [dict, ("a",), [PyTreeStructure()]), PyTreeStructure()])'

    >>> pytree_structure_dump(struct)
    b'["builtins.list",null,[["builtins.dict",["a"],[null]],null]]'

    Args:
        root: Root of a Pytree structure
        indent: If not None, the resulting JSON will be pretty-printed with the
            given indent level. Otherwise, the output will use the most compact
            possible representation
        decode: If True, return a string instead of bytes

    Returns:
        bytes: If ``encode`` is True
        str: If ``encode`` is False
    """
    dump_args = {"indent": indent} if indent else {"separators": (",", ":")}

    data = json.dumps(root, default=_json_default, **dump_args)

    if not decode:
        return data.encode("utf-8")

    return data


def pytree_structure_load(data: str | bytes | bytearray) -> PyTreeStructure:
    """Load a previously serialized Pytree structure.

    >>> from pennylane.pytrees.serialization import pytree_structure_dump

    >>> pytree_structure_load('["builtins.list",null,[["builtins.dict",["a"],[null]],null]')
    PyTreeStructure(list, None, [PyTreeStructure(dict, ["a"], [PyTreeStructure()]), PyTreeStructure()])'
    """
    jsoned = json.loads(data)
    root = PyTreeStructure(get_typename_type(jsoned[0]), jsoned[1], jsoned[2])

    # List of serialized child structures that will be de-serialized in place
    todo: list[list[Any]] = [root.children]
    while todo:
        curr = todo.pop()

        for i, child in enumerate(curr):
            if child is None:
                curr[i] = leaf
                continue

            curr[i] = PyTreeStructure(get_typename_type(child[0]), child[1], child[2])

            todo.append(child[2])

    return root


def _pytree_structure_to_json(obj: PyTreeStructure) -> JSON:
    """JSON handler for serializating ``PyTreeStructure``."""
    if obj.is_leaf:
        return None

    return [get_typename(obj.type_), obj.metadata, obj.children]


def _wires_to_json(obj: Wires) -> JSON:
    """JSON handler for serializing ``Wires``."""
    return obj.tolist()


def _shots_to_json(obj: Shots) -> JSON:
    """JSON handler for serializing ``Shots``."""
    if obj.total_shots is None:
        return None

    return obj.shot_vector


_json_handlers: dict[type, Callable[[Any], JSON]] = {
    PyTreeStructure: _pytree_structure_to_json,
    Wires: _wires_to_json,
    Shots: _shots_to_json,
}


def _json_default(obj: Any) -> JSON:
    """Default function for ``json.dump()``. Calls the handler for the type of ``obj``
    in ``_json_handlers``. Raises ``TypeError`` if ``obj`` cannot be handled.
    """
    try:
        return _json_handlers[type(obj)](obj)
    except KeyError as exc:
        raise TypeError(f"Could not serialize metadata object: {repr(obj)}") from exc
