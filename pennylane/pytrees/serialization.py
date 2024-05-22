import json
from typing import Any, Union, overload, Literal
from .pytrees import get_typename, Leaf, Structure, typenames_to_type
from pennylane.wires import Wires


@overload
def pytree_def_to_json(struct: Structure, *, pretty: bool = False, encode: Literal[True]) -> bytes:
    ...

@overload
def pytree_def_to_json(struct: Structure, *, pretty: bool = False, encode: Literal[False] = False) -> str:
    ...

def pytree_def_to_json(struct: Structure, *, pretty: bool = False, encode: bool = False) -> Union[bytes, str]:
    jsoned = _jsonify_struct(struct)

    if pretty:
        data = json.dumps(jsoned, indent=2, default=_json_default)
    else:
        data = json.dumps(jsoned, separators=(",", ":"), default=_json_default)

    if encode:
        return data.encode('utf-8')
    
    return data

def _jsonify_struct(root: Structure) -> list[Any]:
    jsoned: list[Any] = [get_typename(root.type), root.metadata, list(root.children)]

    todo: list[list[Union[Structure, Leaf]]] = [jsoned[2]]

    while todo:
        curr = todo.pop()

        for i in range(len(curr)):
            child = curr[i]
            if isinstance(child, Leaf):
                curr[i] = "Leaf"
                continue
            
            child_list = list(child.children)
            curr[i] = [get_typename(child.type), child.metadata, child_list]
            todo.append(child_list)

    return jsoned

def pytree_def_from_json(data: str | bytes | bytearray) -> Structure:
    jsoned = json.loads(data)

    root = Structure(
        typenames_to_type[jsoned[0]], jsoned[1], jsoned[2]
    )

    todo: list[list[Any]] = [root.children]

    while todo:
        curr = todo.pop()

        for i in range(len(curr)):
            child = curr[i]
            if child == "Leaf":
                curr[i] = Leaf()
                continue

            curr[i] = Structure(
                typenames_to_type[child[0]], child[1], child[2] 
            )
            todo.append(child[2])

    return root


def _json_default(o: Any):
    if isinstance(o, Wires):
        return o.tolist()
    
    raise TypeError
