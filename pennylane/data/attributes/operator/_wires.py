import json
import numbers
from typing import Any

from pennylane.wires import Wires


class UnserializableWireError(TypeError):
    def __init__(self, wire: Any) -> None:
        super().__init__(
            f"Cannot serialize wire label '{wire}': Type '{type(wire)}' is not json-serializable or cannot be converted."
        )


_JSON_TYPES = {int, str, float, type(None), bool}


def wires_to_json(wires: Wires) -> str:
    jsonable_wires = []
    for w in wires:
        if type(w) not in _JSON_TYPES:
            if isinstance(w, numbers.Integral):
                w_converted = int(w)
                if hash(w_converted) != hash(w):
                    raise UnserializableWireError(w)

                jsonable_wires.append(w_converted)
        else:
            jsonable_wires.append(w)

    return json.dumps(jsonable_wires)
