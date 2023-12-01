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
"""Contains utility function for converting ``Wires`` objects to JSON."""

import json
import numbers
from typing import Any

from pennylane.wires import Wires


class UnserializableWireError(TypeError):
    """Raised if a wire label is not JSON-serializable."""

    def __init__(self, wire: Any) -> None:
        super().__init__(
            f"Cannot serialize wire label '{wire}': Type '{type(wire)}' is not json-serializable."
        )


_JSON_TYPES = {int, str, float, type(None), bool}


def wires_to_json(wires: Wires) -> str:
    """Converts ``wires`` to a JSON list, with wire labels in
    order of their index.

    Returns:
        JSON list of wires

    Raises:
        UnserializableWireError: if any of the wires are not JSON-serializable.
    """
    jsonable_wires = []
    for w in wires:
        if type(w) in _JSON_TYPES:
            jsonable_wires.append(w)
        elif isinstance(w, numbers.Integral):
            w_converted = int(w)
            if hash(w_converted) != hash(w):
                raise UnserializableWireError(w)
            jsonable_wires.append(w_converted)
        else:
            raise UnserializableWireError(w)

    return json.dumps(jsonable_wires)
