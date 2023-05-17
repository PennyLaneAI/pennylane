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
"""Contains AttributeType definition for ``pennylane.wires.Wires`` objects."""
from typing import Tuple, Type

from pennylane.data.base.attribute import AttributeType
from pennylane.data.base.typing_util import ZarrArray, ZarrGroup
from pennylane.wires import Wires
from typing import Any, Union
import numbers
import json

_JSON_TYPES = {int, str, float, type(None), bool}
JsonType = Union[int, str, float, None, bool]


class UnserializableWireError(TypeError):
    def __init__(self, wire: Any) -> None:
        super().__init__(
            f"Cannot serialize wire label '{wire}': Type '{type(wire)}' is not json-serializable or cannot be converted."
        )


class DatasetWires(AttributeType[ZarrArray, Wires, Wires]):
    """The ``DatasetWires`` class implements serialization for ``pennylane.wires.Wires``
    objects.

    Note that the elements of the ``Wires`` instance must all be JSON serializable. This
    includes integers and strings.
    """

    type_id = "wires"

    @classmethod
    def consumes_types(cls) -> Tuple[Type[Wires]]:
        return (Wires,)

    def zarr_to_value(self, bind: ZarrArray) -> Wires:
        return Wires(json.loads(bind.asstr()[()]))

    def value_to_zarr(self, bind_parent: ZarrGroup, key: str, value: Wires) -> ZarrArray:
        json_wires = []
        for w in value:
            if type(w) not in _JSON_TYPES:
                if isinstance(w, numbers.Integral):
                    w_converted = int(w)
                    if hash(w_converted) != hash(w):
                        raise UnserializableWireError(w)

                    w = w_converted

            json_wires.append(w)

        bind_parent[key] = json.dumps(json_wires)

        return bind_parent[key]
