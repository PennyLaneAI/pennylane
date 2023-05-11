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
from typing import Tuple, Type, cast

from pennylane.data.attributes.list import DatasetList
from pennylane.data.base.typing_util import ZarrGroup
from pennylane.wires import Wires


class DatasetWires(DatasetList):
    """The ``DatasetWires`` class implements serialization for ``pennylane.wires.Wires``
    objects.

    Note that the elements of the ``Wires`` instance must all be serializable. This
    includes numbers and strings.
    """

    type_id = "wires"

    @classmethod
    def consumes_types(cls) -> Tuple[Type[Wires]]:
        return (Wires,)

    def zarr_to_value(self, bind: ZarrGroup) -> Wires:
        return Wires(self)

    def get_value(self) -> Wires:
        return cast(Wires, super().get_value())

    def copy_value(self) -> Wires:
        return Wires(super().copy_value())
