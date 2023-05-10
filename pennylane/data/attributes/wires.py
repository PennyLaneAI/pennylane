import typing


from pennylane.wires import Wires
from pennylane.data.base.attribute import AttributeType, AttributeInfo
from pennylane.data.base.typing_util import ZarrGroup
from typing import List, Tuple, cast, Type
from pennylane.data.attributes.list import DatasetList


class DatasetWires(DatasetList):
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
