from collections.abc import Iterator, Mapping, MutableMapping
from typing import Generic, TypeVar, Union

import zarr
import zarr.convenience

from pennylane.data.base.attribute import AttributeType
from pennylane.data.base.mapper import MapperMixin
from pennylane.data.base.typing_util import T, ZarrAny


class DatasetDict(
    Generic[T],
    AttributeType[zarr.Group, Mapping[str, T]],
    MutableMapping[str, T],
    MapperMixin,
):
    Self = TypeVar("Self", bound="DatasetDict")

    type_id = "dict"

    def default_value(self) -> Mapping[str, T]:
        return {}

    def zarr_to_value(self, bind: zarr.Group) -> MutableMapping[str, T]:
        return self

    def value_to_zarr(
        self, bind_parent: zarr.Group, key: str, value: Mapping[str, T]
    ) -> zarr.Group:
        grp = bind_parent.create_group(key)

        self.update(value)

        return grp

    def __getitem__(self, __key: str) -> T:
        return self._mapper[__key].get_value()

    def __setitem__(self, __key: str, __value: Union[T, AttributeType[ZarrAny, T]]) -> None:
        self._mapper[__key] = __value

    def __delitem__(self, __key: str) -> None:
        del self._mapper[__key]

    def __len__(self) -> int:
        return len(self.bind)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, Mapping):
            return False

        if not len(self) == len(__value):
            return False

        return all(
            my_item == other_item for my_item, other_item in zip(self.items(), __value.items())
        )

    def __iter__(self) -> Iterator[str]:
        return (key for key in self.bind.keys())

    def __str__(self) -> str:
        return str(dict(self))

    def __repr__(self):
        return repr(dict(self))
