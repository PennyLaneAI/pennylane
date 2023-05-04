from numbers import Number

import numpy
import zarr
from numpy.typing import ArrayLike

from pennylane.data.base.attribute import AttributeType


class DatasetArray(AttributeType[zarr.Array, ArrayLike]):
    type_id = "array"

    def zarr_to_value(self, bind: zarr.Array) -> ArrayLike:
        return numpy.array(self.bind, dtype=bind.dtype, order=bind.order)

    def value_to_zarr(self, bind_parent: zarr.Group, key: str, value: ArrayLike) -> zarr.Array:
        bind_parent[self.key] = value
        return bind_parent[self.key]


class DatasetScalar(AttributeType[zarr.Array, Number]):
    type_id = "scalar"

    def zarr_to_value(self, bind: zarr.Array) -> Number:
        return bind[()]

    def value_to_zarr(self, bind_parent: zarr.Group, key: str, value: Number) -> zarr.Array:
        bind_parent[key] = value

        return bind_parent[key]
