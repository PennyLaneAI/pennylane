import json

from pennylane.data.base.attribute import AttributeType
from pennylane.data.base.typing_util import JSON, HDF5Array, HDF5Group


class DatasetJSON(AttributeType[HDF5Array, JSON, JSON]):
    """Dataset type for JSON-serializable data."""

    type_id = "json"

    def hdf5_to_value(self, bind: HDF5Array) -> JSON:
        return json.loads(bind.asstr()[()])

    def value_to_hdf5(self, bind_parent: HDF5Group, key: str, value: JSON) -> HDF5Array:
        bind_parent[key] = json.dumps(value)

        return bind_parent[key]
