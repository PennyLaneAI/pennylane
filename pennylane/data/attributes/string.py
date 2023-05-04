import zarr

from pennylane.data.base.attribute import AttributeType


class DatasetString(AttributeType):
    type_id = "string"

    @classmethod
    def consumes_types(cls) -> tuple[type[str]]:
        return (str,)

    def zarr_to_value(self, bind: zarr.Array) -> str:
        return bind[()]

    def value_to_zarr(self, bind_parent: zarr.Group, key: str, value: str) -> zarr.Array:
        bind_parent[self.key] = value

        return bind_parent[self.key]
