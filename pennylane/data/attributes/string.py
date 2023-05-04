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
"""Contains an AttributeType for str objects."""


from pennylane.data.base.attribute import AttributeType
from pennylane.data.base.typing_util import ZarrArray, ZarrGroup


class DatasetString(AttributeType):
    type_id = "string"

    @classmethod
    def consumes_types(cls) -> tuple[type[str]]:
        return (str,)

    def zarr_to_value(self, bind: ZarrArray) -> str:
        return bind[()]

    def value_to_zarr(self, bind_parent: ZarrGroup, key: str, value: str) -> ZarrArray:
        bind_parent[key] = value

        return bind_parent[key]
