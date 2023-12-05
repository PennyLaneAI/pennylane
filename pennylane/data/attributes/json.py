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
"""Contains an DatasetAttribute for JSON-serializable data."""


import json

from pennylane.data.base.attribute import DatasetAttribute
from pennylane.data.base.hdf5 import HDF5Array, HDF5Group
from pennylane.data.base.typing_util import JSON


class DatasetJSON(DatasetAttribute[HDF5Array, JSON, JSON]):
    """Dataset type for JSON-serializable data."""

    type_id = "json"

    def hdf5_to_value(self, bind: HDF5Array) -> JSON:
        return json.loads(bind.asstr()[()])

    def value_to_hdf5(self, bind_parent: HDF5Group, key: str, value: JSON) -> HDF5Array:
        bind_parent[key] = json.dumps(value)

        return bind_parent[key]
