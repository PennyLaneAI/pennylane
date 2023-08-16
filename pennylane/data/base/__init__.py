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
"""The base module contains the base class that defines the underlying
type machinery, and the low-level HDF5 interface of the data module."""

from .attribute import AttributeInfo, DatasetAttribute
from .dataset import Dataset, field
from .mapper import DatasetNotWriteableError

__all__ = ("AttributeInfo", "DatasetAttribute", "Dataset", "DatasetNotWriteableError", "field")
