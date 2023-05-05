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
"""
Provides base `Dataset` class.
"""

from pathlib import Path
from typing import Any, Optional, Union

from pennylane.data.base.dataset import DatasetBase, attribute
from pennylane.data.base.typing_util import ZarrGroup


class Dataset(DatasetBase):
    """Base class for public datasets."""

    description: str = attribute(doc="Description for this Dataset")

    def __init__(
        self,
        bind: Optional[Union[str, Path, ZarrGroup]] = None,
        *,
        description: str,
        **attrs: Any,
    ):
        """
        Load a dataset from a Zarr Group or initialize a new Dataset.

        Args:
            bind: The Zarr group, or path to zarr file, that will contain this dataset.
                If None, the dataset will be stored in memory. Any attributes that
                already exist in ``bind`` will be loaded into this dataset.
            **attrs: Attributes to add to this dataset.
        """
        super().__init__(bind=bind, description=description, **attrs)
