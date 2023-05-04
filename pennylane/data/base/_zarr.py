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
"""Contains a lazy-loaded interface to the Zarr module."""

import importlib
from types import ModuleType
from typing import Any


class _zarr_lazy:
    """Provides a lazy-loaded interface to the Zarr module."""

    convenience: ModuleType

    def __init__(self, __root=None):
        self.__root = __root
        self.__submodules = {}

    def _do_import(self):
        try:
            self.__root = importlib.import_module("zarr")
            self.__submodules["convenience"] = importlib.import_module("zarr.convenience")
        except ImportError as Error:
            raise ImportError(
                "This feature requires the 'zarr' package. "
                "It can be installed with:\n\n pip install zarr."
            ) from Error

    def __getattr__(self, resource: str) -> Any:
        if self.__root is None:
            self._do_import()

        if resource in self.__submodules:
            return self.__submodules[resource]

        return getattr(self.__root, resource)


zarr = _zarr_lazy()
