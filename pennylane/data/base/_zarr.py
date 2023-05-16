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
"""Contains a lazy-loaded interface to the Zarr module. For internal use only."""

import importlib
from types import ModuleType
from typing import Any


class _zarr_lazy:  # pylint: disable=too-few-public-methods
    """Provides a lazy-loaded interface to the Zarr module."""

    convenience: ModuleType

    def __init__(self):
        self.__zarr = None

    def _do_import(self):
        try:
            self.__zarr = importlib.import_module("zarr")
        except ImportError as Error:
            raise ImportError(
                "This feature requires the 'zarr' and 'zstd' packages. "
                "They can be installed with:\n\n pip install zarr zstd."
            ) from Error

        numcodecs = importlib.import_module("numcodecs")
        self.__zarr.storage.default_compressor = numcodecs.Blosc(cname="zstd", clevel=3)

    def __getattr__(self, resource: str) -> Any:
        if self.__zarr is None:
            self._do_import()

        return getattr(self.__zarr, resource)


zarr = _zarr_lazy()
