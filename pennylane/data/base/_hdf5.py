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
"""Contains a lazy-loaded interface to the HDF5 module. For internal use only."""

import importlib
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Union, Optional
from uuid import uuid4
from functools import lru_cache


from .typing_util import HDF5Any, HDF5Group

_MISSING_MODULES_EXC = ImportError(
    "This feature requires the 'aiohttp', 'h5py' and 'fsspec' packages. "
    "They can be installed with:\n\n pip install aiohttp fsspec h5py"
)


class _h5py_lazy:  # pylint: disable=too-few-public-methods
    """Provides a lazy-loaded interface to the H5Py module, and convenience methods."""

    def __init__(self):
        self.__h5 = None

    def _do_import(self):
        try:
            self.__h5 = importlib.import_module("h5py")

        except ImportError as exc:
            raise _MISSING_MODULES_EXC from exc

    def __getattr__(self, resource: str) -> Any:
        if self.__h5 is None:
            self._do_import()

        return getattr(self.__h5, resource)

    def group(self) -> HDF5Group:
        """Creates a new HDF5 group in memory."""
        f = self.File(str(uuid4()), mode="w-", driver="core", backing_store=False)
        return f

    @staticmethod
    def copy(
        source: HDF5Any,
        dest: HDF5Group,
        key: str,
        if_exists: Literal["raise", "overwrite"] = "raise",
    ):
        """Copy HDF5 array or group ``source`` into group ``dest``.

        Args:
            source: HDF5 group or array to copy
            dest: Target HDF5 group
            key: Name to save source into, under dest
            if_exists: How to handle conflicts if ``key`` already exists in
                ``dest``. ``"raise"`` will raise an exception, ``overwrite``
                will overwrite the existing object in ``dest``."""
        if if_exists == "raise" and key in dest:
            raise ValueError(f"Key {key} already exists in in {source.name}")

        dest.copy(source, dest, name=key)

    def copy_all(
        self, source: HDF5Group, dest: HDF5Group, if_exists: Literal["raise", "overwrite"] = "raise"
    ):
        """Copies all values of ``source`` into ``dest``."""
        for key, elem in source.items():
            self.copy(elem, dest, key=key, if_exists=if_exists)

    def open_group(
        self, path: Union[Path, str], mode: Literal["r", "w", "w-", "a"] = "r"
    ) -> HDF5Group:
        """Creates or opens an HDF5 file at ``path`` and returns the root HDF5 group.

        Args:
            path: File system path for HDF5 File
            mode:  File handling mode. Possible values are "w-" (create, fail if file
                exists), "w" (create, overwrite existing), "a" (append existing,
                create if doesn't exist), "r" (read existing, must exist). Default is "r".
        """
        return self.__h5.File(path, mode)


h5py = _h5py_lazy()


@lru_cache(maxsize=1)
def _import_fsspec() -> ModuleType:
    """Import and return the ``fsspec`` module."""
    try:
        fsspec = importlib.import_module("fsspec")
        importlib.import_module("aiohttp")  # make sure it's installed
    except ImportError as exc:
        raise _MISSING_MODULES_EXC from exc

    return fsspec


def open_hdf5_s3(s3_url: str, cache_dir: Optional[Path]) -> HDF5Group:
    """Uses ``fsspec`` module to open the HDF5 file at ``s3_url``.

    This requires both ``fsspec`` and ``aiohttp`` to be installed.
    """
    fsspec = _import_fsspec()

    if cache_dir is not None:
        fs = fsspec.open(f"blockcache::{s3_url}", blockcache={"cache_storage": str(cache_dir)})
    else:
        fs = fsspec.open(s3_url)

    return h5py.File(fs.open())
