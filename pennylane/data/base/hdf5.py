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
"""Contains H5Py lazy module and convenience functions for HDF5."""

from collections.abc import MutableMapping
from pathlib import Path
from typing import Literal, TypeVar, Union
from uuid import uuid4

from numpy.typing import ArrayLike

from ._lazy_modules import fsspec, h5py

# Type aliases for HDF5 objects.
HDF5Attrs = MutableMapping
HDF5Array = ArrayLike
HDF5Group = MutableMapping
HDF5Any = Union[HDF5Array, HDF5Group]
HDF5 = TypeVar("HDF5", HDF5Array, HDF5Group, HDF5Any)


def open_group(path: Union[Path, str], mode: Literal["r", "w", "w-", "a"] = "r") -> HDF5Group:
    """Creates or opens an HDF5 file at ``path`` and returns the root HDF5 group.

    Args:
        path: File system path for HDF5 File
        mode:  File handling mode. Possible values are "w-" (create, fail if file
            exists), "w" (create, overwrite existing), "a" (append existing,
            create if doesn't exist), "r" (read existing, must exist). Default is "r".
    """
    return h5py.File(path, mode)


def create_group() -> HDF5Group:
    """Creates a new HDF5 group in memory."""
    f = h5py.File(str(uuid4()), mode="w-", driver="core", backing_store=False)

    return f


def copy(
    source: HDF5Any,
    dest: HDF5Group,
    key: str,
    on_conflict: Literal["raise", "overwrite", "ignore"] = "raise",
) -> None:
    """Copy HDF5 array or group ``source`` into group ``dest``.

    Args:
        source: HDF5 group or array to copy
        dest: Target HDF5 group
        keys: Name to save source into, under dest
        on_conflict: How to handle conflicts if ``key`` already exists in
            ``dest``. ``"raise"`` will raise an exception, ``overwrite``
            will overwrite the existing object in ``dest``, ``"ignore"`` will
            do nothing
    """
    if key in dest:
        if on_conflict == "overwrite":
            del dest[key]
        elif on_conflict == "raise":
            raise ValueError(f"Key '{key}' already exists in '{source.name}'")
        if on_conflict == "ignore":
            return

    dest.copy(source, dest, name=key)


def copy_all(
    source: HDF5Group,
    dest: HDF5Group,
    *keys: str,
    on_conflict: Literal["raise", "overwrite", "ignore"] = "ignore",
    without_attrs: bool = False,
) -> None:
    """Copies all the elements of ``source`` named ``keys`` into ``dest``. If no keys
    are provided, all elements of ``source`` will be copied."""
    if not keys:
        keys = source

    for key in keys:
        copy(source[key], dest, key=key, on_conflict=on_conflict)

    if not without_attrs:
        dest.attrs.update(source.attrs)


def open_hdf5_s3(s3_url: str, *, block_size: int = 8388608) -> HDF5Group:
    """Uses ``fsspec`` module to open the HDF5 file at ``s3_url``.

    This requires both ``fsspec`` and ``aiohttp`` to be installed.

    Args:
        s3_url: URL of dataset file in S3
        block_size: Number of bytes to fetch per read operation. Larger values
            may improve performance for large datasets
    """

    # Tells fsspec to fetch data in 8MB chunks for faster loading
    memory_cache_args = {"cache_type": "mmap", "block_size": block_size}
    fs = fsspec.open(s3_url, **memory_cache_args)

    return h5py.File(fs.open())
