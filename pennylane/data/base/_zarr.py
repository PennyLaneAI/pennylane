import importlib
from types import ModuleType
from typing import Any, MutableMapping

from numpy.typing import ArrayLike


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
