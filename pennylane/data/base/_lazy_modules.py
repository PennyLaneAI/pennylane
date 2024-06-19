"""Contains a lazy-loaded interface to the HDF5 module. For internal use only."""

import importlib
from types import ModuleType
from typing import Any, Callable, Optional, Union

_MISSING_MODULES_EXC = ImportError(
    "This feature requires the 'aiohttp', 'h5py' and 'fsspec' packages. "
    "They can be installed with:\n\n pip install aiohttp fsspec h5py"
)


class lazy_module:  # pylint: disable=too-few-public-methods
    """Provides a lazy-loaded interface to a Python module, and its submodules. The module will not
    be imported until an attribute is accessed."""

    def __init__(
        self,
        module_name_or_module: Union[str, ModuleType],
        import_exc: Optional[Exception] = None,
        post_import_cb: Optional[Callable[[ModuleType], None]] = None,
    ):
        """Creates a new top-level lazy module or initializes a nested one.

        Args:
            module_name_or_module: Name of module to lazily import, or a module object
                for a nested lazy module.
            import_exc: Custom Exception to raise when an ``ImportError`` occurs. Will only
                be used by the top-level ``lazy_module`` instance, not nested modules
        """
        if isinstance(module_name_or_module, ModuleType):  # pragma: no cover
            self.__module = module_name_or_module
            self.__module_name = self.__module.__name__
        else:
            self.__module = None
            self.__module_name = module_name_or_module

        self.__import_exc = import_exc
        self.__post_import_cb = post_import_cb
        self.__submods = {}

    def __getattr__(self, __name: str) -> Any:
        if self.__module is None:
            self.__import_module()
        elif __name in self.__submods:
            return self.__submods[__name]  # pragma: no cover

        try:
            resource = getattr(self.__module, __name)
        except AttributeError as attr_exc:  # pragma: no cover
            try:
                submod = lazy_module(importlib.import_module(f"{self.__module_name}.{__name}"))
            except ImportError as import_exc:
                raise attr_exc from import_exc

            self.__submods[__name] = submod
            return submod

        return resource

    def __import_module(self) -> None:
        try:
            self.__module = importlib.import_module(self.__module_name)
        except ImportError as exc:  # pragma: no cover
            if self.__import_exc:
                raise self.__import_exc from exc

            raise exc

        if self.__post_import_cb:
            self.__post_import_cb(self.__module)


def _configure_h5py(h5py_module: ModuleType) -> None:
    """Configures the ``h5py`` module after import.

    Sets the ``track_order`` flag, so that groups and files remember
    the insert order of objects, like Python dictionaries.

    See https://docs.h5py.org/en/stable/config.html
    """
    h5py_module.get_config().track_order = True


h5py = lazy_module("h5py", _MISSING_MODULES_EXC, _configure_h5py)
fsspec = lazy_module("fsspec", _MISSING_MODULES_EXC)
