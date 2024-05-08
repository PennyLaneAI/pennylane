import inspect
import sys


def _is_local_fn(f, mod_name):
    """
    Predicate that validates if argument `f` is a local module `mod_name`.
    """
    is_func = inspect.isfunction(f)
    is_local_to_mod = inspect.getmodule(f).__name__ == mod_name
    return is_func and is_local_to_mod


def _add_logging_all(mod_name):
    """
    Modifies the module to add logging implicitly to all free-functions.
    """
    l_func = inspect.getmembers(
        sys.modules[mod_name], predicate=lambda x: _is_local_fn(x, mod_name)
    )
    for f_name, f in l_func:
        globals()[f_name] = debug_logger(f)
