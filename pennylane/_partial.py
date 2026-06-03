"""Utilities for working with :class:`functools.partial` callables."""

from functools import partial


def unwrap_partial(func):
    """Return the base callable and arguments bound by nested partial wrappers."""
    bound_args = ()
    bound_kwargs = {}

    while isinstance(func, partial):
        bound_args = func.args + bound_args
        bound_kwargs = {**(func.keywords or {}), **bound_kwargs}
        func = func.func

    return func, bound_args, bound_kwargs


def merge_partial_args(bound_args, bound_kwargs, call_args, call_kwargs):
    """Merge partial-bound arguments with call-time arguments."""
    return bound_args + call_args, {**bound_kwargs, **call_kwargs}
