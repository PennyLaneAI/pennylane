# Copyright 2018-2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for resolving callables wrapped with ``functools.partial``."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial, wraps
from typing import Any


def unwrap_functools_partial(
    obj: Callable,
) -> tuple[Callable, tuple[Any, ...], dict[str, Any]]:
    """Recursively unwrap ``functools.partial`` objects.

    Args:
        obj (Callable): a callable that may be wrapped in one or more ``functools.partial`` objects.

    Returns:
        tuple[Callable, tuple, dict]: the core callable along with all bound positional arguments and
        keyword arguments collected from the partial wrappers. Inner partial bindings are ordered
        before outer partial bindings.
    """
    if not isinstance(obj, partial):
        return obj, (), {}

    inner, bound_args, bound_kwargs = unwrap_functools_partial(obj.func)
    bound_args = bound_args + obj.args
    bound_kwargs = {**bound_kwargs, **obj.keywords}
    return inner, bound_args, bound_kwargs


def bind_functools_partial(
    callable_: Callable,
    bound_args: tuple[Any, ...],
    bound_kwargs: dict[str, Any],
) -> Callable:
    """Wrap a callable so that bound partial arguments are supplied on invocation.

    Args:
        callable_ (Callable): the callable to wrap.
        bound_args (tuple): positional arguments bound via ``functools.partial``.
        bound_kwargs (dict): keyword arguments bound via ``functools.partial``.

    Returns:
        Callable: a wrapper with the same behaviour as applying the collected partial bindings before
        calling ``callable_``.
    """
    if not bound_args and not bound_kwargs:
        return callable_

    @wraps(callable_)
    def wrapper(*args, **kwargs):
        return callable_(*bound_args, *args, **{**bound_kwargs, **kwargs})

    return wrapper
