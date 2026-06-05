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
"""Utilities for working with callable wrappers."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial, wraps
from typing import Any


def unwrap_partial(fn: Callable) -> tuple[Callable, tuple[Any, ...], dict[str, Any]]:
    """Return the base callable and arguments bound by nested ``functools.partial`` wrappers."""
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = {}

    while isinstance(fn, partial):
        args = fn.args + args
        kwargs = {**(fn.keywords or {}), **kwargs}
        fn = fn.func

    return fn, args, kwargs


def apply_partial_args(fn: Callable, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Callable:
    """Return a callable that prepends partial-bound arguments to call-time arguments."""
    if not args and not kwargs:
        return fn

    @wraps(fn)
    def wrapper(*call_args, **call_kwargs):
        return fn(*args, *call_args, **{**kwargs, **call_kwargs})

    return wrapper
