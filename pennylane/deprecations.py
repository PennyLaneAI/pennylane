# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains everything related to deprecations in PennyLane."""

import functools
from collections.abc import Callable


class PennyLaneDeprecationWarning(UserWarning):
    """Warning raised when a PennyLane feature is being deprecated."""


_TF_DEPRECATION_MSG = (
    "Support for the TensorFlow interface is deprecated and will be removed in v0.44. "
    "Future versions of PennyLane are not guaranteed to work with TensorFlow. "
    "Please migrate your workflows to JAX or Pytorch to benefit from enhanced support and features."
)


def deprecate_argument(
    name: str,
    version: str,
    *,
    new_name: str | None = None,
    removal_version: str | None = None,
) -> Callable:
    """Decorator that assists in deprecating an argument in a function.

    Args:
        name (str): The name of the deprecated argument
        version (str): The version when the deprecation cycle will begin

    Keyword Args:
        new_name (str | None): The argument's new name, if it's being renamed
        removal_version (str | None): The removal version of the deprecated argument

    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator


def deprecate_argument_value(
    name: str,
    value: str,
    version: str,
    *,
    new_value: str | None = None,
    removal_version: str | None = None,
    additional_info: str | None = None,
) -> Callable:
    """Decorator that assists in deprecating an argument in a function.

    Args:
        name (str): The name of the argument with the deprecated value
        value (str): The deprecated value for the argument
        version (str): The version when the deprecation cycle will begin

    Keyword Args:
        new_value (str | None): The new value that should be used
        removal_version (str | None): The removal version of the deprecated value

    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator
