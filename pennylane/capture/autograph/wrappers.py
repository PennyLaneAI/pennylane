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
"""Register adapters for callable classes that wrap functions."""

from collections.abc import Callable
from functools import singledispatch


@singledispatch
def _wrapper_adapter(
    _wrapper: Callable,
) -> tuple[Callable, Callable[[Callable], Callable]] | None:
    """Return ``None`` for unregistered wrapper types."""
    return None


def get_autograph_wrapper(
    wrapper: Callable,
) -> tuple[Callable, Callable[[Callable], Callable]] | None:
    """Return the wrapped function and its rebuild callback, or ``None`` for unregistered types."""
    return _wrapper_adapter.dispatch(type(wrapper))(wrapper)


def register_autograph_wrapper(wrapper_type: type) -> Callable:
    """Register autograph handling for a callable class that wraps a user function.

    Args:
        wrapper_type (type): callable class to register

    Returns:
        Callable: decorator that registers the adapter

    This decorator allows callable classes to propogate autograph to their target functions.
    This is used for :class:`~.QNode`, :class:`~.grad`, :class:`~.jacobian`, and
    :class:`~.value_and_grad`.
    The wrapper takes an instance of the class, and should return both the target function and
    a way to reconstruct the transformed object.

    **Example**

    .. code-block:: python

        from pennylane.capture.autograph import register_autograph_wrapper, run_autograph

        class MyCallableClass:
            def __init__(self, function, param):
                self.function = function
                self.param = param

            def __call__(self, *args, **kwargs):
                return self.function(*args, **kwargs)

        @register_autograph_wrapper(MyCallableClass)
        def adapt(original):

            def rebuild(transformed_function) -> MyCallableClass:
                return MyCallableClass(transformed_function, original.param)

            return original.function, rebuild

        def f(data):
            if data > 0:
                result = data ** 2
            else:
                result = data ** 3
            return result

        converted = run_autograph(MyCallableClass(f, 2))

    >>> converted(2.0)
    4.0
    """
    return _wrapper_adapter.register(wrapper_type)
