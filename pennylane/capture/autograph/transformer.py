# Copyright 2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
AutoGraph is a source-to-source transformation system for converting imperative code into
traceable code for compute graph generation. The system is implemented in the Diastatic-Malt
package (originally from TensorFlow).
Here, we integrate AutoGraph into Catalyst to improve the UX and allow programmers to use built-in
Python control flow and other imperative expressions rather than the functional equivalents provided
by Catalyst.
"""
import copy
import inspect
from contextlib import ContextDecorator

import pennylane as qml
from malt.core import ag_ctx, converter
from malt.impl.api import PyToPy

import catalyst
from catalyst.autograph import ag_primitives, operator_update
from catalyst.utils.exceptions import AutoGraphError


class CatalystTransformer(PyToPy):
    """A source-to-source transformer to convert imperative style control flow into a function style
    suitable for tracing."""

    def __init__(self):
        super().__init__()

        self._extra_locals = None

    def transform(self, obj, user_context):
        """Launch the transformation process. Typically this only works on function objects.
        Here we also allow QNodes to be transformed."""

        # By default AutoGraph will only convert function or method objects, not arbitrary classes
        # such as QNode objects. Here we handle them explicitly, but we might need a more general
        # way to handle these in the future.
        # We may also need to check how this interacts with other common function decorators.
        fn = obj
        if isinstance(obj, qml.QNode):
            fn = obj.func
        elif inspect.isfunction(fn) or inspect.ismethod(fn):
            pass
        elif callable(obj):
            # pylint: disable=unnecessary-lambda,unnecessary-lambda-assignment
            fn = lambda *args, **kwargs: obj(*args, **kwargs)
        else:
            raise AutoGraphError(f"Unsupported object for transformation: {type(fn)}")

        new_fn, module, source_map = self.transform_function(fn, user_context)
        new_obj = new_fn

        if isinstance(obj, qml.QNode):
            new_obj = copy.copy(obj)
            new_obj.func = new_fn

        return new_obj, module, source_map

    def get_extra_locals(self):
        """Here we can provide any extra names that the converted function should have access to.
        At a minimum we need to provide the module with definitions for AutoGraph primitives."""

        if self._extra_locals is None:
            extra_locals = super().get_extra_locals()
            updates = {key: ag_primitives.__dict__[key] for key in ag_primitives.__all__}
            extra_locals["ag__"].__dict__.update(updates)
            self._extra_locals = extra_locals

        return self._extra_locals

    def has_cache(self, fn):
        """Check for the presence of the given function in the cache. Functions to be converted are
        cached by the function object itself as well as the conversion options."""

        return (
            self._cache.has(fn, TOPLEVEL_OPTIONS)
            or self._cache.has(fn, NESTED_OPTIONS)
            or self._cache.has(fn, STANDARD_OPTIONS)
        )

    def get_cached_function(self, fn):
        """Retrieve a Python function object for a previously converted function.
        Note that repeatedly calling this function with the same arguments will result in new
        function objects every time, however their source code should be identical with the
        exception of auto-generated names."""

        # Converted functions are cached as a _PythonFnFactory object.
        if self._cache.has(fn, TOPLEVEL_OPTIONS):
            cached_factory = self._cached_factory(fn, TOPLEVEL_OPTIONS)
        elif self._cache.has(fn, NESTED_OPTIONS):
            cached_factory = self._cached_factory(fn, NESTED_OPTIONS)
        else:
            cached_factory = self._cached_factory(fn, STANDARD_OPTIONS)

        # Convert to a Python function object before returning (e.g. to obtain its source code).
        new_fn = cached_factory.instantiate(
            fn.__globals__,
            fn.__closure__ or (),
            defaults=fn.__defaults__,
            kwdefaults=getattr(fn, "__kwdefaults__", None),
        )

        return new_fn

    def transform_ast(self, node, ctx):
        """Overload of PyToPy.transform_ast from DiastaticMalt

        .. note::
            Once the operator_update interface has been migrated to the
            DiastaticMalt project, this overload can be deleted."""
        # The operator_update transform would be more correct if placed with
        # slices.transform in PyToPy.transform_ast in DiastaticMalt rather than
        # at the beginning of the transformation. operator_update.transform
        # should come after the unsupported features check and intial analysis,
        # but it fails if it does not come before variables.transform.
        node = operator_update.transform(node, ctx)
        node = super().transform_ast(node, ctx)
        return node


def run_autograph(fn):
    """Decorator that converts the given function into graph form."""

    user_context = converter.ProgramContext(TOPLEVEL_OPTIONS)

    new_fn, module, source_map = TRANSFORMER.transform(fn, user_context)
    new_fn.ag_module = module
    new_fn.ag_source_map = source_map
    new_fn.ag_unconverted = fn

    return new_fn


def autograph_source(fn):
    """Utility function to retrieve the source code of a function converted by AutoGraph.

    .. warning::

        Nested functions (those not directly decorated with ``@qjit``) are only lazily converted by
        AutoGraph. Make sure that the function has been traced at least once before accessing its
        transformed source code, for example by specifying the signature of the compiled program
        or by running it at least once.

    Args:
        fn (Callable): the original function object that was converted

    Returns:
        str: the source code of the converted function

    Raises:
        AutoGraphError: If the given function was not converted by AutoGraph, an error will be
                        raised.

    **Example**

    .. code-block:: python

        def decide(x):
            if x < 5:
                y = 15
            else:
                y = 1
            return y

        @qjit(autograph=True)
        def func(x: int):
            y = decide(x)
            return y ** 2

    >>> print(autograph_source(decide))
    def decide_1(x):
        with ag__.FunctionScope('decide', 'fscope', ag__.STD) as fscope:
            def get_state():
                return (y,)
            def set_state(vars_):
                nonlocal y
                (y,) = vars_
            def if_body():
                nonlocal y
                y = 15
            def else_body():
                nonlocal y
                y = 1
            y = ag__.Undefined('y')
            ag__.if_stmt(x < 5, if_body, else_body, get_state, set_state, ('y',), 1)
            return y
    """

    # Handle directly converted objects.
    if hasattr(fn, "ag_unconverted"):
        return inspect.getsource(fn)

    # Unwrap known objects to get the function actually transformed by autograph.
    if isinstance(fn, catalyst.QJIT):
        fn = fn.original_function
    if isinstance(fn, qml.QNode):
        fn = fn.func

    if TRANSFORMER.has_cache(fn):
        new_fn = TRANSFORMER.get_cached_function(fn)
        return inspect.getsource(new_fn)

    raise AutoGraphError(
        "The given function was not converted by AutoGraph. If you expect the"
        "given function to be converted, please submit a bug report."
    )


class DisableAutograph(ag_ctx.ControlStatusCtx, ContextDecorator):
    """Context decorator that disables AutoGraph for the given function/context.

    .. note::

        A singleton instance is used for discarding parentheses usage:

        @disable_autograph
        instead of
        @DisableAutograph()

        with disable_autograph:
        instead of
        with DisableAutograph()

    **Example 1: as a function decorator**

    .. code-block:: python

        @disable_autograph
        def f():
            x = 6
            if x > 5:
                y = x ** 2
            else:
                y = x ** 3
            return y

        @qjit(autograph=True)
        def g(x: float, n: int):
            for _ in range(n):
                x = x + f()
            return x

    >>> print(g(0.4, 6))
    216.4

    **Example 2: as a context manager**

    .. code-block:: python

        def f():
            x = 6
            if x > 5:
                y = x ** 2
            else:
                y = x ** 3
            return y

        @qjit(autograph=True)
        def g():
            x = 0.4
            with disable_autograph:
                x += f()
            return x

    >>> print(g())
    36.4
    """

    def __init__(self):
        super().__init__(status=ag_ctx.Status.DISABLED)


# Singleton instance of DisableAutograph
disable_autograph = DisableAutograph()

# converter.Feature.LISTS permits overloading the 'set_item' function in 'ag_primitives.py'
OPTIONAL_FEATURES = [converter.Feature.BUILTIN_FUNCTIONS, converter.Feature.LISTS]

TOPLEVEL_OPTIONS = converter.ConversionOptions(
    recursive=True,
    user_requested=True,
    internal_convert_user_code=True,
    optional_features=OPTIONAL_FEATURES,
)

NESTED_OPTIONS = converter.ConversionOptions(
    recursive=True,
    user_requested=False,
    internal_convert_user_code=True,
    optional_features=OPTIONAL_FEATURES,
)

STANDARD_OPTIONS = converter.STANDARD_OPTIONS

# Keep a global instance of the transformer to benefit from caching.
TRANSFORMER = CatalystTransformer()
