# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Here, we integrate AutoGraph into PennyLane to improve the UX and allow programmers to use built-in
Python control flow and other imperative expressions rather than the functional equivalents provided
by PennyLane.
"""
import copy
import inspect

from malt.core import converter
from malt.impl.api import PyToPy

import pennylane as qml

from . import ag_primitives
from .ag_primitives import AutoGraphError


class PennyLaneTransformer(PyToPy):
    """A source-to-source transformer to convert imperative style control flow into a function style
    suitable for tracing."""

    def __init__(self):
        super().__init__()

        self._extra_locals = None

    def transform(self, obj, user_context):
        """Launch the transformation process. Typically, this only works on function objects.
        Here we also allow QNodes to be transformed."""

        # By default, AutoGraph will only convert function or method objects, not arbitrary classes
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
        function objects every time, however their source code should be identical except for
        the auto-generated names."""

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


def run_autograph(fn):
    """Decorator that converts the given function into graph form.

    AutoGraph can be used in PennyLane's capture workflow to convert Pythonic control flow to PennyLane
    native control flow. This requires the ``diastatic-malt`` package, a standalone fork of the AutoGraph
    module in TensorFlow (`official documentation <https://github.com/PennyLaneAI/diastatic-malt/blob/main/g3doc/reference/index.md>`_
    ).

    Args:
        fn (Callable): The callable to be converted. This could be a function, a QNode, or another callable object.
            For a QNode, the ``QNode.func`` will be converted. For another callable object, a function calling the
            object will be converted.

    Returns:
        Callable: For a function, the converted function is returned directly.
        For a QNode, a copy of the QNode will be returned with ``QNode.func`` replaced with the converted version of ``func``.
        For any other callable ``obj``, the returned function will be a converted version of
        ``lambda *args, **kwargs: obj(*args, **kwargs)``

    .. note::

        There are some limitations and sharp bits regarding AutoGraph; to better understand
        supported behaviour and limitations, see :doc:`/development/autograph`.

    .. warning::

        Nested functions are only lazily converted by AutoGraph. If the input includes nested
        functions, these won't be converted until the first time the function is traced.

    **Example**

    Consider the following function including Pythonic control flow, which can't be captured directly:

    >>> def f(x, n):
    ...     for i in range(n):
    ...          x += 1
    ...     return x
    >>> jax.make_jaxpr(f)(2, 4)
    TracerIntegerConversionError: The __index__() method was called on traced array with shape int64[].
    The error occurred while tracing the function f at /var/folders/61/wr1fxnf95tg9k56bz1_7g29r0000gq/T/ipykernel_23187/3992882129.py:1 for make_jaxpr. This concrete value was not available in Python because it depends on the value of the argument n.

    Passing it thorough AutoGraph converts the structure of the function to native PennyLane control flow
    with :func:`~.cond`, :func:`~.for_loop`, and :func:`~.while_loop`, making it possible to capture:

    >>> ag_fn = run_autograph(f)
    >>> jax.make_jaxpr(ag_fn)(2, 4)
    { lambda ; a:i64[] b:i64[]. let
        c:i64[] = for_loop[
          args_slice=slice(0, None, None)
          consts_slice=slice(0, 0, None)
          jaxpr_body_fn={ lambda ; d:i64[] e:i64[]. let f:i64[] = add e 1 in (f,) }
        ] 0 b 1 a
      in (c,) }
    """

    user_context = converter.ProgramContext(TOPLEVEL_OPTIONS)

    new_fn, module, source_map = TRANSFORMER.transform(fn, user_context)

    # needed for autograph_source when examining a converted QNode
    if isinstance(new_fn, qml.QNode):
        new_fn.func.ag_unconverted = fn.func

    new_fn.ag_module = module
    new_fn.ag_source_map = source_map
    new_fn.ag_unconverted = fn

    return new_fn


def autograph_source(fn):
    """Utility function to retrieve the source code of a function converted by AutoGraph.

    .. warning::

        Nested functions are only lazily converted by AutoGraph. Make sure that the function has
        been traced at least once before accessing its transformed source code, for example by
        specifying the signature of the compiled program or by running it at least once.

    Args:
        fn (Callable): the original function object that was converted

    Returns:
        str: the source code of the converted function

    Raises:
        AutoGraphError: If the given function was not converted by AutoGraph, an error will be
                        raised.

    **Example**

    .. code-block:: python

        from pennylane.capture.autograph import run_autograph, autograph_source

        def decide(x):
            if x < 5:
                y = 15
            else:
                y = 1
            return y

        ag_decide = run_autograph(decide)

    >>> print(autograph_source(ag_fn))
    def ag__decide(x):
        with ag__.FunctionScope('decide', 'fscope', ag__.ConversionOptions(recursive=True, user_requested=True, optional_features=ag__.Feature.BUILTIN_FUNCTIONS, internal_convert_user_code=True)) as fscope:
            do_return = False
            retval_ = ag__.UndefinedReturnValue()

            def get_state():
                return (y,)

            def set_state(vars_):
                nonlocal y
                y, = vars_

            def if_body():
                nonlocal y
                y = 15

            def else_body():
                nonlocal y
                y = 1
            y = ag__.Undefined('y')
            ag__.if_stmt(ag__.ld(x) < 5, if_body, else_body, get_state, set_state, ('y',), 1)
            try:
                do_return = True
                retval_ = ag__.ld(y)
            except:
                do_return = False
                raise
            return fscope.ret(retval_, do_return)
    """

    # Handle directly converted objects.
    if hasattr(fn, "ag_unconverted"):
        return inspect.getsource(fn)

    # Unwrap known objects to get the function actually transformed by autograph.
    if isinstance(fn, qml.QNode):
        fn = fn.func

    if TRANSFORMER.has_cache(fn):
        new_fn = TRANSFORMER.get_cached_function(fn)
        return inspect.getsource(new_fn)

    raise AutoGraphError(
        "The given function was not converted by AutoGraph. If you expect the "
        "given function to be converted, please submit a bug report."
    )


TOPLEVEL_OPTIONS = converter.ConversionOptions(
    recursive=True,
    user_requested=True,
    internal_convert_user_code=True,
    optional_features=[converter.Feature.BUILTIN_FUNCTIONS],
)

NESTED_OPTIONS = converter.ConversionOptions(
    recursive=True,
    user_requested=False,
    internal_convert_user_code=True,
    optional_features=[converter.Feature.BUILTIN_FUNCTIONS],
)

STANDARD_OPTIONS = converter.STANDARD_OPTIONS

# Keep a global instance of the transformer to benefit from caching.
TRANSFORMER = PennyLaneTransformer()
