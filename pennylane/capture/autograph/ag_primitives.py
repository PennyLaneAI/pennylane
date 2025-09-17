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
This module provides the implementation of AutoGraph primitives in terms of traceable PennyLane
functions. The purpose is to convert imperative style code to functional or graph-style code.
"""
import copy
import functools
import operator
from collections.abc import Callable, Iterator
from numbers import Number
from typing import Any, SupportsIndex, Union

from malt.core import config as ag_config
from malt.impl import api as ag_api
from malt.impl.api import converted_call as ag_converted_call
from malt.operators import py_builtins as ag_py_builtins
from malt.operators.variables import Undefined

import pennylane as qml
from pennylane.exceptions import AutoGraphError

has_jax = True
try:
    import jax
    import jax.numpy as jnp
    from jax.interpreters.partial_eval import DynamicJaxprTracer
except ImportError:  # pragma: no cover
    has_jax = False


__all__ = [
    "if_stmt",
    "for_stmt",
    "while_stmt",
    "converted_call",
    "and_",
    "or_",
    "not_",
    "set_item",
    "update_item_with_op",
]


def set_item(
    target: Union["DynamicJaxprTracer", list],
    index: Union[int, "DynamicJaxprTracer"],
    x: Union[Number, "DynamicJaxprTracer"],
):
    """An implementation of the AutoGraph 'set_item' function."""

    if qml.math.is_abstract(target):
        target = target.at[index].set(x)
    else:
        target[index] = x

    return target


def update_item_with_op(
    target: Union["DynamicJaxprTracer", list],
    index: Union[int, "DynamicJaxprTracer"],
    x: Union[Number, "DynamicJaxprTracer"],
    op: str,
):
    """An implementation of the AutoGraph 'update_item_with_op' function."""

    gast_op_map = {"mult": "multiply", "div": "divide", "add": "add", "sub": "add", "pow": "power"}
    inplace_operation_map = {
        "mult": "mul",
        "div": "truediv",
        "add": "add",
        "sub": "add",
        "pow": "pow",
    }
    if op == "sub":
        x = -x

    if qml.math.is_abstract(target):
        if isinstance(index, slice):
            target = getattr(target.at[index.start : index.stop : index.step], gast_op_map[op])(x)
        else:
            target = getattr(target.at[index], gast_op_map[op])(x)
    else:
        # Use Python's in-place operator
        target[index] = getattr(operator, f"__i{inplace_operation_map[op]}__")(target[index], x)
    return target


def _assert_results(results, var_names):
    """Assert that none of the results are undefined, i.e. have no value."""

    assert len(results) == len(var_names)

    for r, v in zip(results, var_names):
        if isinstance(r, Undefined):
            raise AutoGraphError(f"Some branches did not define a value for variable '{v}'")

    return results


# pylint: disable=too-many-arguments, too-many-positional-arguments
def if_stmt(
    pred: bool,
    true_fn: Callable[[], Any],
    false_fn: Callable[[], Any],
    get_state: Callable[[], tuple],
    set_state: Callable[[tuple], None],
    symbol_names: tuple[str],
    _num_results: int,
):
    """An implementation of the AutoGraph 'if' statement. The interface is defined by AutoGraph,
    here we merely provide an implementation of it in terms of PennyLane primitives."""

    # Cache the initial state of all modified variables. Required because we trace all branches,
    # and want to restore the initial state before entering each branch.
    init_state = get_state()

    @qml.cond(pred)
    def functional_cond():
        set_state(init_state)
        true_fn()
        results = get_state()
        return _assert_results(results, symbol_names)

    @functional_cond.otherwise
    def functional_cond():
        set_state(init_state)
        false_fn()
        results = get_state()
        return _assert_results(results, symbol_names)

    results = functional_cond()
    set_state(results)


def _assert_iteration_inputs(inputs, symbol_names):
    """Assert that all loop carried values, variables that are updated each iteration or accessed after the
    loop terminates, are initialized prior to entering the loop.

    The reason is two-fold:
      - the type information from those variables is required for tracing
      - we want to avoid accessing a variable that is uninitialized, or uninitialized in a subset
        of execution paths

    Additionally, these types need to be valid JAX types.

    Args:
        inputs (Tuple): The loop carried values
        symbol_names (Tuple[str]): The names of the loop carried values.
    """

    if not has_jax:  # pragma: no cover
        raise ImportError("autograph capture requires JAX to be installed.")

    for i, inp in enumerate(inputs):
        if isinstance(inp, Undefined):
            raise AutoGraphError(
                f"The variable '{inp}' is potentially uninitialized:\n"
                " - you may have forgotten to initialize it prior to accessing it inside a loop, or"
                "\n"
                " - you may be attempting to access a variable local to the body of a loop in an "
                "outer scope.\n"
                f"Please ensure '{inp}' is initialized with a value before entering the loop."
            )

        try:
            jax.api_util.shaped_abstractify(inp)
        except TypeError as e:
            raise AutoGraphError(
                f"The variable '{symbol_names[i]}' was initialized with type {type(inp)}, "
                "which is not compatible with JAX. Typically, this is the case for non-numeric "
                "values.\n"
                "You may still use such a variable as a constant inside a loop, but it cannot "
                "be updated from one iteration to the next, or accessed outside the loop scope "
                "if it was defined inside of it."
            ) from e


def _assert_iteration_results(inputs, outputs, symbol_names):
    """The results of a for loop should have the identical type as the inputs since they are
    "passed" as inputs to the next iteration. A mismatch here may indicate that a loop-carried
    variable was initialized with the wrong type.
    """

    for i, (inp, out) in enumerate(zip(inputs, outputs)):
        inp_t, out_t = jax.api_util.shaped_abstractify(inp), jax.api_util.shaped_abstractify(out)
        if inp_t.dtype != out_t.dtype or inp_t.shape != out_t.shape:
            raise AutoGraphError(
                f"The variable '{symbol_names[i]}' was initialized with the wrong type, or you may "
                f"be trying to change its type from one iteration to the next. "
                f"Expected: {out_t}, Got: {inp_t}"
            )


# pylint: disable=too-many-positional-arguments
def _call_pennylane_for(
    start,
    stop,
    step,
    body_fn,
    get_state,
    set_state,
    symbol_names,
    enum_start=None,
    array_iterable=None,
):
    """Dispatch to a PennyLane implementation of for loops."""

    # Ensure iteration arguments are properly initialized. We cannot process uninitialized
    # loop carried values as we need their type information for tracing.
    init_iter_args = get_state()
    _assert_iteration_inputs(init_iter_args, symbol_names)

    @qml.for_loop(start, stop, step)
    def functional_for(i, *iter_args):
        # Assign tracers to the iteration variables identified by AutoGraph (iter_args in mlir).
        set_state(iter_args)

        # The iteration index/element (for <...> in) is already handled by the body function, e.g.:
        #   def body_fn(itr):
        #     i, x = itr
        #     ...
        if enum_start is None and array_iterable is None:
            # for i in range(..)
            body_fn(i)
        elif enum_start is None:
            # for x in array
            body_fn(array_iterable[i])
        else:
            # for (i, x) in enumerate(array)
            body_fn((i + enum_start, array_iterable[i]))

        return get_state()

    final_iter_args = functional_for(*init_iter_args)
    _assert_iteration_results(init_iter_args, final_iter_args, symbol_names)
    return final_iter_args


def for_stmt(
    iteration_target: Any,
    _extra_test: Callable[[], bool] | None,
    body_fn: Callable[[int], None],
    get_state: Callable[[], tuple],
    set_state: Callable[[tuple], None],
    symbol_names: tuple[str],
    _opts: dict,
):
    """An implementation of the AutoGraph 'for .. in ..' statement. The interface is defined by
    AutoGraph, here we merely provide an implementation of it in terms of PennyLane primitives."""

    assert _extra_test is None

    # The general approach is to convert as much code as possible into a graph-based form:
    # - For loops over iterables will attempt a conversion of the iterable to array
    # - For loops over a Python range will be converted to a native PennyLane for_loop. The now
    #   dynamic iteration variable can cause issues in downstream user code that raise an error.
    # - For loops over a Python enumeration use a combination of the above, providing a dynamic
    #   iteration variable and conversion of the iterable to array.

    # Any of these could fail depending on the compatibility of the user code. A failure could
    # also occur because an exception is raised during the tracing of the loop body after conversion
    # (for example because the user forgot to use a list instead of an array)
    # The PennyLane autograph implementation does not currently fall back to a Python loop in this case,
    # but this has been implemented in Catalyst and could be extended to this. It does, however, require an
    # active queuing context.

    exception_raised = None
    init_state = get_state()
    assert len(init_state) == len(symbol_names)

    if isinstance(iteration_target, PRange):
        start, stop, step = iteration_target.get_raw_range()
        enum_start = None
        iteration_array = None
    elif isinstance(iteration_target, PEnumerate):
        start, stop, step = 0, len(iteration_target.iteration_target), 1
        enum_start = iteration_target.start_idx
        try:
            iteration_array = jnp.asarray(iteration_target.iteration_target)
        except Exception as e:  # pylint: disable=broad-exception-caught,broad-except
            exception_raised = e
    else:
        start, stop, step = 0, len(iteration_target), 1
        enum_start = None
        try:
            iteration_array = jnp.asarray(iteration_target)
        except Exception as e:  # pylint: disable=broad-exception-caught,broad-except
            exception_raised = e

    if exception_raised:

        raise AutoGraphError(
            f"Could not convert the iteration target {iteration_target} to array while processing "
            f"a for-loop with AutoGraph."
        ) from exception_raised

    try:
        set_state(init_state)
        results = _call_pennylane_for(
            start,
            stop,
            step,
            body_fn,
            get_state,
            set_state,
            symbol_names,
            enum_start,
            iteration_array,
        )
    except Exception as e:
        # pylint: disable=import-outside-toplevel
        import textwrap

        raise AutoGraphError(
            f"Tracing of an AutoGraph converted for loop failed with an exception:\n"
            f"  {type(e).__name__}:{textwrap.indent(str(e), '    ')}\n"
            f"\n"
            f"Make sure that loop variables are not used in tracing-incompatible ways, for instance "
            f"by indexing a Python list with it (rather than a JAX array). Also ensure all variables "
            f"are initialized before the loop begins, and that they don't change type across iterations.\n"
            f"To understand different types of JAX tracing errors, please refer to the guide at: "
            f"https://jax.readthedocs.io/en/latest/errors.html"
        ) from e

    set_state(results)


def _call_pennylane_while(loop_test, loop_body, get_state, set_state, symbol_names):
    """Dispatch to a PennyLane implementation of while loops."""

    init_iter_args = get_state()
    _assert_iteration_inputs(init_iter_args, symbol_names)

    def test(state):
        old = get_state()
        set_state(state)
        res = loop_test()
        set_state(old)
        return res

    @qml.while_loop(test)
    def functional_while(iter_args):
        set_state(iter_args)
        loop_body()
        return get_state()

    final_iter_args = functional_while(init_iter_args)

    return final_iter_args


def while_stmt(loop_test, loop_body, get_state, set_state, symbol_names, _opts):
    """An implementation of the AutoGraph 'while ..' statement. The interface is defined by
    AutoGraph, here we merely provide an implementation of it in terms of PennyLane primitives."""

    results = _call_pennylane_while(loop_test, loop_body, get_state, set_state, symbol_names)
    set_state(results)


def _logical_op(*args, jax_fn: Callable, python_fn: Callable):
    """A helper function to implement logical operations in a way that is compatible with both
    JAX and Python. It checks if any of the arguments are undefined, and raises an error if so.
    Otherwise, it applies the specified logical operation using either JAX or Python functions."""

    values = [arg() if callable(arg) else arg for arg in args]

    if any(qml.math.is_abstract(val) for val in values):
        result = jax_fn(*values)
    else:
        result = python_fn(*values)

    return result


def and_(a, b):
    """A wrapper for the AutoGraph 'and' operator. It returns the result of the logical 'and'
    operation between two values, `a` and `b`. If either value is undefined, it raises an error."""
    return _logical_op(a, b, jax_fn=jax.numpy.logical_and, python_fn=lambda x, y: x and y)


def or_(a, b):
    """A wrapper for the AutoGraph 'or' operator. It returns the result of the logical 'or'
    operation between two values, `a` and `b`. If either value is undefined, it raises an error."""
    return _logical_op(a, b, jax_fn=jax.numpy.logical_or, python_fn=lambda x, y: x or y)


def not_(a):
    """A wrapper for the AutoGraph 'not' operator. It returns the result of the logical 'not'
    operation on a value `a`. If `a` is undefined, it raises an error."""
    return _logical_op(a, jax_fn=jax.numpy.logical_not, python_fn=lambda x: not x)


# Prevent autograph from converting PennyLane and Catalyst library code, this can lead to many
# issues such as always tracing through code that should only be executed conditionally. We might
# have to be even more restrictive in the future to prevent issues if necessary.
module_allowlist = (
    ag_config.DoNotConvert("pennylane"),
    ag_config.DoNotConvert("catalyst"),
    ag_config.DoNotConvert("optax"),
    ag_config.DoNotConvert("jax"),
    *ag_config.CONVERSION_RULES,
)


class Patcher:
    """Patcher, a class to replace object attributes.

    Args:
        patch_data: List of triples. The first element in the triple corresponds to the object
        whose attribute is to be replaced. The second element is the attribute name. The third
        element is the new value assigned to the attribute.
    """

    def __init__(self, *patch_data):
        self.backup = {}
        self.patch_data = patch_data

        assert all(len(data) == 3 for data in patch_data)

    def __enter__(self):
        for obj, attr_name, fn in self.patch_data:
            self.backup[(obj, attr_name)] = getattr(obj, attr_name)
            setattr(obj, attr_name, fn)

    def __exit__(self, _type, _value, _traceback):
        for obj, attr_name, _ in self.patch_data:
            setattr(obj, attr_name, self.backup[(obj, attr_name)])


def converted_call(fn, args, kwargs, caller_fn_scope=None, options=None):
    """A wrapper for the autograph ``converted_call`` function, imported here as
    ``ag_converted_call``. It returns the result of executing a possibly-converted
     function ``fn`` with the specified ``args`` and ``kwargs``.

     We want AutoGraph to use its standard behaviour with a few exceptions:

       1. We want to use our own instance of the AST transformer when
           recursively transforming functions
       2. We want to ignore certain PennyLane modules and functions when
           converting (i.e. don't let autograph convert them)
       3. We want to handle QNodes, while AutoGraph generally only works on
           functions, and to handle PennyLane wrapper functions like ctrl
           and adjoint
    """

    # TODO: eliminate the need for patching by improving the autograph interface
    with Patcher(
        (ag_api, "_TRANSPILER", qml.capture.autograph.transformer.TRANSFORMER),
        (ag_config, "CONVERSION_RULES", module_allowlist),
        (ag_py_builtins, "BUILTIN_FUNCTIONS_MAP", py_builtins_map),
    ):
        # HOTFIX: pass through calls of known PennyLane wrapper functions
        if fn in (
            qml.adjoint,
            qml.ctrl,
            qml.grad,
            qml.jacobian,
            qml.vjp,
            qml.jvp,
        ):
            if not args:
                raise ValueError(f"{fn.__name__} requires at least one argument")

            is_abstract_operator = qml.math.is_abstract(args[0]) and isinstance(
                args[0].aval, qml.capture.primitives.AbstractOperator
            )
            # If first argument is already an operator, pass it through directly
            if isinstance(args[0], qml.operation.Operator) or (
                is_abstract_operator and fn in {qml.adjoint, qml.ctrl}
            ):
                return ag_converted_call(fn, args, kwargs, caller_fn_scope, options)

            # Otherwise, handle the callable case
            wrapped_fn = args[0]
            if not callable(wrapped_fn):
                raise ValueError(
                    f"First argument to {fn.__name__} must be callable or an Operation"
                )

            @functools.wraps(wrapped_fn)
            def passthrough_wrapper(*args, **kwargs):
                return converted_call(wrapped_fn, args, kwargs, caller_fn_scope, options)

            return fn(
                passthrough_wrapper,
                *args[1:],
                **(kwargs if kwargs is not None else {}),
            )

        # For QNode calls, we employ a wrapper to forward the quantum function call to autograph
        if isinstance(fn, qml.QNode):

            @functools.wraps(fn.func)
            def qnode_call_wrapper():
                return ag_converted_call(fn.func, args, kwargs, caller_fn_scope, options)

            # Copy the original qnode but replace its function.
            new_qnode = copy.copy(fn)
            new_qnode.func = qnode_call_wrapper
            return new_qnode()

        return ag_converted_call(fn, args, kwargs, caller_fn_scope, options)


class PRange:
    """PennyLane range object. This class re-implements the built-in range class
    (which can't be inherited from). The only change is saving and accessing the
    inputs directly, to circumvent some JAX-unfriendly code in the Python range.
    """

    def __init__(self, start_stop, stop=None, step=None):
        self._py_range = None
        self._start = start_stop if stop is not None else 0
        self._stop = stop if stop is not None else start_stop
        self._step = step if step is not None else 1

    def get_raw_range(self):
        """Get the raw values defining this range: start, stop, step."""
        return self._start, self._stop, self._step

    @property
    def py_range(self):
        """Access the underlying Python range object. If it doesn't exist, create one."""
        if self._py_range is None:
            self._py_range = range(self._start, self._stop, self._step)
        return self._py_range

    # Interface of the Python range class.
    # pylint: disable=missing-function-docstring

    @property
    def start(self) -> int:  # pragma: no cover
        return self.py_range.start

    @property
    def stop(self) -> int:  # pragma: no cover
        return self.py_range.stop

    @property
    def step(self) -> int:  # pragma: no cover
        return self.py_range.step

    def count(self, __value: int) -> int:  # pragma: no cover
        return self.py_range.count(__value)

    def index(self, __value: int) -> int:  # pragma: no cover
        return self.py_range.index(__value)

    def __len__(self) -> int:  # pragma: no cover
        return self.py_range.__len__()

    def __eq__(self, __value: object) -> bool:  # pragma: no cover
        return self.py_range.__eq__(__value)

    def __hash__(self) -> int:  # pragma: no cover
        return self.py_range.__hash__()

    def __contains__(self, __key: object) -> bool:  # pragma: no cover
        return self.py_range.__contains__(__key)

    def __iter__(self) -> Iterator[int]:  # pragma: no cover
        return self.py_range.__iter__()

    def __getitem__(self, __key: SupportsIndex | slice) -> int | range:  # pragma: no cover
        return self.py_range.__getitem__(__key)

    def __reversed__(self) -> Iterator[int]:  # pragma: no cover
        return self.py_range.__reversed__()


# pylint: disable=too-few-public-methods
class PEnumerate(enumerate):
    """PennyLane enumeration object. Inherits from Python ``enumerate``, but adds storing the
    input iteration_target and start_idx, which are used by the for-loop conversion.
    """

    def __init__(self, iterable, start=0):

        # TODO: original enumerate constructor cannot be called as it causes some tests to break
        self.iteration_target = iterable
        self.start_idx = start


py_builtins_map = {
    **ag_py_builtins.BUILTIN_FUNCTIONS_MAP,
    "range": PRange,
    "enumerate": PEnumerate,
}
