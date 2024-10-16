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
This module provides the implementation of AutoGraph primitives in terms of traceable PennyLane
functions. The purpose is to convert imperative style code to functional or graph-style code.
"""
import copy
import functools
from typing import Any, Callable, Tuple

from malt.core import config as ag_config
from malt.impl import api as ag_api
from malt.impl.api import converted_call as ag_converted_call
from malt.operators.variables import Undefined

import pennylane as qml

from .utils import AutoGraphError, Patcher

__all__ = [
    "if_stmt",
    "converted_call",
]


def assert_results(results, var_names):
    """Assert that none of the results are undefined, i.e. have no value."""

    assert len(results) == len(var_names)

    for r, v in zip(results, var_names):
        if isinstance(r, Undefined):
            raise AutoGraphError(f"Some branches did not define a value for variable '{v}'")

    return results


# pylint: disable=too-many-arguments
def if_stmt(
    pred: bool,
    true_fn: Callable[[], Any],
    false_fn: Callable[[], Any],
    get_state: Callable[[], Tuple],
    set_state: Callable[[Tuple], None],
    symbol_names: Tuple[str],
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
        return assert_results(results, symbol_names)

    @functional_cond.otherwise
    def functional_cond():
        set_state(init_state)
        false_fn()
        results = get_state()
        return assert_results(results, symbol_names)

    results = functional_cond()
    set_state(results)


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


def converted_call(fn, args, kwargs, caller_fn_scope=None, options=None):
    """We want AutoGraph to use our own instance of the AST transformer when recursively
    transforming functions, but otherwise duplicate the same behaviour."""

    # TODO: eliminate the need for patching by improving the autograph interface
    with Patcher(
        (ag_api, "_TRANSPILER", qml.capture.autograph.transformer.TRANSFORMER),
        (ag_config, "CONVERSION_RULES", module_allowlist),
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
            assert args and callable(args[0])
            wrapped_fn = args[0]

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
