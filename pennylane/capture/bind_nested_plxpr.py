# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

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
This submodule provides a decorator for binding a qfunc transform as a PLXPR primitive.
"""
from functools import partial, wraps
from typing import Callable

from .switches import enabled

has_jax = True
try:
    import jax
except ImportError:
    has_jax = False

QFunc = Callable
QFuncTransform = Callable[[QFunc], QFunc]


def bind_nested_plxpr(fn: QFuncTransform) -> QFuncTransform:
    """Allows a qfunc transform to become a plxpr primitive.

    Args:
        fn (Callable): a function from a qfunc to a qfunc. May accept arbitrary arguments
            and keyword arguments.

    **Example:**

    Suppose we have a transform that takes a quantum function and repeats it ``n`` times:

    >>> @qml.capture.bind_nested_plxpr
    ... def repeat_qfunc(qfunc, n=1):
    ...    def new_qfunc(*args, **kwargs):
    ...         for _ in range(n):
    ...             qfunc(*args, **kwargs)
    ...    return new_qfunc

    Once we place use this qfunc in a workflow, we can now both execute it and convert it to jaxpr.

    >>> def workflow(x):
    ...     repeat_qfunc(qml.RX, n=3)(x, wires=1)
    >>> jax.make_jaxpr(workflow)(0.1)
    { lambda ; a:f32[]. let
        _:AbstractOperator() = repeat_qfunc[
        fn_kwargs={'n': 3}
        jaxpr={ lambda ; b:f32[]. let
            c:AbstractOperator() = RX[n_wires=1] b 1
            in (c,) }
        n_args=1
        ] a
    in () }

    Note that ``fn_kwargs`` stores the ``n=3`` keyword arguments to the qfunc transform.  ``n_args=1`` indicates that the qfunc accepted
    a single argument.  This is used to split the arguments between the transform and the qfunc. For example, we could have instead passed ``n``
    positionally to the transform:

    >>> def workflow2(x):
    ...     repeat_qfunc(qml.RX, 3)(x, wires=1)
    >>> jax.make_jaxpr(workflow2)(0.1)
    { lambda ; a:f32[]. let
        _:AbstractOperator() = repeat_qfunc[
        fn_kwargs={}
        jaxpr={ lambda ; b:f32[]. let
            c:AbstractOperator() = RX[n_wires=1] b 1
            in (c,) }
        n_args=1
        ] a 3
    in () }

    The workflow can still be executed as normal without jitting:

    >>> with qml.queuing.AnnotatedQueue() as q:
    ...     workflow(0.1)
    >>> q.queue
    RX(0.1, wires=[1]), RX(0.1, wires=[1]), RX(0.1, wires=[1])]

    And the produced jaxpr can still be evaluated:

    >>> jaxpr = jax.make_jaxpr(workflow)(0.1)
    >>> with qml.queuing.AnnotatedQueue() as q:
    ...     jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.1)
    >>> q.queue
    [RX(0.1, wires=[1]), RX(0.1, wires=[1]), RX(0.1, wires=[1])]

    """
    if not has_jax:
        return fn

    prim = jax.core.Primitive(fn.__name__)
    prim.multiple_results = True

    @prim.def_abstract_eval
    def _(*_, jaxpr, **__):
        return jaxpr.out_avals

    @prim.def_impl
    def _(*total_args, jaxpr, n_args, fn_kwargs=None):
        qfunc_args = total_args[:n_args]
        fn_args = total_args[n_args:]

        # the new qfunc, created by evaluating the jaxpr.
        bound = partial(jax.core.eval_jaxpr, jaxpr.jaxpr)

        fn_kwargs = fn_kwargs or {}
        return [fn(bound, *fn_args, **fn_kwargs)(jaxpr.consts, *qfunc_args)]

    @wraps(fn)
    def new_fn(qfunc: QFunc, *fn_args, **fn_kwargs) -> QFunc:
        if not enabled():
            return fn(qfunc, *fn_args, **fn_kwargs)

        @wraps(qfunc)
        def new_qfunc(*args, **kwargs):
            jaxpr = jax.make_jaxpr(partial(qfunc, **kwargs))(*args)
            n_args = len(args)
            return prim.bind(*args, *fn_args, jaxpr=jaxpr, n_args=n_args, fn_kwargs=fn_kwargs)

        return new_qfunc

    new_fn.primitive = prim

    return new_fn
