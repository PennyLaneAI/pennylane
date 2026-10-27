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
"""
This submodule defines the ``compute_uncompute`` context manager.

ROUGH PROTOTYPE: relies on private JAX tracing internals (``jax._src``) to capture the body of a
``with`` block as a nested jaxpr.
"""

from pennylane import capture
from pennylane.capture.custom_primitives import QpPrimitive
from pennylane.capture.op_eqns import pop_op_eqns
from pennylane.core.queuing import AnnotatedQueue

compute_uncompute_prim = QpPrimitive("compute_uncompute")
compute_uncompute_prim.multiple_results = True
compute_uncompute_prim.prim_type = "higher_order"


@compute_uncompute_prim.def_abstract_eval
def _compute_uncompute_abstract_eval(*_, **__):
    return []


@compute_uncompute_prim.def_impl
def _compute_uncompute_impl(*invals, **params):
    # pylint: disable=import-outside-toplevel
    from pennylane.tape.plxpr_conversion import CollectOpsandMeas, _compute_uncompute_prim

    collector = CollectOpsandMeas()
    collector.setup()
    # Constructing the compute-uncompute operator (with capture paused) queues it.
    _compute_uncompute_prim(collector, *invals, **params)
    return []


def split_consts(invals, n_consts):
    """Split the flat inputs of a ``compute_uncompute`` equation into per-region consts."""
    out, start = [], 0
    for n in n_consts:
        out.append(invals[start : start + n])
        start += n
    return out


def combine_ops(ops):
    """Combine operators applied in order into a single operator (or ``None`` if empty)."""
    from pennylane.ops.op_math import prod  # pylint: disable=import-outside-toplevel

    if not ops:
        return None
    return ops[0] if len(ops) == 1 else prod(*ops[::-1])


class _SubTrace:
    """Open a nested JAX trace on enter and turn it into a jaxpr on exit.

    Tracers from the enclosing trace that are used inside become consts of the nested jaxpr.
    """

    # pylint: disable=import-outside-toplevel, attribute-defined-outside-init
    def __enter__(self):
        from jax._src import core
        from jax._src.interpreters import partial_eval as pe

        self._dbg = core.DebugInfo(
            "compute_uncompute", "<compute_uncompute block>", None, lambda: ()
        )
        self.trace = pe.DynamicJaxprTrace(self._dbg, parent_trace=core.trace_ctx.trace)
        self._ctx = core.set_current_trace(self.trace)
        self._ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        from jax._src import source_info_util

        self._ctx.__exit__(exc_type, exc_value, tb)
        if exc_type is None:
            self.jaxpr, self.consts = self.trace.frame.to_jaxpr(
                self.trace, [], self._dbg, source_info_util.current()
            )
            if not self.jaxpr.eqns:
                self.jaxpr = None
        return False


def _trace_op(op):
    """Move the equation of an already-captured ``Operator2`` into its own nested jaxpr."""
    pop_op_eqns((op,))
    with _SubTrace() as sub:
        op._bind_primitive()  # pylint: disable=protected-access
    return sub


class compute_uncompute:  # pylint: disable=invalid-name, too-few-public-methods
    """Context manager for a compute-uncompute block.

    The operators applied inside the ``with`` block form the target. On exit, the block becomes a
    single compute-uncompute operator (the same operator :func:`~.change_op_basis` produces).

    Args:
        compute_block (Operator): the compute operator
        uncompute_block (Operator | None): the uncompute operator. ``None`` (default) means
            ``qp.adjoint(compute_block)``.

    With program capture enabled, the block is captured as a single higher-order
    ``compute_uncompute`` primitive with the nested jaxprs ``compute_jaxpr``, ``target_jaxpr``
    and ``uncompute_jaxpr``. ``uncompute_jaxpr=None`` represents the default adjoint.

    **Example**

    >>> import jax
    >>> qp.capture.enable()
    >>> def f(x):
    ...     with qp.compute_uncompute(qp.RY(x, 1)):
    ...         qp.RX(2 * x, 1)
    ...         qp.CNOT([1, 2])
    >>> jaxpr = jax.make_jaxpr(f)(0.5)
    >>> jaxpr.eqns[0].primitive
    compute_uncompute
    """

    def __init__(self, compute_block, uncompute_block=None):
        self.compute_block = compute_block
        self.uncompute_block = uncompute_block
        self._capture = None

    def __enter__(self):
        self._capture = capture.enabled()
        if self._capture:
            self._compute = _trace_op(self.compute_block)
            self._uncompute = (
                _trace_op(self.uncompute_block) if self.uncompute_block is not None else None
            )
            self._body = _SubTrace().__enter__()
        else:
            self._queue = AnnotatedQueue().__enter__()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if self._capture:
            self._body.__exit__(exc_type, exc_value, tb)
            if exc_type is not None:
                return False
            if self._body.jaxpr is None:
                raise ValueError("compute_uncompute requires at least one operation in its body.")
            regions = [self._compute, self._body]
            if self._uncompute is not None:
                regions.append(self._uncompute)
            compute_uncompute_prim.bind(
                *(c for r in regions for c in r.consts),
                compute_jaxpr=self._compute.jaxpr,
                target_jaxpr=self._body.jaxpr,
                uncompute_jaxpr=self._uncompute.jaxpr if self._uncompute is not None else None,
                n_consts=tuple(len(r.consts) for r in regions),
            )
            return False

        self._queue.__exit__(exc_type, exc_value, tb)
        if exc_type is not None:
            return False
        target = combine_ops(self._queue.queue)
        if target is None:
            raise ValueError("compute_uncompute requires at least one operation in its body.")
        from pennylane.ops.op_math import change_op_basis  # pylint: disable=import-outside-toplevel

        change_op_basis(self.compute_block, target, self.uncompute_block)
        return False
