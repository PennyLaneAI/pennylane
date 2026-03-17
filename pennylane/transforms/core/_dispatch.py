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
This module defines registered dispatches of a `Transform` object.
"""

from collections.abc import Callable, Sequence
from functools import lru_cache, singledispatch, wraps

from pennylane import capture, math
from pennylane.capture import autograph
from pennylane.exceptions import TransformError
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operator
from pennylane.pytrees import flatten
from pennylane.queuing import AnnotatedQueue, QueuingManager, apply
from pennylane.tape import QuantumScript
from pennylane.typing import ResultBatch

from .transform import BoundTransform, Transform


def specific_apply_transform(transform, obj, *targs, **tkwargs):
    """The default behavior for Transform._apply_transform. By default, it dispatches to the
    generic registration."""
    return transform.generic_apply_transform(obj, *targs, **tkwargs)


@singledispatch
def generic_apply_transform(obj, transform, *targs, **tkwargs):
    """Apply an generic transform to a specific type of object. A singledispatch function
    used by ``TransformDipsatcher.generic_apply_transform``, but with a different order of arguments
    to allow is to be used by singledispatch.

    When called with an object that is not a valid dispatch target (e.g., not a QNode, tape, etc.),
    this returns a BoundTransform with the supplied args and kwargs. This enables patterns like:

        decompose(gate_set=gate_set) + merge_rotations(1e-6)

    where transforms are called with just configuration parameters and combined into a CompilePipeline
    """
    # If the first argument is not a valid dispatch target, return a BoundTransform
    # with the first argument and any additional args/kwargs stored as transform parameters.
    targs, tkwargs = transform.setup_inputs(obj, *targs, **tkwargs)
    return BoundTransform(transform, args=targs, kwargs=tkwargs)


@lru_cache
def _create_transform_primitive():

    transform_prim = capture.QmlPrimitive("transform")
    transform_prim.multiple_results = True
    transform_prim.prim_type = "transform"

    # pylint: disable=too-many-arguments, disable=unused-argument
    @transform_prim.def_impl
    def _impl(*all_args, inner_jaxpr, args_slice, consts_slice, **_):
        args = all_args[slice(*args_slice)]
        consts = all_args[slice(*consts_slice)]
        return capture.eval_jaxpr(inner_jaxpr, consts, *args)

    @transform_prim.def_abstract_eval
    def _abstract_eval(*_, inner_jaxpr, **__):
        return [out.aval for out in inner_jaxpr.outvars]

    return transform_prim


@Transform.generic_register
def apply_to_callable(obj: Callable, transform, *targs, **tkwargs):
    """Apply a transform to a Callable object."""
    if obj.__class__.__name__ == "QJIT":
        raise TransformError(
            "Functions that are wrapped / decorated with qjit cannot subsequently be"
            f" transformed with a PennyLane transform (attempted {transform})."
            f" For the desired affect, ensure that qjit is applied after {transform}."
        )
    targs, tkwargs = transform.setup_inputs(*targs, **tkwargs)

    @wraps(obj)
    def qfunc_transformed(*args, **kwargs):
        if capture.enabled():
            return _capture_apply(obj, transform, *targs, **tkwargs)(*args, **kwargs)

        # removes the argument to the qfuncs from the active queuing context.
        leaves, _ = flatten((args, kwargs), lambda obj: isinstance(obj, Operator))
        for l in leaves:
            if isinstance(l, Operator):
                QueuingManager.remove(l)

        with AnnotatedQueue() as q:
            qfunc_output = obj(*args, **kwargs)

        tape = QuantumScript.from_queue(q)

        with QueuingManager.stop_recording():
            if transform.is_informative:
                transformed_tapes, processing_fn = transform.tape_transform(tape, *targs, **tkwargs)
            else:
                transformed_tapes, processing_fn = transform(tape, *targs, **tkwargs)

        if len(transformed_tapes) != 1:
            raise TransformError(
                "Impossible to dispatch your transform on quantum function, because more than "
                "one tape is returned"
            )

        transformed_tape = transformed_tapes[0]

        if transform.is_informative:
            return processing_fn(transformed_tapes)

        for op in transformed_tape.operations:
            apply(op)

        mps = [apply(mp) for mp in transformed_tape.measurements]

        if not mps:
            return qfunc_output

        if isinstance(qfunc_output, MeasurementProcess):
            return tuple(mps) if len(mps) > 1 else mps[0]

        if isinstance(qfunc_output, (tuple, list)):
            return type(qfunc_output)(mps)

        interface = math.get_interface(qfunc_output)
        return math.asarray(mps, like=interface)

    return qfunc_transformed


@Transform.generic_register
def _apply_to_sequence(obj: Sequence, transform, *targs, **tkwargs):
    if not all(isinstance(t, QuantumScript) for t in obj):
        # not a sequence of quantum script, treat as first argument
        targs, tkwargs = transform.setup_inputs(obj, *targs, **tkwargs)
        return BoundTransform(transform, args=targs, kwargs=tkwargs)
    targs, tkwargs = transform.setup_inputs(*targs, **tkwargs)
    execution_tapes = []
    batch_fns = []
    tape_counts = []

    for t in obj:
        # Preprocess the tapes by applying transforms
        # to each tape, and storing corresponding tapes
        # for execution, processing functions, and list of tape lengths.
        new_tapes, fn = transform(t, *targs, **tkwargs)
        execution_tapes.extend(new_tapes)
        batch_fns.append(fn)
        tape_counts.append(len(new_tapes))

    def processing_fn(res: ResultBatch) -> ResultBatch:
        """Applies a batch of post-processing functions to results.

        Args:
            res (ResultBatch): the results of executing a batch of circuits.

        Returns:
            ResultBatch: results that have undergone classical post processing.

        Closure variables:
            tape_counts: the number of tapes outputted from each application of the transform.
            batch_fns: the post processing functions to apply to each sub-batch.

        """
        count = 0
        final_results = []

        for f, s in zip(batch_fns, tape_counts):
            # apply any batch transform post-processing
            new_res = f(res[count : count + s])
            final_results.append(new_res)
            count += s

        return tuple(final_results)

    return tuple(execution_tapes), processing_fn


def _capture_apply(obj, transform, *targs, **tkwargs):
    @autograph.wraps(obj)
    def qfunc_transformed(*args, **kwargs):
        import jax  # pylint: disable=import-outside-toplevel

        flat_qfunc = capture.flatfn.FlatFn(obj)
        jaxpr = jax.make_jaxpr(flat_qfunc)(*args, **kwargs)
        flat_args = jax.tree_util.tree_leaves(args)

        n_args = len(flat_args)
        n_consts = len(jaxpr.consts)
        args_slice = slice(0, n_args)
        consts_slice = slice(n_args, n_args + n_consts)
        targs_slice = slice(n_args + n_consts, None)

        results = _create_transform_primitive().bind(  # pylint: disable=protected-access
            *flat_args,
            *jaxpr.consts,
            *targs,
            inner_jaxpr=jaxpr.jaxpr,
            args_slice=args_slice,
            consts_slice=consts_slice,
            targs_slice=targs_slice,
            tkwargs=tkwargs,
            transform=transform,
        )

        assert flat_qfunc.out_tree is not None
        return jax.tree_util.tree_unflatten(flat_qfunc.out_tree, results)

    return qfunc_transformed


@Transform.generic_register
def _apply_to_tape(obj: QuantumScript, transform, *targs, **tkwargs):
    if transform.tape_transform is None:
        raise NotImplementedError(
            f"Transform {transform} has no defined tape implementation, "
            "and can only be applied when decorating the entire workflow "
            "with '@qml.qjit' and when it is placed after all transforms "
            "that only have a tape implementation."
        )
    targs, tkwargs = transform.setup_inputs(*targs, **tkwargs)
    if transform.expand_transform:
        expanded_tapes, expand_processing = transform.expand_transform(obj, *targs, **tkwargs)
        transformed_tapes = []
        processing_and_slices = []
        start = 0
        for tape in expanded_tapes:
            intermediate_tapes, post_processing_fn = transform.tape_transform(
                tape, *targs, **tkwargs
            )
            transformed_tapes.extend(intermediate_tapes)
            end = start + len(intermediate_tapes)
            processing_and_slices.append(tuple([post_processing_fn, slice(start, end)]))
            start = end

        def processing_fn(results):
            processed_results = [fn(results[slice]) for fn, slice in processing_and_slices]
            return expand_processing(processed_results)

    else:
        transformed_tapes, processing_fn = transform.tape_transform(obj, *targs, **tkwargs)

    if transform.is_informative:
        return processing_fn(transformed_tapes)

    return transformed_tapes, processing_fn
