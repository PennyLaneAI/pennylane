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
This module contains the transform dispatcher and the transform container.
"""
import functools
import os
import warnings
from collections.abc import Callable, Sequence
from copy import copy

import pennylane as qml
from pennylane import capture, math
from pennylane.exceptions import TransformError
from pennylane.operation import classproperty
from pennylane.queuing import AnnotatedQueue, QueuingManager, apply
from pennylane.tape import QuantumScript
from pennylane.typing import ResultBatch


def _create_transform_primitive(name):
    try:
        # pylint: disable=import-outside-toplevel
        from pennylane.capture.custom_primitives import QmlPrimitive
    except ImportError:
        return None

    transform_prim = QmlPrimitive(name + "_transform")
    transform_prim.multiple_results = True
    transform_prim.prim_type = "transform"

    @transform_prim.def_impl
    def _(
        *all_args, inner_jaxpr, args_slice, consts_slice, targs_slice, tkwargs
    ):  # pylint: disable=unused-argument
        args = all_args[args_slice]
        consts = all_args[consts_slice]
        return capture.eval_jaxpr(inner_jaxpr, consts, *args)

    @transform_prim.def_abstract_eval
    def _(*_, inner_jaxpr, **__):
        return [out.aval for out in inner_jaxpr.outvars]

    return transform_prim


def _create_plxpr_fallback_transform(tape_transform):
    # pylint: disable=import-outside-toplevel
    try:
        import jax
    except ImportError:
        return None

    def plxpr_fallback_transform(jaxpr, consts, targs, tkwargs, *args):

        def wrapper(*inner_args):
            tape = qml.tape.plxpr_to_tape(jaxpr, consts, *inner_args)
            with capture.pause():
                tapes, _ = tape_transform(tape, *targs, **tkwargs)

            if len(tapes) > 1:
                raise TransformError(
                    f"Cannot apply {tape_transform.__name__} transform with program "
                    "capture enabled. Only transforms that return a single QuantumTape "
                    "and null processing function are usable with program capture."
                )

            for op in tapes[0].operations:
                data, struct = jax.tree_util.tree_flatten(op)
                jax.tree_util.tree_unflatten(struct, data)

            out = []
            for mp in tapes[0].measurements:
                data, struct = jax.tree_util.tree_flatten(mp)
                out.append(jax.tree_util.tree_unflatten(struct, data))

            return tuple(out)

        abstracted_axes, abstract_shapes = capture.determine_abstracted_axes(args)
        return jax.make_jaxpr(wrapper, abstracted_axes=abstracted_axes)(*abstract_shapes, *args)

    return plxpr_fallback_transform


def _register_primitive_for_expansion(primitive, plxpr_transform):
    """Register a transform such that it can be expanded when applied to a function with
    program capture enabled."""
    # pylint: disable=import-outside-toplevel
    try:
        import jax

        from pennylane.capture.expand_transforms import ExpandTransformsInterpreter
    except ImportError:
        return

    @ExpandTransformsInterpreter.register_primitive(primitive)
    def _(
        self, *invals, inner_jaxpr, args_slice, consts_slice, targs_slice, tkwargs
    ):  # pylint: disable=too-many-arguments,missing-docstring
        args = invals[args_slice]
        consts = invals[consts_slice]
        targs = invals[targs_slice]

        def wrapper(*inner_args):
            return copy(self).eval(inner_jaxpr, consts, *inner_args)

        jaxpr = jax.make_jaxpr(wrapper)(*args)
        jaxpr = plxpr_transform(jaxpr.jaxpr, jaxpr.consts, targs, tkwargs, *args)
        return copy(self).eval(jaxpr.jaxpr, jaxpr.consts, *args)


def specific_apply_transform(obj, transform, *targs, **tkwargs):
    """The default behavior for TransformDispatcher._apply_transform. By default, it dispatches to the
    generic registration."""
    return transform.generic_apply_transform(obj, *targs, **tkwargs)


class TransformDispatcher:  # pylint: disable=too-many-instance-attributes
    r"""Converts a transform that has the signature ``(tape -> Sequence(tape), fn)`` to a transform dispatcher
    that can act on :class:`pennylane.tape.QuantumTape`, quantum function, :class:`pennylane.QNode`,
    :class:`pennylane.devices.Device`.

    .. warning::

        This class is developer-facing and should not be used directly. Instead, use
        :func:`qml.transform <pennylane.transform>` if you would like to make a custom
        transform.

    .. seealso:: :func:`~.pennylane.transform`
    """

    def __new__(cls, *args, **__):
        if os.environ.get("SPHINX_BUILD") == "1":
            # If called during a Sphinx documentation build,
            # simply return the original function rather than
            # instantiating the object. This allows the signature to
            # be correctly displayed in the documentation.

            warnings.warn(
                "Transforms have been disabled, as a Sphinx "
                "build has been detected via SPHINX_BUILD='1'. If this is not the "
                "case, please set the environment variable SPHINX_BUILD='0'.",
                UserWarning,
            )

            args[0].custom_qnode_transform = lambda x: x
            args[0].register = lambda x: x

            return args[0]

        return super().__new__(cls)

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        transform,
        expand_transform=None,
        classical_cotransform=None,
        is_informative=False,
        final_transform=False,
        use_argnum_in_expand=False,
        plxpr_transform=None,
    ):
        self._transform = transform
        self._expand_transform = expand_transform
        self._classical_cotransform = classical_cotransform
        self._is_informative = is_informative
        # is_informative supersedes final_transform
        self._final_transform = is_informative or final_transform
        self._custom_qnode_transform = None

        self._use_argnum_in_expand = use_argnum_in_expand
        functools.update_wrapper(self, transform)

        self._apply_transform = functools.singledispatch(specific_apply_transform)

        self._plxpr_transform = plxpr_transform or _create_plxpr_fallback_transform(self._transform)
        self._primitive = _create_transform_primitive(self._transform.__name__)
        _register_primitive_for_expansion(self._primitive, self._plxpr_transform)

    @property
    def register(self):
        """Returns a decorator for registering a specific application behavior for a given transform
        and a new class.

        .. code-block:: python

            @qml.transform
            def printer(tape):
                print("I have a tape: ", tape)
                return (tape, ), lambda x: x[0]

            @printer.register
            def _(obj: qml.operation.Operator, transform, *targs, **tkwargs):
                print("I have an operator:", obj)
                return obj

        >>> printer(qml.X(0))
        I have an operator: X(0)
        X(0)

        .. warning::

            Note that this function, while similar, requires a different signature than ``generic_register``.
            ``generic_register`` requires ``(self, obj: TypeHint)``, but this registration requires
            ``(obj: TypeHint, transform)``.

        """
        return self._apply_transform.register

    @functools.singledispatchmethod
    def generic_apply_transform(self, obj, *targs, **tkwargs):
        """Generic application of a transform to an object valid for all transforms."""
        raise TransformError(
            "Decorating a QNode with @transform_fn(**transform_kwargs) has been "
            "removed. Please decorate with @functools.partial(transform_fn, **transform_kwargs) "
            "instead, or call the transform directly using qnode = transform_fn(qnode, "
            "**transform_kwargs). Visit the deprecations page for more details: "
            "https://docs.pennylane.ai/en/stable/development/deprecations.html#completed-deprecation-cycles",
        )

    @classproperty
    def generic_register(cls):  # pylint: disable=no-self-argument
        """Returns a decorator for registering a default application behavior for a transform for a new class.

        Suppose I had a special class:

        .. code-block:: python

            class Subroutine:

                def __repr__(self):
                    return f"<Subroutine: {self.ops}>"

                def __init__(self, ops):
                    self.ops = ops

        I can register the application of a transform to ``Subroutine`` by doing:

        .. code-block:: python

            from pennylane.transforms.core import TransformDispatcher

            @TransformDispatcher.generic_register
            def apply_to_subroutine(self, obj: Subroutine, *targs, **tkwargs):
                tape = qml.tape.QuantumScript(obj.ops)
                batch, _ = self(tape, *targs, **tkwargs)
                return Subroutine(batch[0].operations)

        >>> qml.transforms.cancel_inverses(Subroutine([qml.Y(0), qml.X(0), qml.X(0)]))
        <Subroutine: [Y(0)]>

        .. warning::

            This function maps to the ``register`` method of a ``functools.singledispatchmethod`` object.
            In order to keep this working correctly, ``self`` *must* be the name of the first argument, and it
            *must not* have a type hint.

            .. code-block:: python

                def apply_to_subroutine(self: TransformDispatcher, obj: Subroutine, *targs, **tkwargs)

                def apply_to_subroutine(transform, obj: Subroutine, *targs, **tkwargs)

            will not work. The type can also be explicitly provided like:

            .. code-block:: python

                @TransformDispatcher.generic_register(Subroutine)
                def apply_to_subroutine(self, obj: Subroutine, *targs, **tkwargs):

            to more explicitly force registration for a given type.

        """
        return cls.generic_apply_transform.register

    def __call__(self, obj, *targs, **tkwargs):
        return self._apply_transform(obj, self, *targs, **tkwargs)

    def __repr__(self):
        return f"<transform: {self._transform.__name__}>"

    @property
    def transform(self):
        """The quantum transform."""
        return self._transform

    @property
    def expand_transform(self):
        """The expand transform."""
        return self._expand_transform

    @property
    def classical_cotransform(self):
        """The classical co-transform."""
        return self._classical_cotransform

    @property
    def plxpr_transform(self):
        """Function for transforming plxpr."""
        return self._plxpr_transform

    @property
    def is_informative(self):
        """``True`` if the transform is informative."""
        return self._is_informative

    @property
    def final_transform(self):
        """``True`` if the transformed tapes must be executed."""
        return self._final_transform

    def custom_qnode_transform(self, fn):
        """Register a custom QNode execution wrapper function
        for the batch transform.

        **Example**

        .. code-block:: python

            @transform
            def my_transform(tape, *targs, **tkwargs):
                ...
                return tapes, processing_fn

            @my_transform.custom_qnode_transform
            def my_custom_qnode_wrapper(self, qnode, targs, tkwargs):
                tkwargs = {**tkwargs, shots=100}
                return self.default_qnode_transform(qnode, targs, tkwargs)

        The custom QNode execution wrapper must have arguments
        ``self`` (the batch transform object), ``qnode`` (the input QNode
        to transform and execute), ``targs`` and ``tkwargs`` (the transform
        arguments and keyword arguments respectively).

        It should return a QNode that accepts the *same* arguments as the
        input QNode with the transform applied.

        The default :meth:`~.default_qnode_transform` method may be called
        if only pre- or post-processing dependent on QNode arguments is required.
        """
        # unfortunately, we don't have access to qml.QNode here, or in the places where
        # transforms are defining custom qnode transforms, so we still need to have this
        # "hold onto until later" approach
        # potentially can remove this patch by moving source code
        self._custom_qnode_transform = fn

    def default_qnode_transform(self, qnode, targs, tkwargs):
        """
        The default method that takes in a QNode and returns another QNode
        with the transform applied.
        """
        # same comment as custom_qnode_transform :(
        qnode = copy(qnode)

        if self.expand_transform:
            qnode.add_transform(
                TransformContainer(
                    self._expand_transform,
                    args=targs,
                    kwargs=tkwargs,
                    use_argnum=self._use_argnum_in_expand,
                )
            )
        qnode.add_transform(
            TransformContainer(
                self._transform,
                args=targs,
                kwargs=tkwargs,
                classical_cotransform=self._classical_cotransform,
                plxpr_transform=self._plxpr_transform,
                is_informative=self._is_informative,
                final_transform=self._final_transform,
            )
        )
        return qnode


class TransformContainer:  # pylint: disable=too-many-instance-attributes
    """Class to store a quantum transform with its ``args``, ``kwargs`` and classical co-transforms.  Use
    :func:`~.pennylane.transform`.

    .. warning::

        This class is developer-facing and should not be used directly. Instead, use
        :func:`qml.transform <pennylane.transform>` if you would like to make a custom
        transform.

    .. seealso:: :func:`~.pennylane.transform`
    """

    def __init__(
        self,
        transform,
        args=None,
        kwargs=None,
        classical_cotransform=None,
        plxpr_transform=None,
        is_informative=False,
        final_transform=False,
        use_argnum=False,
    ):  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self._transform = transform
        self._args = args or []
        self._kwargs = kwargs or {}
        self._classical_cotransform = classical_cotransform
        self._plxpr_transform = plxpr_transform
        self._is_informative = is_informative
        self._final_transform = is_informative or final_transform
        self._use_argnum = use_argnum

    def __repr__(self):
        return f"<{self._transform.__name__}({self._args}, {self._kwargs})>"

    def __iter__(self):
        return iter(
            (
                self._transform,
                self._args,
                self._kwargs,
                self._classical_cotransform,
                self._plxpr_transform,
                self._is_informative,
                self.final_transform,
            )
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TransformContainer):
            return False
        return (
            self.args == other.args
            and self.transform == other.transform
            and self.kwargs == other.kwargs
            and self.classical_cotransform == other.classical_cotransform
            and self.is_informative == other.is_informative
            and self.final_transform == other.final_transform
        )

    @property
    def transform(self):
        """The stored quantum transform."""
        return self._transform

    @property
    def args(self):
        """The stored quantum transform's ``args``."""
        return self._args

    @property
    def kwargs(self):
        """The stored quantum transform's ``kwargs``."""
        return self._kwargs

    @property
    def classical_cotransform(self):
        """The stored quantum transform's classical co-transform."""
        return self._classical_cotransform

    @property
    def plxpr_transform(self):
        """The stored quantum transform's PLxPR transform."""
        return self._plxpr_transform

    @property
    def is_informative(self):
        """``True`` if the transform is informative."""
        return self._is_informative

    @property
    def final_transform(self):
        """``True`` if the transform needs to be executed"""
        return self._final_transform


@TransformDispatcher.generic_register
def _apply_to_tape(self, obj: qml.tape.QuantumScript, *targs, **tkwargs):
    if self.expand_transform:
        expanded_tapes, expand_processing = self.expand_transform(obj, *targs, **tkwargs)
        transformed_tapes = []
        processing_and_slices = []
        start = 0
        for tape in expanded_tapes:
            intermediate_tapes, post_processing_fn = self.transform(tape, *targs, **tkwargs)
            transformed_tapes.extend(intermediate_tapes)
            end = start + len(intermediate_tapes)
            processing_and_slices.append(tuple([post_processing_fn, slice(start, end)]))
            start = end

        def processing_fn(results):
            processed_results = [fn(results[slice]) for fn, slice in processing_and_slices]
            return expand_processing(processed_results)

    else:
        transformed_tapes, processing_fn = self.transform(obj, *targs, **tkwargs)

    if self.is_informative:
        return processing_fn(transformed_tapes)

    return transformed_tapes, processing_fn


def _capture_apply(obj, transform, *targs, **tkwargs):
    @functools.wraps(obj)
    def qfunc_transformed(*args, **kwargs):
        import jax  # pylint: disable=import-outside-toplevel

        flat_qfunc = capture.flatfn.FlatFn(obj)
        jaxpr = jax.make_jaxpr(functools.partial(flat_qfunc, **kwargs))(*args)
        flat_args = jax.tree_util.tree_leaves(args)

        n_args = len(flat_args)
        n_consts = len(jaxpr.consts)
        args_slice = slice(0, n_args)
        consts_slice = slice(n_args, n_args + n_consts)
        targs_slice = slice(n_args + n_consts, None)

        results = transform._primitive.bind(  # pylint: disable=protected-access
            *flat_args,
            *jaxpr.consts,
            *targs,
            inner_jaxpr=jaxpr.jaxpr,
            args_slice=args_slice,
            consts_slice=consts_slice,
            targs_slice=targs_slice,
            tkwargs=tkwargs,
        )

        assert flat_qfunc.out_tree is not None
        return jax.tree_util.tree_unflatten(flat_qfunc.out_tree, results)

    return qfunc_transformed


@TransformDispatcher.generic_register
def _apply_to_callable(self, obj: Callable, *targs, **tkwargs):
    if capture.enabled():
        return _capture_apply(obj, self, *targs, **tkwargs)

    @functools.wraps(obj)
    def qfunc_transformed(*args, **kwargs):
        with AnnotatedQueue() as q:
            qfunc_output = obj(*args, **kwargs)

        tape = qml.tape.QuantumScript.from_queue(q)
        with QueuingManager.stop_recording():
            transformed_tapes, processing_fn = self.transform(tape, *targs, **tkwargs)

        if len(transformed_tapes) != 1:
            raise TransformError(
                "Impossible to dispatch your transform on quantum function, because more than "
                "one tape is returned"
            )

        transformed_tape = transformed_tapes[0]

        if self.is_informative:
            return processing_fn(transformed_tapes)

        for op in transformed_tape.circuit:
            apply(op)

        mps = transformed_tape.measurements

        if not mps:
            return qfunc_output

        if isinstance(qfunc_output, qml.measurements.MeasurementProcess):
            return tuple(mps) if len(mps) > 1 else mps[0]

        if isinstance(qfunc_output, (tuple, list)):
            return type(qfunc_output)(mps)

        interface = math.get_interface(qfunc_output)
        return math.asarray(mps, like=interface)

    return qfunc_transformed


@TransformDispatcher.generic_register
def _apply_to_sequence(self, obj: Sequence, *targs, **tkwargs):
    if not all(isinstance(t, QuantumScript) for t in obj):
        raise ValueError("Can only apply transform to sequences of QuantumScript.")
    execution_tapes = []
    batch_fns = []
    tape_counts = []

    for t in obj:
        # Preprocess the tapes by applying transforms
        # to each tape, and storing corresponding tapes
        # for execution, processing functions, and list of tape lengths.
        new_tapes, fn = _apply_to_tape(self, t, *targs, **tkwargs)
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
