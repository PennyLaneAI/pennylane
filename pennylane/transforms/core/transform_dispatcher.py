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
import types
import warnings
from collections.abc import Callable, Sequence
from copy import copy

import pennylane as qml
from pennylane import capture, math
from pennylane.capture.autograph import wraps
from pennylane.exceptions import TransformError
from pennylane.operation import Operator
from pennylane.queuing import AnnotatedQueue, QueuingManager, apply
from pennylane.typing import ResultBatch


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
    ):  # pylint: disable=too-many-arguments
        args = invals[args_slice]
        consts = invals[consts_slice]
        targs = invals[targs_slice]

        def wrapper(*inner_args):
            return copy(self).eval(inner_jaxpr, consts, *inner_args)

        jaxpr = jax.make_jaxpr(wrapper)(*args)
        jaxpr = plxpr_transform(jaxpr.jaxpr, jaxpr.consts, targs, tkwargs, *args)
        return copy(self).eval(jaxpr.jaxpr, jaxpr.consts, *args)


def _preprocess_device(original_device, transform, targs, tkwargs):
    class TransformedDevice(type(original_device)):
        """A transformed device with updated preprocess method."""

        def __init__(self, original_device, transform, targs, tkwargs):
            for key, value in original_device.__dict__.items():
                self.__setattr__(key, value)
            self.transform = transform
            self.targs = targs
            self.tkwargs = tkwargs
            self._original_device = original_device

        def __repr__(self):
            return f"Transformed Device({repr(original_device)} with additional preprocess transform {self.transform.transform})"

        def preprocess(
            self,
            execution_config: qml.devices.ExecutionConfig | None = None,
        ):
            """This function updates the original device transform program to be applied."""
            program, config = self.original_device.preprocess(execution_config)
            program.push_back(
                TransformContainer(self.transform, args=self.targs, kwargs=self.tkwargs)
            )
            return program, config

        @property
        def original_device(self):
            """Return the original device."""
            return self._original_device

    return TransformedDevice(original_device, transform, targs, tkwargs)


def _preprocess_transforms_device(original_device, transform, targs, tkwargs):
    class TransformedDevice(type(original_device)):
        """A transformed device with updated preprocess method."""

        def __init__(self, original_device, transform, targs, tkwargs):
            for key, value in original_device.__dict__.items():
                self.__setattr__(key, value)
            self.transform = transform
            self.targs = targs
            self.tkwargs = tkwargs
            self._original_device = original_device

        def __repr__(self):
            return f"Transformed Device({repr(original_device)} with additional preprocess transform {self.transform.transform})"

        def preprocess_transforms(
            self,
            execution_config: qml.devices.ExecutionConfig | None = None,
        ):
            """This function updates the original device transform program to be applied."""
            program = self.original_device.preprocess_transforms(execution_config)
            program.push_back(
                TransformContainer(self.transform, args=self.targs, kwargs=self.tkwargs)
            )
            return program

        @property
        def original_device(self):
            """Return the original device."""
            return self._original_device

    return TransformedDevice(original_device, transform, targs, tkwargs)


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
        self._qnode_transform = self.default_qnode_transform

        self._use_argnum_in_expand = use_argnum_in_expand
        functools.update_wrapper(self, transform)

        self._plxpr_transform = plxpr_transform or _create_plxpr_fallback_transform(self._transform)
        self._primitive = _create_transform_primitive(self._transform.__name__)
        _register_primitive_for_expansion(self._primitive, self._plxpr_transform)

    def __call__(
        self, *targs, **tkwargs
    ):  # pylint: disable=too-many-return-statements,too-many-branches
        obj = None

        if targs:
            # assume the first argument passed to the transform
            # is the object we wish to transform
            obj, *targs = targs

        if isinstance(obj, qml.tape.QuantumScript):
            if self._expand_transform:
                expanded_tapes, expand_processing = self._expand_transform(obj, *targs, **tkwargs)
                transformed_tapes = []
                processing_and_slices = []
                start = 0
                for tape in expanded_tapes:
                    intermediate_tapes, post_processing_fn = self._transform(
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
                transformed_tapes, processing_fn = self._transform(obj, *targs, **tkwargs)

            if self.is_informative:
                return processing_fn(transformed_tapes)
            return transformed_tapes, processing_fn

        if isinstance(obj, qml.workflow.QNode):
            return self._qnode_transform(obj, targs, tkwargs)

        if isinstance(obj, qml.devices.Device):
            return self._device_transform(obj, targs, tkwargs)

        if obj.__class__.__name__ == "QJIT":
            raise TransformError(
                "Functions that are wrapped / decorated with qjit cannot subsequently be"
                f" transformed with a PennyLane transform (attempted {self})."
                f" For the desired affect, ensure that qjit is applied after {self}."
            )

        if callable(obj):
            return self._qfunc_transform(obj, targs, tkwargs)

        if isinstance(obj, Sequence) and all(isinstance(q, qml.tape.QuantumScript) for q in obj):
            return self._batch_transform(obj, targs, tkwargs)

        # Input is not a QNode nor a quantum tape nor a device.
        # Assume Python decorator syntax:
        #
        # result = some_transform(*transform_args)(qnode)(*qnode_args)

        raise TransformError(
            "Decorating a QNode with @transform_fn(**transform_kwargs) has been "
            "removed. Please decorate with @functools.partial(transform_fn, **transform_kwargs) "
            "instead, or call the transform directly using qnode = transform_fn(qnode, "
            "**transform_kwargs). Visit the deprecations page for more details: "
            "https://docs.pennylane.ai/en/stable/development/deprecations.html#completed-deprecation-cycles",
        )

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
        self._qnode_transform = types.MethodType(fn, self)

    def default_qnode_transform(self, qnode, targs, tkwargs):
        """
        The default method that takes in a QNode and returns another QNode
        with the transform applied.
        """

        qnode = copy(qnode)

        if self.expand_transform:
            qnode.transform_program.push_back(
                TransformContainer(
                    TransformDispatcher(self._expand_transform),
                    args=targs,
                    kwargs=tkwargs,
                    use_argnum=self._use_argnum_in_expand,
                )
            )
        qnode.transform_program.push_back(
            TransformContainer(
                self,
                args=targs,
                kwargs=tkwargs,
            )
        )
        return qnode

    def _capture_callable_transform(self, qfunc, targs, tkwargs):
        """Apply the transform on a quantum function when program capture is enabled"""

        @wraps(qfunc)
        def qfunc_transformed(*args, **kwargs):
            import jax  # pylint: disable=import-outside-toplevel

            flat_qfunc = capture.flatfn.FlatFn(qfunc)
            jaxpr = jax.make_jaxpr(functools.partial(flat_qfunc, **kwargs))(*args)
            flat_args = jax.tree_util.tree_leaves(args)

            n_args = len(flat_args)
            n_consts = len(jaxpr.consts)
            args_slice = slice(0, n_args)
            consts_slice = slice(n_args, n_args + n_consts)
            targs_slice = slice(n_args + n_consts, None)

            results = self._primitive.bind(
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

    def _qfunc_transform(self, qfunc, targs, tkwargs):
        """Apply the transform on a quantum function."""

        @wraps(qfunc)
        def qfunc_transformed(*args, **kwargs):

            if qml.capture.enabled():
                return self._capture_callable_transform(qfunc, targs, tkwargs)(*args, **kwargs)
            # removes the argument to the qfuncs from the active queuing context.
            leaves, _ = qml.pytrees.flatten((args, kwargs), lambda obj: isinstance(obj, Operator))
            for l in leaves:
                if isinstance(l, Operator):
                    qml.QueuingManager.remove(l)

            with AnnotatedQueue() as q:
                qfunc_output = qfunc(*args, **kwargs)

            tape = qml.tape.QuantumScript.from_queue(q)
            with QueuingManager.stop_recording():
                transformed_tapes, processing_fn = self._transform(tape, *targs, **tkwargs)

            if len(transformed_tapes) != 1:
                raise TransformError(
                    "Impossible to dispatch your transform on quantum function, because more than "
                    "one tape is returned"
                )

            transformed_tape = transformed_tapes[0]

            if self.is_informative:
                return processing_fn(transformed_tapes)

            for op in transformed_tape.operations:
                apply(op)

            mps = [qml.apply(mp) for mp in transformed_tape.measurements]

            if not mps:
                return qfunc_output

            if isinstance(qfunc_output, qml.measurements.MeasurementProcess):
                return tuple(mps) if len(mps) > 1 else mps[0]

            if isinstance(qfunc_output, (tuple, list)):
                return type(qfunc_output)(mps)

            interface = math.get_interface(qfunc_output)
            return math.asarray(mps, like=interface)

        return qfunc_transformed

    def _device_transform(self, original_device, targs, tkwargs):
        """Apply the transform on a device"""
        if self._expand_transform:
            raise TransformError("Device transform does not support expand transforms.")
        if self._is_informative:
            raise TransformError("Device transform does not support informative transforms.")
        if self._final_transform:
            raise TransformError("Device transform does not support final transforms.")

        if type(original_device).preprocess != qml.devices.Device.preprocess:
            return _preprocess_device(original_device, self, targs, tkwargs)

        return _preprocess_transforms_device(original_device, self, targs, tkwargs)

    def _batch_transform(self, original_batch, targs, tkwargs):
        """Apply the transform on a batch of tapes."""
        execution_tapes = []
        batch_fns = []
        tape_counts = []

        for t in original_batch:
            # Preprocess the tapes by applying transforms
            # to each tape, and storing corresponding tapes
            # for execution, processing functions, and list of tape lengths.
            new_tapes, fn = self(t, *targs, **tkwargs)
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

            for f, s in zip(batch_fns, tape_counts, strict=True):
                # apply any batch transform post-processing
                new_res = f(res[count : count + s])
                final_results.append(new_res)
                count += s

            return tuple(final_results)

        return tuple(execution_tapes), processing_fn


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
        transform: TransformDispatcher,
        args: tuple | list = (),
        kwargs: None | dict = None,
        use_argnum: bool = False,
        **transform_config,
    ):
        if not isinstance(transform, TransformDispatcher):
            transform = TransformDispatcher(transform, **transform_config)
        elif transform_config:
            raise ValueError(
                f"transform_config kwargs {transform_config} cannot be passed if a TransformDispatcher is provided."
            )
        self._transform_dispatcher = transform
        self._args = tuple(args)
        self._kwargs = kwargs or {}
        self._use_argnum = use_argnum

    def __repr__(self):
        return f"<{self._transform_dispatcher.transform.__name__}({self._args}, {self._kwargs})>"

    def __call__(self, obj):
        return self._transform_dispatcher(obj, *self.args, **self.kwargs)

    def __iter__(self):
        return iter(
            (
                self._transform_dispatcher.transform,
                self._args,
                self._kwargs,
                self._transform_dispatcher._classical_cotransform,
                self._transform_dispatcher._plxpr_transform,
                self._transform_dispatcher._is_informative,
                self._transform_dispatcher.final_transform,
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
    def transform(self) -> Callable:
        """The stored quantum transform."""
        return self._transform_dispatcher.transform

    @property
    def args(self) -> tuple:
        """The stored quantum transform's ``args``."""
        return self._args

    @property
    def kwargs(self) -> dict:
        """The stored quantum transform's ``kwargs``."""
        return self._kwargs

    @property
    def classical_cotransform(self) -> None | Callable:
        """The stored quantum transform's classical co-transform."""
        return self._transform_dispatcher.classical_cotransform

    @property
    def plxpr_transform(self) -> None | Callable:
        """The stored quantum transform's PLxPR transform."""
        return self._transform_dispatcher.plxpr_transform

    @property
    def is_informative(self) -> bool:
        """``True`` if the transform is informative."""
        return self._transform_dispatcher.is_informative

    @property
    def final_transform(self) -> bool:
        """``True`` if the transform needs to be executed"""
        return self._transform_dispatcher.final_transform


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
