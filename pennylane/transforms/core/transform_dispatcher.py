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
import copy
import warnings
import types

from typing import Sequence

import pennylane as qml
from pennylane.typing import ResultBatch


class TransformError(Exception):
    """Raised when there is an error with the transform logic."""


class TransformDispatcher:
    r"""Converts a transform that has the signature ``(tape -> Sequence(tape), fn)`` to a transform dispatcher
    that can act on :class:`pennylane.tape.QuantumTape`, quantum function, :class:`pennylane.QNode`,
    :class:`pennylane.devices.Device`.

    .. warning::

        This class is developer-facing and should not be used directly. Instead, use
        :func:`qml.transform <pennylane.transform>` if you would like to make a custom
        transform.

    .. seealso:: :func:`~.pennylane.transform`
    """

    def __new__(cls, *args, **kwargs):  # pylint: disable=unused-argument
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

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        transform,
        expand_transform=None,
        classical_cotransform=None,
        is_informative=False,
        final_transform=False,
        use_argnum_in_expand=False,
    ):  # pylint:disable=redefined-outer-name
        self._transform = transform
        self._expand_transform = expand_transform
        self._classical_cotransform = classical_cotransform
        self._is_informative = is_informative
        # is_informative supersedes final_transform
        self._final_transform = is_informative or final_transform
        self._qnode_transform = self.default_qnode_transform
        self._use_argnum_in_expand = use_argnum_in_expand
        functools.update_wrapper(self, transform)

    def __call__(self, *targs, **tkwargs):  # pylint: disable=too-many-return-statements
        obj = None

        if targs:
            # assume the first argument passed to the transform
            # is the object we wish to transform
            obj, *targs = targs

        if isinstance(obj, qml.tape.QuantumScript):
            if self._expand_transform:
                expanded_tapes, expand_processing = self._expand_transform(obj, *targs, **tkwargs)
                transformed_tapes = []
                processing_and_sclices = []
                start = 0
                for tape in expanded_tapes:
                    intermediate_tapes, post_processing_fn = self._transform(
                        tape, *targs, **tkwargs
                    )
                    transformed_tapes.extend(intermediate_tapes)
                    end = start + len(intermediate_tapes)
                    processing_and_sclices.append(tuple([post_processing_fn, slice(start, end)]))
                    start = end

                def processing_fn(results):
                    processed_results = [fn(results[slice]) for fn, slice in processing_and_sclices]
                    return expand_processing(processed_results)

            else:
                transformed_tapes, processing_fn = self._transform(obj, *targs, **tkwargs)

            if self.is_informative:
                return processing_fn(transformed_tapes)
            return transformed_tapes, processing_fn

        if isinstance(obj, qml.QNode):
            return self._qnode_transform(obj, targs, tkwargs)
        # TODO: Remove with the previous device generation
        if isinstance(obj, qml.Device):
            return self._old_device_transform(obj, targs, tkwargs)
        if isinstance(obj, qml.devices.Device):
            return self._device_transform(obj, targs, tkwargs)
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

        qnode = copy.copy(qnode)

        if self.expand_transform:
            qnode.add_transform(
                TransformContainer(
                    self._expand_transform, targs, tkwargs, use_argnum=self._use_argnum_in_expand
                )
            )
        qnode.add_transform(
            TransformContainer(
                self._transform,
                targs,
                tkwargs,
                self._classical_cotransform,
                self._is_informative,
                self._final_transform,
            )
        )
        return qnode

    def _qfunc_transform(self, qfunc, targs, tkwargs):
        """Apply the transform on a quantum function."""

        def qfunc_transformed(*args, **kwargs):
            with qml.queuing.AnnotatedQueue() as q:
                qfunc_output = qfunc(*args, **kwargs)

            tape = qml.tape.QuantumScript.from_queue(q)
            with qml.QueuingManager.stop_recording():
                transformed_tapes, processing_fn = self._transform(tape, *targs, **tkwargs)

            if len(transformed_tapes) != 1:
                raise TransformError(
                    "Impossible to dispatch your transform on quantum function, because more than "
                    "one tape is returned"
                )

            transformed_tape = transformed_tapes[0]

            if self.is_informative:
                return processing_fn(transformed_tapes)

            for op in transformed_tape.circuit:
                qml.apply(op)

            mps = transformed_tape.measurements

            if not mps:
                return qfunc_output

            if isinstance(qfunc_output, qml.measurements.MeasurementProcess):
                return tuple(mps) if len(mps) > 1 else mps[0]

            if isinstance(qfunc_output, (tuple, list)):
                return type(qfunc_output)(mps)

            interface = qml.math.get_interface(qfunc_output)
            return qml.math.asarray(mps, like=interface)

        return qfunc_transformed

    def _old_device_transform(self, original_device, targs, tkwargs):
        """Apply the transform on a device"""
        if self._expand_transform:
            raise TransformError("Device transform does not support expand transforms.")
        if self._is_informative:
            raise TransformError("Device transform does not support informative transforms.")
        if self._final_transform:
            raise TransformError("Device transform does not support final transforms.")
        new_dev = copy.deepcopy(original_device)
        transform = self._transform

        @new_dev.custom_expand
        def new_expand_fn(self, tape, *args, **kwargs):  # pylint: disable=unused-variable
            tapes, _ = transform(tape, *targs, **tkwargs)
            tape = tapes[0]
            return self.default_expand_fn(tape, *args, **kwargs)

        return new_dev

    def _device_transform(self, original_device, targs, tkwargs):
        """Apply the transform on a device"""
        if self._expand_transform:
            raise TransformError("Device transform does not support expand transforms.")
        if self._is_informative:
            raise TransformError("Device transform does not support informative transforms.")
        if self._final_transform:
            raise TransformError("Device transform does not support final transforms.")

        class TransformedDevice(type(original_device)):
            """A transformed device with updated preprocess method."""

            def __init__(self, original_device, transform):
                for key, value in original_device.__dict__.items():
                    self.__setattr__(key, value)
                self.transform = transform
                self._original_device = original_device

            def __repr__(self):
                return f"Transformed Device({original_device.__repr__()} with additional preprocess transform {self.transform})"

            def preprocess(
                self,
                execution_config: qml.devices.ExecutionConfig = qml.devices.DefaultExecutionConfig,
            ):
                """This function updates the original device transform program to be applied."""
                program, config = self.original_device.preprocess(execution_config)
                program.push_back(TransformContainer(self.transform, targs, tkwargs))
                return program, config

            @property
            def original_device(self):
                """Return the original device."""
                return self._original_device

        return TransformedDevice(original_device, self._transform)

    def _batch_transform(self, original_batch, targs, tkwargs):
        """Apply the transform on a batch of tapes"""
        execution_tapes = []
        batch_fns = []
        tape_counts = []

        for t in original_batch:
            # Preprocess the tapes by applying batch transforms
            # to each tape, and storing corresponding tapes
            # for execution, processing functions, and list of tape lengths.
            new_tapes, fn = self(t, *targs, **tkwargs)
            execution_tapes.extend(new_tapes)
            batch_fns.append(fn)
            tape_counts.append(len(new_tapes))

        def processing_fn(res: ResultBatch) -> ResultBatch:
            """Applies a batch of post-processing functions to results.

            Args:
                res (ResultBatch): the results of executing a batch of circuits

            Returns:
                ResultBatch : results that have undergone classical post processing

            Closure variables:
                tape_counts: the number of tapes outputted from each application of the transform
                batch_fns: the post processing functions to apply to each sub-batch

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


class TransformContainer:
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
        is_informative=False,
        final_transform=False,
        use_argnum=False,
    ):  # pylint:disable=redefined-outer-name,too-many-arguments
        self._transform = transform
        self._args = args or []
        self._kwargs = kwargs or {}
        self._classical_cotransform = classical_cotransform
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
    def is_informative(self):
        """``True`` if the transform is informative."""
        return self._is_informative

    @property
    def final_transform(self):
        """``True`` if the transform needs to be executed"""
        return self._final_transform
