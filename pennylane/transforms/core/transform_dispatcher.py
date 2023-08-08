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
This module contains the transform function, the transform dispatcher and the transform container.
"""
import copy
import pennylane as qml


class TransformError(Exception):
    """Raised when there is an error with the transform logic"""


class TransformDispatcher:
    r"""This object is developer facing and should not be used directly to create transforms. Use
    :func:`~.pennylane.transforms.core.transform`.

    Convert a transform that has the signature (tape -> Sequence(tape), fn) to a transform dispatcher that can act
    on tape, qfunc and qnode.

    .. warning::

        This class is developer-facing and should not be used directly.

    .. seealso:: :func:`~.pennylane.transforms.core.transform`
    """

    def __init__(
        self,
        transform,
        expand_transform=None,
        classical_cotransform=None,
        is_informative=False,
    ):  # pylint:disable=redefined-outer-name
        self._transform = transform
        self._expand_transform = expand_transform
        self._classical_cotransform = classical_cotransform
        self._is_informative = is_informative

    def __call__(self, *targs, **tkwargs):
        obj = None

        if targs:
            # assume the first argument passed to the transform
            # is the object we wish to transform
            obj, *targs = targs

        if isinstance(obj, (qml.tape.QuantumTape, qml.tape.QuantumScript)):
            new_tape = copy.deepcopy(obj)
            return self._transform(new_tape, *targs, **tkwargs)
        if isinstance(obj, qml.QNode):
            return self._qnode_transform(
                obj,
                targs,
                tkwargs,
            )
        if callable(obj):
            return self._qfunc_transform(obj, targs, tkwargs)

        # Input is not a QNode nor a quantum tape nor a device.
        # Assume Python decorator syntax:
        #
        # result = some_transform(*transform_args)(qnode)(*qnode_args)

        if obj is not None:
            targs = (obj, *targs)

        def wrapper(obj):
            return self(obj, *targs, **tkwargs)

        return wrapper

    @property
    def transform(self):
        """Return the quantum transform."""
        return self._transform

    @property
    def expand_transform(self):
        """Return the expand transform."""
        return self._expand_transform

    @property
    def classical_cotransform(self):
        """Return the classical co-transform."""
        return self._classical_cotransform

    @property
    def is_informative(self):
        """Return True is the transform does not need to be executed."""
        return self._is_informative

    def _qfunc_transform(self, qfunc, targs, tkwargs):
        """Apply the transform on a quantum function."""

        def qfunc_transformed(*args, **kwargs):
            tape = qml.tape.make_qscript(qfunc)(*args, **kwargs)
            transformed_tapes, _ = self._transform(tape, *targs, **tkwargs)

            if len(transformed_tapes) != 1:
                raise TransformError(
                    "Impossible to dispatch your transform on quantum function, because more than "
                    "one tape is returned"
                )

            transformed_tape = transformed_tapes[0]

            for op in transformed_tape.circuit:
                qml.apply(op)

            return tape._qfunc_output  # pylint:disable=protected-access

        return qfunc_transformed

    def _qnode_transform(self, qnode, targs, tkwargs):
        """Apply the transform on a QNode. It populates the transform program of a QNode"""
        if qnode.transform_program.is_empty():
            qnode = copy.deepcopy(qnode)

        if self.expand_transform:
            qnode.add_transform(TransformContainer(self._expand_transform))
        qnode.add_transform(
            TransformContainer(
                self._transform, targs, tkwargs, self._classical_cotransform, self._is_informative
            )
        )
        return qnode


class TransformContainer:
    """Class to store a quantum transform with its args, kwargs and classical co-transforms.  Use
    :func:`~.pennylane.transforms.core.transform`.

    .. warning::

        This class is developer-facing and should not be used directly.

    .. seealso:: :func:`~.pennylane.transforms.core.transform`
    """

    def __init__(
        self, transform, args=None, kwargs=None, classical_cotransform=None, is_informative=False
    ):  # pylint:disable=redefined-outer-name,too-many-arguments
        self._transform = transform
        self._args = args or []
        self._kwargs = kwargs or {}
        self._classical_cotransform = classical_cotransform
        self._is_informative = is_informative

    def __iter__(self):
        return iter(
            (
                self._transform,
                self._args,
                self._kwargs,
                self._classical_cotransform,
                self._is_informative,
            )
        )

    @property
    def transform(self):
        """Return the stored quantum transform."""
        return self._transform

    @property
    def args(self):
        """Return the stored quantum transform's args."""
        return self._args

    @property
    def kwargs(self):
        """Return the stored quantum transform's arkwgs."""
        return self._kwargs

    @property
    def classical_cotransform(self):
        """Return the stored quantum transform's classical co-transform."""
        return self._classical_cotransform

    @property
    def is_informative(self):
        """Return True is the transform does not need to be executed."""
        return self._is_informative
