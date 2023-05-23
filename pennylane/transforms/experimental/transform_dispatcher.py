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
This module contains the transform dispatcher.
"""
import functools
from typing import get_type_hints
from collections.abc import Sequence
import pennylane as qml


class TransformError(Exception):
    """Raised when there is an error with the transform logic"""


class TransformDispatcher:
    r"""This object is developer facing and should not be used directly.

    Convert a transform that has the signature (tape -> Sequence(tape), fn) to a transform dispatcher that can act
    on tape, qfunc and qnode.
    """

    def __init__(self, transform, expand_transform=None, qnode_postprocessing=None):
        if not callable(transform):
            raise TransformError(
                f"The function to register, {transform}, "
                "does not appear to be a valid Python function or callable."
            )

        self._transform = transform
        functools.update_wrapper(self, transform)
        self._expand_transform = expand_transform
        self._qnode_postprocessing = qnode_postprocessing

    def __call__(self, *targs, **tkwargs):
        obj = None

        if targs:
            # assume the first argument passed to the transform
            # is the object we wish to transform
            obj, *targs = targs

        if isinstance(obj, qml.tape.QuantumTape):
            return self._transform(obj, *targs, **tkwargs)
        elif isinstance(obj, qml.QNode):
            return self._qnode_transform(
                obj,
                targs,
                tkwargs,
                expand_transform=self._expand_transform,
                qnode_postprocessing=self._qnode_postprocessing,
            )
        elif callable(obj):
            return self.default_qfunc_transform(obj, targs, tkwargs)
        else:
            raise TransformError(
                "The object on which is the transform is applied is not valid. It can only be a "
                "tape, a QNode or a qfunc."
            )

    @property
    def transform(self):
        return self._transform

    @property
    def expand_transform(self):
        return self._expand_transform

    @property
    def qnode_postprocessing(self):
        return self._qnode_postprocessing

    def _qfunc_transform(self, qfunc, targs, tkwargs):
        """Apply the transform on a quantum function."""

        def qfunc_transformed(*args, **kwargs):
            tape = qml.tape.make_qscript(qfunc)(*args, **kwargs)
            transformed_tapes, _ = self._transform(tape, *targs, **tkwargs)

            for tape in transformed_tapes:
                for op in tape.circuit:
                    qml.apply(op)

        return qfunc_transformed

    def _qnode_transform(self, qnode, targs, tkwargs):
        """Apply the transform on a QNode. It populates the transform program of a QNode"""
        if self.expand_transform:
            qnode.add_transform(TransformContainer(self._expand_transform))
        qnode.add_transform(
            TransformContainer(self._transform, targs, tkwargs, self._qnode_postprocessing)
        )
        return qnode


def transform(transform, expand_transform=None, qnode_postprocessing=None):
    """
    Signature and type validation for the transform and its expand function and qnode post processing.

    TODO: Make it clear what is a transform

    TODO: add example of what is expected from a transform.
    """
    # Check signature of transform to force the fn style (tape, ...) - > (Sequence(tape), fn)
    signature_transform = get_type_hints(transform)
    _transform_signature_check(signature_transform)

    # Check the signature of expand_transform to force the fn style tape - > (Sequence(tape), fn)
    if expand_transform is not None:
        signature_expand_transform = get_type_hints(expand_transform)
        _transform_signature_check(signature_expand_transform)

        if len(signature_expand_transform) > 2:
            raise TransformError(
                "The expand transform does not support arg and kwargs other than tape."
            )

    # Check that the qnode post processing is callable
    if qnode_postprocessing is not None:
        if not callable(qnode_postprocessing):
            raise TransformError("The qnode post processing must be a valid Python function.")

    return TransformDispatcher(
        transform, expand_transform=expand_transform, qnode_postprocessing=qnode_postprocessing
    )


def _transform_signature_check(signature):
    """Check the signature of a transform (tape, ...) - > (Sequence(tape), fn)"""
    # Check that the arguments of the transforms follows: (tape: qml.tape.QuantumTape, ...)
    tape = signature.get("tape", None)

    if tape is None:
        raise TransformError("The first argument of a transform must be tape.")

    if tape != qml.tape.QuantumTape:
        raise TransformError("The type of the tape argument must be a QuantumTape")

    # Check return is (qml.tape.QuantumTape, callable):
    ret = signature.get("return", None)

    if ret is None or not isinstance(ret, tuple):
        raise TransformError(
            "The return of a transform must match (collections.abc.Sequence["
            "pennylane.tape.tape.QuantumTape], <built-in function callable>)"
        )

    if not isinstance(ret[0], Sequence):
        raise TransformError(
            "The first return of a transform must be a sequence of tapes: collections.abc.Sequence["
            "pennylane.tape.tape.QuantumTape]"
        )

    if not callable(ret[1]):
        raise TransformError(
            "The second return of a transform must be a callable: <built-in function callable>"
        )


class TransformContainer:
    def __init__(self, transform, args=None, kwargs=None, qnode_processing=None):
        self.transform = transform
        self.targs = args if args else []
        self.tkwargs = kwargs if kwargs else {}
        self.qnode_processing = qnode_processing
