# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains tape expansion functions and stopping criteria to
generate such functions from."""
# pylint: disable=unused-argument
import contextlib

import pennylane as qml
from pennylane import math
from pennylane.measurements import MeasurementProcess


def _update_trainable_params(tape):
    params = tape.get_parameters(trainable_only=False)
    tape.trainable_params = math.get_trainable_indices(params)


def create_expand_fn(depth, stop_at=None, device=None, docstring=None):
    """
    .. warning::
        Please use the :func:`qml.transforms.decompose <.transforms.decompose>` function for decomposing circuits.

    Create a function for expanding a tape to a given depth, and
    with a specific stopping criterion. This is a wrapper around
    :meth:`~.QuantumTape.expand`.

    Args:
        depth (int): Depth for the expansion
        stop_at (callable): Stopping criterion. This must be a function with signature
            ``stop_at(obj)``, where ``obj`` is a *queueable* PennyLane object such as
            :class:`~.Operation` or :class:`~.MeasurementProcess`. It must return a
            boolean, indicating if the expansion should stop at this object.
        device (pennylane.devices.LegacyDevice): Ensure that the expanded tape only uses native gates of the
            given device.
        docstring (str): docstring for the generated expansion function

    Returns:
        callable: Tape expansion function. The returned function accepts a :class:`~.QuantumTape`,
        and returns an expanded :class:`~.QuantumTape`.

    **Example**

    Let us construct an expansion function that expands a tape in order to
    decompose trainable multi-parameter gates. We allow for up to five expansion
    steps, which can be controlled with the argument ``depth``.
    The stopping criterion is easy to write as

    >>> def stop_at(obj):
    ...     return not (len(obj.data) > 1 and any(qml.math.requires_grad(d) for d in obj.data))

    Then the expansion function can be obtained via

    >>> expand_fn = qml.transforms.create_expand_fn(depth=5, stop_at=stop_at)

    We can test the newly generated function on an example tape:

    .. code-block:: python

        ops = [
            qml.RX(0.2, wires=0),
            qml.RX(qml.numpy.array(-2.4, requires_grad=True), wires=1),
            qml.Rot(1.7, 0.92, -1.1, wires=0),
            qml.Rot(*qml.numpy.array([-3.1, 0.73, 1.36], requires_grad=True), wires=1)
        ]
        tape = qml.tape.QuantumTape(ops)

    >>> new_tape = expand_fn(tape)
    >>> print(qml.drawer.tape_text(tape, decimals=1))
    0: ──RX(0.2)───Rot(1.7,0.9,-1.1)─┤
    1: ──RX(-2.4)──Rot(-3.1,0.7,1.4)─┤
    >>> print(qml.drawer.tape_text(new_tape, decimals=1))
    0: ──RX(0.2)───Rot(1.7,0.9,-1.1)───────────────────┤
    1: ──RX(-2.4)──RZ(-3.1)───────────RY(0.7)──RZ(1.4)─┤

    """
    # pylint: disable=unused-argument
    if device is not None:
        if stop_at is None:
            stop_at = device.stopping_condition
        else:
            orig_stop_at = stop_at

            def stop_at(obj):
                return orig_stop_at(obj) and device.stopping_condition(obj)

    def expand_fn(tape, depth=depth, **kwargs):
        with qml.QueuingManager.stop_recording():
            if not all(stop_at(op) for op in tape.operations):
                (tape,), _ = qml.transforms.decompose(
                    tape, max_expansion=depth, stopping_condition=stop_at
                )
            else:
                return tape

            _update_trainable_params(tape)

        return tape

    if docstring:
        expand_fn.__doc__ = docstring

    return expand_fn


_expand_multipar_doc = """Expand out a tape so that all its parametrized
operations have a single parameter.

This is achieved by decomposing all parametrized operations that do not have
a generator, up to maximum depth ``depth``.
For a sufficient ``depth``, it should always be possible to obtain a tape containing
only single-parameter operations.

Args:
    tape (.QuantumTape): the input tape to expand
    depth (int) : the maximum expansion depth
    **kwargs: additional keyword arguments are ignored

Returns:
    .QuantumTape: the expanded tape
"""


def _multipar_stopping_fn(obj):
    try:
        return (
            isinstance(obj, MeasurementProcess)
            or len(obj.data) == 0
            or (obj.has_generator and len(obj.generator().terms()[0]) == 1)
        )
    except qml.operation.TermsUndefinedError:
        return True


expand_multipar = create_expand_fn(
    depth=None,
    stop_at=_multipar_stopping_fn,
    docstring=_expand_multipar_doc,
)

_expand_trainable_multipar_doc = """Expand out a tape so that all its trainable
operations have a single parameter.

This is achieved by decomposing all trainable operations that do not have
a generator, up to maximum depth ``depth``.
For a sufficient ``depth``, it should always be possible to obtain a tape containing
only single-parameter operations.

Args:
    tape (.QuantumTape): the input tape to expand
    depth (int) : the maximum expansion depth
    **kwargs: additional keyword arguments are ignored

Returns:
    .QuantumTape: the expanded tape
"""


def _trainable_multipar_stopping_fn(obj):
    return _multipar_stopping_fn(obj) or not any(math.requires_grad(d) for d in obj.data)


expand_trainable_multipar = create_expand_fn(
    depth=None,
    stop_at=_trainable_multipar_stopping_fn,
    docstring=_expand_trainable_multipar_doc,
)


def create_expand_trainable_multipar(tape, use_tape_argnum=False):
    """Creates the expand_trainable_multipar expansion transform with an option to include argnums."""

    if not use_tape_argnum:
        return expand_trainable_multipar

    trainable_par_info = [tape.par_info[i] for i in tape.trainable_params]
    trainable_ops = [info["op"] for info in trainable_par_info]

    def _argnum_trainable_multipar(obj):
        return _multipar_stopping_fn(obj) or obj not in trainable_ops

    return create_expand_fn(
        depth=None,
        stop_at=_argnum_trainable_multipar,
        docstring=_expand_trainable_multipar_doc,
    )


_expand_nonunitary_gen_doc = """Expand out a tape so that all its parametrized
operations have a unitary generator.

This is achieved by decomposing all parametrized operations that either do not have
a generator or have a non-unitary generator, up to maximum depth ``depth``.
For a sufficient ``depth``, it should always be possible to obtain a tape containing
only unitarily generated operations.

Args:
    tape (.QuantumTape): the input tape to expand
    depth (int) : the maximum expansion depth
    **kwargs: additional keyword arguments are ignored

Returns:
    .QuantumTape: the expanded tape
"""


def _expand_nonunitary_gen_stop_at(obj):
    return (
        isinstance(obj, MeasurementProcess)
        or len(obj.data) == 0
        or (obj.has_generator and obj in qml.ops.qubit.attributes.has_unitary_generator)
    )


expand_nonunitary_gen = create_expand_fn(
    depth=None,
    stop_at=_expand_nonunitary_gen_stop_at,
    docstring=_expand_nonunitary_gen_doc,
)

_expand_invalid_trainable_doc = """Expand out a tape so that it supports differentiation
of requested operations.

This is achieved by decomposing all trainable operations that have
``Operation.grad_method=None`` until all resulting operations
have a defined gradient method, up to maximum depth ``depth``. Note that this
might not be possible, in which case the gradient rule will fail to apply.

Args:
    tape (.QuantumTape): the input tape to expand
    depth (int) : the maximum expansion depth
    **kwargs: additional keyword arguments are ignored

Returns:
    .QuantumTape: the expanded tape
"""


def _stop_at_expand_invalid_trainable(obj):
    return (
        isinstance(obj, MeasurementProcess)
        or not any(math.requires_grad(d) for d in obj.data)
        or obj.grad_method is not None
    )


expand_invalid_trainable = create_expand_fn(
    depth=None,
    stop_at=_stop_at_expand_invalid_trainable,
    docstring=_expand_invalid_trainable_doc,
)
