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

import pennylane as qml
from pennylane.operation import (
    has_gen,
    has_grad_method,
    has_nopar,
    has_unitary_gen,
    is_measurement,
    is_trainable,
    not_tape,
)


def create_expand_fn(depth, stop_at, docstring=None):
    """Create a function for expanding a tape to a given depth, and
    with a specific stopping criterion. This is a wrapper around
    :meth:`~.QuantumTape.expand`.

    Args:
        depth (int): Depth for the expansion
        stop_at (callable): Stopping criterion. This must be a function with signature
            ``stop_at(obj)``, where ``obj`` is a *queueable* PennyLane object such as
            :class:`~.Operation` or :class:`~.MeasurementProcess`. It must return a
            boolean, indicating if the expansion should stop at this object.
        docstring (str): docstring for the generated expansion function

    Returns:
        callable: Tape expansion function. The returned function accepts a :class:`~.QuantumTape`,
        and returns an expanded :class:`~.QuantumTape`.

    **Example**

    Let us construct an expansion function that expands a tape in order to
    decompose trainable multi-parameter gates. We allow for up to five expansion
    steps, which can be controlled with the argument ``depth``.
    The stopping criterion is easy to write as

    >>> stop_at = ~(qml.operation.has_multipar & qml.operation.is_trainable)

    Then the expansion function can be obtained via

    >>> expand_fn = qml.transforms.create_expand_fn(depth=5, stop_at=stop_at)

    We can test the newly generated function on an example tape:

    .. code-block:: python

        with qml.tape.JacobianTape() as tape:
            qml.RX(0.2, wires=0)
            qml.RX(qml.numpy.array(-2.4, requires_grad=True), wires=1)
            qml.Rot(1.7, 0.92, -1.1, wires=0)
            qml.Rot(*qml.numpy.array([-3.1, 0.73, 1.36], requires_grad=True), wires=1)

    >>> new_tape = expand_fn(tape)
    >>> print(tape.draw())
     0: ──RX(0.2)───Rot(1.7, 0.92, -1.1)───┤
     1: ──RX(-2.4)──Rot(-3.1, 0.73, 1.36)──┤
    >>> print(new_tape.draw())
     0: ──RX(0.2)───Rot(1.7, 0.92, -1.1)──────────────────────┤
     1: ──RX(-2.4)──RZ(-3.1)──────────────RY(0.73)──RZ(1.36)──┤

    """
    # pylint: disable=unused-argument

    def expand_fn(tape, _depth=depth, **kwargs):
        if not all(stop_at(op) for op in tape.operations):

            with qml.tape.stop_recording():
                tape = tape.expand(depth=_depth, stop_at=stop_at)

            params = tape.get_parameters(trainable_only=False)
            tape.trainable_params = qml.math.get_trainable_indices(params)
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

expand_multipar = create_expand_fn(
    depth=10,
    stop_at=not_tape | is_measurement | has_nopar | has_gen,
    docstring=_expand_multipar_doc,
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

expand_nonunitary_gen = create_expand_fn(
    depth=10,
    stop_at=not_tape | is_measurement | has_nopar | (has_gen & has_unitary_gen),
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

expand_invalid_trainable = create_expand_fn(
    depth=10,
    stop_at=not_tape | is_measurement | (~is_trainable) | has_grad_method,
    docstring=_expand_invalid_trainable_doc,
)
